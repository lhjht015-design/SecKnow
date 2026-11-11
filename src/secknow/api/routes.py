from __future__ import annotations

import hashlib
import sqlite3
import uuid
from pathlib import Path as FileSystemPath
from threading import RLock
from typing import Annotated, Any

import fitz
from fastapi import APIRouter, File, Form, HTTPException, Path, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse

from secknow.answer.evidence import EvidenceBuilder
from secknow.answer.llm import LocalLlmClient
from secknow.api.schemas import (
    IndexRequest,
    JobAccepted,
    LlmConfigRequest,
    SearchRequest,
    SettingsDependency,
    SnapshotPublishRequest,
    SnapshotRollbackRequest,
    StoreDependency,
)
from secknow.api.security import media_type_for, parse_json_fields, resolve_corpus_path
from secknow.config import RuntimeMode, Settings
from secknow.ingestion.discovery import MAGIC, SUPPORTED_FORMATS
from secknow.models.bge import model_directory_ready
from secknow.retrieval.service import RetrievalService
from secknow.runtime import capability_report
from secknow.sources.manifest import load_manifest
from secknow.sources.snapshot import SnapshotCatalog
from secknow.stores.milvus import MilvusStore
from secknow.stores.neo4j import Neo4jStore

router = APIRouter(prefix="/api")


@router.get("/health", tags=["system"])
def health(settings: SettingsDependency, store: StoreDependency) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {
        "sqlite": {"status": "ok", "path": str(settings.database_path)},
        "tfidf": {"status": "ok" if store.get_setting("active_tfidf_version") else "not-ready"},
        "graph": {"status": "ok" if store.get_setting("active_graph_version") else "not-ready"},
        "bge": {
            "status": "installed"
            if model_directory_ready(settings.bge_model_path)
            else "not-installed"
        },
    }
    if settings.infrastructure_enabled:
        milvus = MilvusStore(
            settings.milvus_uri,
            settings.milvus_token.get_secret_value() if settings.milvus_token else None,
        )
        neo4j = Neo4jStore(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password.get_secret_value(),
        )
        components["milvus"] = {"status": "ok" if milvus.healthy() else "unavailable"}
        components["neo4j"] = {"status": "ok" if neo4j.healthy() else "unavailable"}
        neo4j.close()
    else:
        components["milvus"] = {"status": "disabled"}
        components["neo4j"] = {"status": "disabled"}
    return {
        "status": "ok",
        "mode": settings.mode.value,
        "offline": settings.offline,
        "components": components,
    }


@router.get("/config", tags=["system"])
def safe_config(settings: SettingsDependency, store: StoreDependency) -> dict[str, Any]:
    milvus_ready: bool | None = None
    milvus_status = "disabled"
    if settings.infrastructure_enabled:
        milvus = MilvusStore(
            settings.milvus_uri,
            settings.milvus_token.get_secret_value() if settings.milvus_token else None,
        )
        milvus_ready = milvus.healthy()
        milvus_status = "ok" if milvus_ready else "unavailable"
    report = capability_report(
        settings,
        tfidf_ready=bool(store.get_setting("active_tfidf_version")),
        graph_ready=bool(store.get_setting("active_graph_version")),
        milvus_ready=milvus_ready,
    )
    return {
        "mode": settings.mode.value,
        "offline": settings.offline,
        "semantic_enabled": settings.semantic_enabled,
        "generative_enabled": settings.generative_enabled,
        "reranker_enabled": settings.reranker_enabled,
        "corpus_root": str(settings.corpus_root),
        "result_top_k": settings.result_top_k,
        "retrieval_top_k": settings.retrieval_top_k,
        "reranker_top_k": settings.reranker_top_k,
        "max_upload_mb": settings.max_upload_mb,
        "milvus_service": milvus_status,
        **report,
    }


@router.get("/sources", tags=["sources"])
def sources(settings: SettingsDependency) -> dict[str, Any]:
    try:
        manifest = load_manifest(settings.source_manifest)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail="source manifest is unavailable") from error
    catalog = SnapshotCatalog(settings.snapshot_root)
    entries: list[dict[str, Any]] = []
    for source in manifest.sources:
        current = catalog.current(source.id)
        entries.append(
            {
                "id": source.id,
                "name": source.name,
                "description": source.description,
                "adapter": source.adapter,
                "risk_state": source.risk_state.value,
                "license": source.license.model_dump(mode="json"),
                "current_snapshot": current.model_dump(mode="json") if current else None,
            }
        )
    return {"version": manifest.version, "sources": entries}


@router.post("/sources/snapshots/{snapshot_id}/publish", tags=["sources"])
def publish_snapshot(
    snapshot_id: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")],
    request: SnapshotPublishRequest,
    settings: SettingsDependency,
) -> dict[str, Any]:
    try:
        metadata = SnapshotCatalog(settings.snapshot_root).publish(
            snapshot_id, allow_quarantine=request.allow_quarantine
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="snapshot not found") from error
    except PermissionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=409, detail="snapshot cannot be published") from error
    return metadata.model_dump(mode="json")


@router.post("/sources/{source_id}/rollback", tags=["sources"])
def rollback_snapshot(
    source_id: Annotated[
        str,
        Path(min_length=1, max_length=64, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$"),
    ],
    request: SnapshotRollbackRequest,
    settings: SettingsDependency,
) -> dict[str, Any]:
    try:
        metadata = SnapshotCatalog(settings.snapshot_root).rollback(
            source_id, request.snapshot_id
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="source has no published snapshots") from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="rollback snapshot is missing") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return metadata.model_dump(mode="json")


@router.get("/documents", tags=["documents"])
def documents(
    store: StoreDependency,
    active_only: bool = Query(default=True),
) -> list[dict[str, Any]]:
    return [parse_json_fields(row) for row in store.list_documents(active_only)]


@router.post("/documents/upload", status_code=201, tags=["documents"])
async def upload_documents(
    settings: SettingsDependency,
    store: StoreDependency,
    files: Annotated[list[UploadFile], File(description="Documents to add to the corpus")],
    auto_index: Annotated[bool, Form()] = True,
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="at least one file is required")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="at most 20 files can be uploaded at once")

    uploads_root = settings.corpus_root.resolve() / "uploads"
    uploads_root.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[FileSystemPath, str, str, int]] = []
    created_temporary: list[FileSystemPath] = []
    max_bytes = settings.max_upload_mb * 1024 * 1024
    try:
        for upload in files:
            safe_name = _safe_upload_name(upload.filename or "")
            suffix = FileSystemPath(safe_name).suffix.casefold()
            if suffix not in SUPPORTED_FORMATS:
                supported = ", ".join(sorted(SUPPORTED_FORMATS))
                raise HTTPException(
                    status_code=415,
                    detail=f"unsupported document format; allowed: {supported}",
                )
            temporary = uploads_root / f".incoming-{uuid.uuid4().hex}.partial"
            created_temporary.append(temporary)
            digest = hashlib.sha256()
            size = 0
            with temporary.open("xb") as stream:
                while content := await upload.read(1024 * 1024):
                    size += len(content)
                    if size > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{safe_name} exceeds the {settings.max_upload_mb} MiB limit",
                        )
                    digest.update(content)
                    stream.write(content)
            if size == 0:
                raise HTTPException(status_code=400, detail=f"{safe_name} is empty")
            _verify_upload_signature(temporary, suffix)
            staged.append((temporary, safe_name, digest.hexdigest(), size))

        uploaded: list[dict[str, Any]] = []
        relative_paths: list[str] = []
        for temporary, safe_name, digest_hex, size in staged:
            target = uploads_root / f"{digest_hex}-{safe_name}"
            if target.exists():
                if _file_sha256(target) != digest_hex:
                    raise HTTPException(status_code=409, detail="upload hash collision detected")
                temporary.unlink(missing_ok=True)
            else:
                temporary.replace(target)
            relative = target.relative_to(settings.corpus_root.resolve()).as_posix()
            relative_paths.append(relative)
            uploaded.append(
                {
                    "name": safe_name,
                    "relative_path": relative,
                    "format": FileSystemPath(safe_name).suffix.casefold().lstrip("."),
                    "size_bytes": size,
                    "sha256": digest_hex,
                }
            )

        job_id: str | None = None
        if auto_index:
            job_id = store.create_job(
                f"job-{uuid.uuid4()}",
                "ingest",
                {"paths": relative_paths, "recursive": False},
            )
        return {"files": uploaded, "job_id": job_id, "status": "queued" if job_id else "stored"}
    finally:
        for upload in files:
            await upload.close()
        for temporary, *_ in staged:
            temporary.unlink(missing_ok=True)
        for temporary in created_temporary:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                continue


@router.get("/documents/{source_id}/pages", tags=["documents"])
def document_pages(
    source_id: str,
    store: StoreDependency,
    attention_only: bool = Query(default=False),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    audit = store.document_page_audit(
        source_id,
        attention_only=attention_only,
        offset=offset,
        limit=limit,
    )
    if audit is None:
        raise HTTPException(status_code=404, detail="document not found")
    return audit


@router.post("/index", response_model=JobAccepted, status_code=202, tags=["jobs"])
def enqueue_index(
    request: IndexRequest,
    settings: SettingsDependency,
    store: StoreDependency,
) -> JobAccepted:
    path = resolve_corpus_path(settings, request.path)
    relative = path.relative_to(settings.corpus_root.resolve()).as_posix() or "."
    job_id = store.create_job(
        f"job-{uuid.uuid4()}",
        "ingest",
        {"path": relative, "recursive": request.recursive},
    )
    return JobAccepted(job_id=job_id)


@router.get("/jobs/{job_id}", tags=["jobs"])
def job(job_id: str, store: StoreDependency) -> dict[str, Any]:
    row = store.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return parse_json_fields(row)


@router.post("/search", tags=["search"])
def search(
    search_request: SearchRequest,
    request: Request,
    settings: SettingsDependency,
) -> dict[str, Any]:
    request_settings = _settings_for_search(settings, search_request.mode)
    service, service_lock = _retrieval_runtime(request, request_settings)
    try:
        with service_lock:
            response = service.search(search_request.query, search_request.top_k)
    except (ValueError, sqlite3.OperationalError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    evidence = EvidenceBuilder().build(response)
    payload = response.model_dump(mode="json")
    payload["answer"] = evidence.model_dump(mode="json")
    payload["generated_answer"] = None
    if not search_request.use_llm:
        return payload
    if not request_settings.generative_enabled:
        raise HTTPException(status_code=409, detail="local LLM generation is disabled")
    try:
        payload["generated_answer"] = LocalLlmClient(request_settings).generate(evidence)
    except (RuntimeError, ValueError, OSError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return payload


@router.get("/llm/config", tags=["system"])
def llm_status(settings: SettingsDependency, store: StoreDependency) -> dict[str, Any]:
    saved = store.get_setting("llm_config", {})
    return {
        "enabled": bool(saved.get("enabled", settings.llm_enabled)),
        "base_url": str(saved.get("base_url", settings.llm_base_url)),
        "model": str(saved.get("model", settings.llm_model)),
        "api_key_configured": bool(
            store.get_setting("llm_api_key_configured", False)
            or settings.llm_api_key.get_secret_value()
        ),
        "active": settings.generative_enabled,
        "cloud_endpoints_allowed": False,
    }


@router.put("/llm/config", tags=["system"])
def save_llm_config(
    request: LlmConfigRequest,
    settings: SettingsDependency,
    store: StoreDependency,
) -> dict[str, Any]:
    candidate = settings.model_copy(
        update={
            "llm_enabled": request.enabled,
            "llm_base_url": request.base_url,
            "llm_model": request.model,
        }
    )
    try:
        LocalLlmClient(candidate).validate_endpoint()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    store.set_setting(
        "llm_config",
        {"enabled": request.enabled, "base_url": request.base_url, "model": request.model},
    )
    if request.api_key is not None:
        store.set_setting("llm_api_key", request.api_key)
        store.set_setting("llm_api_key_configured", bool(request.api_key))
    return llm_status(settings, store)


@router.get("/files/{relative_path:path}", tags=["documents"])
def source_file(relative_path: str, settings: SettingsDependency) -> FileResponse:
    path = resolve_corpus_path(settings, relative_path, require_file=True)
    if path.suffix.casefold() != ".pdf":
        raise HTTPException(status_code=415, detail="only PDF preview is supported")
    return FileResponse(path, media_type=media_type_for(path), filename=path.name)


@router.get("/pages/{relative_path:path}", tags=["documents"])
def source_page_image(
    relative_path: str,
    settings: SettingsDependency,
    page: int = Query(ge=1, le=5000),
    dpi: int = Query(default=144, ge=72, le=240),
) -> Response:
    path = resolve_corpus_path(settings, relative_path, require_file=True)
    if path.suffix.casefold() != ".pdf":
        raise HTTPException(status_code=415, detail="only PDF page preview is supported")
    try:
        with fitz.open(path) as pdf:
            if page > len(pdf):
                raise HTTPException(status_code=404, detail="PDF page not found")
            scale = dpi / 72
            pixmap = pdf[page - 1].get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
            )
            image = pixmap.tobytes("png")
    except HTTPException:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="PDF page cannot be rendered") from error
    return Response(
        content=image,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


def _settings_for_search(settings: Settings, mode: RuntimeMode | None) -> Settings:
    if mode is None or mode is settings.mode:
        return settings
    capability = {
        RuntimeMode.STRICT_LEXICAL: 0,
        RuntimeMode.COMPETITION: 1,
        RuntimeMode.ENHANCED: 2,
        RuntimeMode.LLM: 3,
    }
    if capability[mode] > capability[settings.mode]:
        return settings
    return settings.model_copy(
        update={
            "mode": mode,
            "reranker_enabled": settings.reranker_enabled
            and mode in {RuntimeMode.ENHANCED, RuntimeMode.LLM},
            "llm_enabled": settings.llm_enabled and mode is RuntimeMode.LLM,
        }
    )


def _retrieval_runtime(
    request: Request,
    settings: Settings,
) -> tuple[RetrievalService, RLock]:
    key = (
        settings.mode.value,
        settings.reranker_enabled,
        str(settings.database_path),
        str(settings.bge_model_path),
        str(settings.reranker_model_path),
    )
    state = request.app.state
    cache_lock: RLock = state.retrieval_cache_lock
    with cache_lock:
        service = state.retrieval_services.get(key)
        service_lock = state.retrieval_locks.get(key)
        if service is None or service_lock is None:
            service = RetrievalService(settings)
            service_lock = RLock()
            state.retrieval_services[key] = service
            state.retrieval_locks[key] = service_lock
    return service, service_lock


def _safe_upload_name(filename: str) -> str:
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1].strip().rstrip(". ")
    basename = "".join(
        character
        for character in basename
        if ord(character) >= 32 and character not in '<>:"/\\|?*'
    )
    if not basename or basename in {".", ".."}:
        raise HTTPException(status_code=400, detail="upload filename is invalid")
    path = FileSystemPath(basename)
    stem = path.stem[:120].strip().rstrip(". ")
    suffix = path.suffix.casefold()
    if not stem:
        raise HTTPException(status_code=400, detail="upload filename is invalid")
    if stem.casefold() in {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }:
        stem = f"document-{stem}"
    return f"{stem}{suffix}"


def _verify_upload_signature(path: FileSystemPath, suffix: str) -> None:
    expected = MAGIC.get(suffix)
    if expected is None:
        return
    with path.open("rb") as stream:
        header = stream.read(len(expected))
    if header != expected:
        raise HTTPException(
            status_code=415,
            detail=f"file content does not match the {suffix} extension",
        )


def _file_sha256(path: FileSystemPath) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
