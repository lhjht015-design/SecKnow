from __future__ import annotations

from importlib.util import find_spec
from typing import Any

from secknow.config import Settings
from secknow.ingestion.document_parser import select_ocr_engine
from secknow.models.bge import model_directory_ready


def package_available(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def accelerator_report() -> dict[str, Any]:
    """真实探测加速器是否可用。

    为什么要单独做这件事：本项目的检索链路曾经整整跑在 CPU 上——
    `torch` 装成了 CPU-only 构建，`torch.cuda.is_available()` 恒为 False，
    而 `doctor` 只看「模型目录是否存在」，所以报告一直是绿的。
    结果是 BGE 编码与重排慢了十几倍却毫无提示（见 laws/LAW-0004）。
    「权重就位」不等于「后端可用」不等于「加速器在用」，三者必须分别如实上报。
    """
    probe: dict[str, Any] = {"torch_installed": False, "cuda_available": False, "device": None}
    try:
        import torch
    except ImportError:
        return probe
    probe["torch_installed"] = True
    probe["torch_version"] = torch.__version__
    probe["cuda_build"] = bool(torch.version.cuda)
    try:
        if torch.cuda.is_available():
            probe["cuda_available"] = True
            probe["device"] = torch.cuda.get_device_name(0)
    except (RuntimeError, OSError, AssertionError):
        pass
    return probe


def capability_report(
    settings: Settings,
    *,
    tfidf_ready: bool = False,
    graph_ready: bool = False,
    milvus_ready: bool | None = None,
) -> dict[str, Any]:
    bge_ready = package_available("FlagEmbedding") and model_directory_ready(
        settings.bge_model_path
    )
    reranker_ready = package_available("FlagEmbedding") and model_directory_ready(
        settings.reranker_model_path
    )
    ocr = select_ocr_engine(
        settings.ocr_backend,
        languages=settings.ocr_languages,
        executable=settings.tesseract_executable,
        tessdata_dir=settings.tesseract_tessdata_dir,
    )
    capabilities = {
        "llamaindex": package_available("llama_index"),
        "unstructured": package_available("unstructured"),
        "tree_sitter": package_available("tree_sitter_language_pack"),
        "bge_m3": bge_ready,
        "reranker": reranker_ready,
        "ocr": ocr is not None,
        "ocr_backend": type(ocr).__name__ if ocr is not None else None,
        "rapidocr": package_available("rapidocr_onnxruntime"),
        "tesseract": select_ocr_engine(
            "tesseract",
            languages=settings.ocr_languages,
            executable=settings.tesseract_executable,
            tessdata_dir=settings.tesseract_tessdata_dir,
        )
        is not None,
        "milvus_driver": package_available("pymilvus"),
        "neo4j_driver": package_available("neo4j"),
        "tfidf": tfidf_ready,
        "deterministic_reasoning": graph_ready,
    }
    semantic_stack_installed = bool(
        settings.semantic_enabled and bge_ready and capabilities["milvus_driver"]
    )
    effective_semantic_enabled = (
        semantic_stack_installed
        if milvus_ready is None
        else semantic_stack_installed and milvus_ready
    )
    return {
        "capabilities": capabilities,
        "accelerator": accelerator_report(),
        "semantic_stack_installed": semantic_stack_installed,
        "effective_semantic_enabled": effective_semantic_enabled,
        "effective_reranker_enabled": bool(settings.reranker_enabled and reranker_ready),
        "deterministic_reasoning_enabled": graph_ready,
    }
