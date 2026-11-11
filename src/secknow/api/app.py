from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from secknow.api.graph import router as graph_router
from secknow.api.routes import router as api_router
from secknow.config import Settings, get_settings
from secknow.stores.metadata import MetadataStore


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    settings.ensure_directories()
    MetadataStore(settings.database_path).initialize()
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or get_settings()
    app = FastAPI(
        title="SecKnow",
        description="Offline CTF knowledge retrieval and graph reasoning API",
        version="0.1.0",
        lifespan=_lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = configured
    app.state.retrieval_services = {}
    app.state.retrieval_locks = {}
    app.state.retrieval_cache_lock = RLock()
    origins = [
        f"http://127.0.0.1:{configured.port}",
        f"http://localhost:{configured.port}",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Content-Type", "Accept"],
    )
    app.include_router(api_router)
    app.include_router(graph_router)
    _mount_frontend(app, configured.frontend_dist)
    return app


def _mount_frontend(app: FastAPI, dist: Path) -> None:
    if not dist.is_dir():
        return
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")
    index = dist / "index.html"
    if not index.is_file():
        return

    @app.get("/", include_in_schema=False)
    def frontend_index() -> FileResponse:
        return FileResponse(index)

    @app.get("/{frontend_path:path}", include_in_schema=False)
    def frontend_fallback(frontend_path: str) -> FileResponse:
        candidate = (dist / frontend_path).resolve()
        try:
            candidate.relative_to(dist.resolve())
        except ValueError:
            return FileResponse(index)
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)
