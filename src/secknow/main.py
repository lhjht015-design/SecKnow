from __future__ import annotations

import uvicorn

from secknow.api.app import create_app
from secknow.config import get_settings

app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "secknow.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    run()
