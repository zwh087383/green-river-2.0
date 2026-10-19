from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("RAG_SYSTEM_HOST", "127.0.0.1")
    port = int(os.environ.get("RAG_SYSTEM_PORT", "8000"))
    # The kiosk page polls display state regularly; access logging would grow
    # without bound during an unattended deployment.
    uvicorn.run(
        "rag_system.web.app:app",
        host=host,
        port=port,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
