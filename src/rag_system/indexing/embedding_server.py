"""Loopback-only EmbeddingGemma 2 service for the direct voice pipeline."""
from __future__ import annotations

import argparse
import asyncio
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
MODEL_DIR = Path(__file__).resolve().parents[3] / "data/models/embeddinggemma-2"


class EmbeddingRequest(BaseModel):
    model: Literal["google/embeddinggemma-2"] = MODEL_ID
    input: str | list[str]
    dimensions: Literal[768] = 768
    encoding_format: Literal["float"] = "float"
    prompt_name: Literal["Document", "SearchQuery"] = "Document"


def load_model(model_dir: Path) -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = SentenceTransformer(
        str(model_dir), device=device, local_files_only=True,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": dtype},
    )
    model.encode(["检索预热"], prompt_name="SearchQuery", normalize_embeddings=True, show_progress_bar=False)
    return model


def create_app(model_dir: Path = MODEL_DIR, *, model_factory: Callable[[], Any] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.encoder = await asyncio.to_thread(model_factory or (lambda: load_model(model_dir)))
        app.state.encode_lock = asyncio.Lock()
        yield
        app.state.encoder = None

    app = FastAPI(title="Green River EmbeddingGemma 2", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        if getattr(app.state, "encoder", None) is None:
            raise HTTPException(503, "Embedding model is not ready")
        return {"status": "ready", "model": MODEL_ID, "dimensions": 768,
                "device": str(getattr(app.state.encoder, "device", "unknown"))}

    @app.post("/v1/embeddings")
    async def embed(payload: EmbeddingRequest) -> dict[str, Any]:
        texts = [payload.input] if isinstance(payload.input, str) else payload.input
        if not 1 <= len(texts) <= 16 or any(not text.strip() or len(text) > 64000 for text in texts):
            raise HTTPException(400, "Provide 1–16 nonempty texts of at most 64000 characters")
        if getattr(app.state, "encoder", None) is None:
            raise HTTPException(503, "Embedding model is not ready")
        async with app.state.encode_lock:
            vectors = await asyncio.to_thread(
                app.state.encoder.encode, texts, prompt_name=payload.prompt_name,
                normalize_embeddings=True, show_progress_bar=False,
            )
        values = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        if len(values) != len(texts) or any(len(row) != 768 or
                not all(math.isfinite(float(v)) for v in row) or
                not math.isclose(sum(float(v) ** 2 for v in row), 1, abs_tol=0.01) for row in values):
            raise HTTPException(500, "Embedding model produced invalid vectors")
        return {"object": "list", "model": MODEL_ID, "data": [
            {"object": "embedding", "index": i, "embedding": row} for i, row in enumerate(values)
        ]}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(args.model_dir))
        return
    import uvicorn
    uvicorn.run(create_app(args.model_dir), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
