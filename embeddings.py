from __future__ import annotations

from typing import Iterable, List, Sequence
import torch
from sentence_transformers import SentenceTransformer

from config import EmbeddingConfig
from utils import chunked, timeit, logger


def _resolve_device(cfg: EmbeddingConfig) -> str | None:
    if cfg.device:
        return cfg.device
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_model(cfg: EmbeddingConfig | None = None) -> SentenceTransformer:
    ecfg = cfg or EmbeddingConfig()
    model = SentenceTransformer(ecfg.model_name, device=_resolve_device(ecfg))
    return model


@timeit
def encode_sentences(sentences: Sequence[str], cfg: EmbeddingConfig | None = None, model: SentenceTransformer | None = None) -> torch.Tensor:
    ecfg = cfg or EmbeddingConfig()
    m = model or get_model(ecfg)

    if len(sentences) == 0:
        return torch.empty((0, 0))

    batches: List[torch.Tensor] = []
    for batch in chunked(list(sentences), ecfg.batch_size):
        logger.debug(f"encoding batch size={len(batch)}")
        emb = m.encode(batch, convert_to_tensor=True, device=_resolve_device(ecfg))
        batches.append(emb)
    return torch.cat(batches, dim=0) if len(batches) > 1 else batches[0]


def compute_central_embedding(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.mean(embeddings, dim=0)
