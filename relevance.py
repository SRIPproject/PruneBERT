from __future__ import annotations

from typing import List
import numpy as np
import torch
from sentence_transformers import util

from config import RelevanceConfig, RelevanceStrategy
from utils import timeit, logger


def _cosine_to_centroid(embeddings: torch.Tensor, centroid: torch.Tensor) -> List[float]:
    sims = util.pytorch_cos_sim(embeddings, centroid.unsqueeze(0)).squeeze(1)
    return sims.detach().cpu().numpy().tolist()


def _average_pairwise(embeddings: torch.Tensor) -> List[float]:
    # pairwise cosine similarity matrix
    sims = util.pytorch_cos_sim(embeddings, embeddings)
    n = sims.shape[0]
    mask = torch.ones_like(sims, dtype=torch.bool)
    mask.fill_diagonal_(False)
    sums = (sims * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1)
    means = sums / counts
    return means.detach().cpu().numpy().tolist()


@timeit
def compute_similarity_scores(
    embeddings: torch.Tensor,
    central_embedding: torch.Tensor | None,
    cfg: RelevanceConfig | None = None,
) -> List[float]:
    rcfg = cfg or RelevanceConfig()
    if rcfg.strategy == RelevanceStrategy.COSINE_TO_CENTROID:
        assert central_embedding is not None, "central_embedding required for COSINE_TO_CENTROID"
        return _cosine_to_centroid(embeddings, central_embedding)
    if rcfg.strategy == RelevanceStrategy.AVERAGE_PAIRWISE:
        return _average_pairwise(embeddings)
    raise ValueError(f"Unsupported strategy: {rcfg.strategy}")


def identify_irrelevant_sentences(sentences: List[str], similarity_scores: List[float], threshold: float) -> List[str]:
    if len(sentences) <= 3:
        min_index = int(np.argmin(np.array(similarity_scores)))
        return [sentences[min_index]]
    return [s for s, sc in zip(sentences, similarity_scores) if sc < threshold]
