from __future__ import annotations

from typing import Iterable, List
import numpy as np

from config import ThresholdConfig, ThresholdStrategy
from utils import timeit


def _stat_percentile_blend(scores: List[float], percentile: float, k: float) -> float:
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    stat = mean - k * std
    perc = float(np.percentile(scores, percentile))
    return (stat + perc) / 2.0


def _pure_percentile(scores: List[float], percentile: float) -> float:
    return float(np.percentile(scores, percentile))


def _z_score(scores: List[float], k: float) -> float:
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    return mean - k * std


@timeit
def calculate_threshold(similarity_scores: List[float], text_length: int, cfg: ThresholdConfig | None = None) -> float:
    if len(similarity_scores) == 0:
        return 0.0
    tcfg = cfg or ThresholdConfig()

    base: float
    if tcfg.strategy == ThresholdStrategy.STAT_PERCENTILE_BLEND:
        base = _stat_percentile_blend(similarity_scores, tcfg.percentile, tcfg.z_k)
    elif tcfg.strategy == ThresholdStrategy.PURE_PERCENTILE:
        base = _pure_percentile(similarity_scores, tcfg.percentile)
    elif tcfg.strategy == ThresholdStrategy.Z_SCORE:
        base = _z_score(similarity_scores, tcfg.z_k)
    else:
        raise ValueError(f"Unsupported threshold strategy: {tcfg.strategy}")

    length_factor = min(1.0, text_length / 1000.0)
    blended = base * (1.0 - length_factor) + _pure_percentile(similarity_scores, tcfg.percentile) * length_factor
    return float(blended)
