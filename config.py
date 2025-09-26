from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


class ThresholdStrategy(Enum):
    STAT_PERCENTILE_BLEND = auto()
    PURE_PERCENTILE = auto()
    Z_SCORE = auto()


class RelevanceStrategy(Enum):
    COSINE_TO_CENTROID = auto()
    AVERAGE_PAIRWISE = auto()


@dataclass
class PreprocessConfig:
    lowercase: bool = True
    remove_stopwords: bool = True
    remove_non_word: bool = True


@dataclass
class EmbeddingConfig:
    model_name: str = "paraphrase-MiniLM-L6-v2"
    batch_size: int = 16
    device: str | None = None  # e.g., "cpu", "cuda"


@dataclass
class ThresholdConfig:
    strategy: ThresholdStrategy = ThresholdStrategy.STAT_PERCENTILE_BLEND
    percentile: float = 10.0
    z_k: float = 1.5


@dataclass
class RelevanceConfig:
    strategy: RelevanceStrategy = RelevanceStrategy.COSINE_TO_CENTROID


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    relevance: RelevanceConfig = field(default_factory=RelevanceConfig)
