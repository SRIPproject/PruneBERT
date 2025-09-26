from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

from utils import memoize, timeit, logger
from config import PreprocessConfig


@dataclass(frozen=True)
class _Stage:
    name: str
    func: Callable[[str], str]


def ensure_nltk_resources() -> None:
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    try:
        word_tokenize("test")
    except LookupError:
        nltk.download("punkt")


@memoize
def _stopwords() -> set[str]:
    ensure_nltk_resources()
    return set(stopwords.words("english"))


def _lowercase(text: str) -> str:
    return text.lower()


def _remove_non_word(text: str) -> str:
    return re.sub(r"\W", " ", text)


def _tokenize(text: str) -> List[str]:
    return word_tokenize(text)


def _remove_stopwords(tokens: List[str]) -> List[str]:
    sw = _stopwords()
    return [t for t in tokens if t not in sw]


def _detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


@timeit
def preprocess_text(text: str, config: PreprocessConfig | None = None) -> str:
    cfg = config or PreprocessConfig()
    stages: List[_Stage] = []

    s = text
    if cfg.remove_non_word:
        stages.append(_Stage("remove_non_word", _remove_non_word))
    if cfg.lowercase:
        stages.append(_Stage("lowercase", _lowercase))

    for st in stages:
        s = st.func(s)
        logger.debug(f"preprocess stage {st.name}: {s[:60]}")

    tokens = _tokenize(s)
    if cfg.remove_stopwords:
        tokens = _remove_stopwords(tokens)

    out = _detokenize(tokens)
    logger.debug(f"preprocess output: {out[:60]}")
    return out
