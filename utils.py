from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")
R = TypeVar("R")


# Configure a verbose logger for the repository
logger = logging.getLogger("prunebert")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def timeit(func: Callable[..., R]) -> Callable[..., R]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.debug(f"{func.__name__} executed in {elapsed_ms:.2f} ms")
    return wrapper


def retry(times: int = 3, delay: float = 0.1) -> Callable[[Callable[..., R]], Callable[..., R]]:
    def deco(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            last_err: Optional[BaseException] = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except BaseException as e:  # noqa: BLE001
                    last_err = e
                    logger.warning(f"Attempt {attempt}/{times} failed for {func.__name__}: {e}")
                    time.sleep(delay)
            assert last_err is not None
            raise last_err
        return wrapper
    return deco


def chunked(seq: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size <= 0:
        raise ValueError("size must be > 0")
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@dataclass(frozen=True)
class Stopwatch:
    start_ns: int

    @classmethod
    def start(cls) -> "Stopwatch":
        return cls(time.perf_counter_ns())

    def elapsed_ms(self) -> float:
        return (time.perf_counter_ns() - self.start_ns) / 1_000_000.0


def memoize(func: Callable[..., R]) -> Callable[..., R]:
    cache: dict[Tuple[Any, ...], R] = {}

    @functools.wraps(func)
    def wrapper(*args: Any) -> R:
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return wrapper
