from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Selection:
    name: str
    indices: np.ndarray


def validate_scores(scores: np.ndarray, dataset_size: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("attribution scores must be one-dimensional")
    if len(scores) != dataset_size:
        raise ValueError(
            f"attribution count {len(scores)} does not match dataset size {dataset_size}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("attribution scores contain NaN or infinite values")
    return scores


def deciles(scores: np.ndarray, divisions: int = 10) -> list[Selection]:
    if divisions < 2:
        raise ValueError("divisions must be at least 2")
    scores = validate_scores(scores, len(scores))
    ordered = np.argsort(scores, kind="stable")[::-1]
    boundaries = np.linspace(0, len(scores), divisions + 1, dtype=int)
    return [
        Selection(
            name=f"decile_{index:02d}",
            indices=ordered[boundaries[index] : boundaries[index + 1]],
        )
        for index in range(divisions)
    ]


def extreme(
    scores: np.ndarray,
    *,
    fraction: float,
    side: Literal["top", "bottom"],
) -> Selection:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    scores = validate_scores(scores, len(scores))
    count = max(1, int(round(len(scores) * fraction)))
    ordered = np.argsort(scores, kind="stable")
    indices = ordered[-count:] if side == "top" else ordered[:count]
    return Selection(name=f"{side}_{fraction:g}", indices=indices)


def random_subset(dataset_size: int, *, fraction: float, seed: int) -> Selection:
    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    count = max(1, int(round(dataset_size * fraction)))
    rng = np.random.default_rng(seed)
    return Selection(
        name=f"random_{fraction:g}_seed_{seed}",
        indices=rng.choice(dataset_size, size=count, replace=False),
    )


def complement(dataset_size: int, removed: np.ndarray) -> np.ndarray:
    removed = np.asarray(removed, dtype=int)
    if ((removed < 0) | (removed >= dataset_size)).any():
        raise ValueError("removed indices fall outside the dataset")
    return np.setdiff1d(np.arange(dataset_size), np.unique(removed))


def resample(indices: np.ndarray, *, target_size: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        raise ValueError("cannot resample an empty selection")
    if target_size < 1:
        raise ValueError("target_size must be positive")
    rng = np.random.default_rng(seed)
    if target_size <= len(indices):
        return rng.choice(indices, size=target_size, replace=False)
    extra = rng.choice(indices, size=target_size - len(indices), replace=True)
    result = np.concatenate([indices, extra])
    rng.shuffle(result)
    return result

