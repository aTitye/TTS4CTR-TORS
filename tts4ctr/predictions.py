"""Prediction-bank loading, pooling, and selection utilities."""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .metrics import auc_score, logloss_score, pool_logits, pool_probabilities, sigmoid


@dataclass(frozen=True)
class PredictionBank:
    name: str
    labels_path: Path
    prediction_paths: tuple[tuple[str, Path], ...]

    @property
    def member_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.prediction_paths)


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return (base_seed + zlib.crc32(payload)) % (2**32 - 1)


def load_labels(path: str | Path) -> np.ndarray:
    return np.asarray(np.load(path, mmap_mode="r")).reshape(-1).astype(np.float64, copy=False)


def load_predictions(
    bank: PredictionBank, members: Sequence[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    wanted = set(members) if members is not None else None
    columns = []
    names = []
    for name, path in bank.prediction_paths:
        if wanted is not None and name not in wanted:
            continue
        columns.append(np.asarray(np.load(path, mmap_mode="r")).reshape(-1).astype(np.float64, copy=False))
        names.append(name)
    if not columns:
        raise ValueError("no requested prediction members were found")
    expected = load_labels(bank.labels_path).shape[0]
    if any(column.shape[0] != expected for column in columns):
        raise ValueError("prediction and label lengths differ")
    return np.stack(columns, axis=1), load_labels(bank.labels_path), names


def to_logits(values: np.ndarray) -> np.ndarray:
    """Convert probability columns to stable reconstructed logits."""
    clipped = np.clip(values, 1e-7, 1 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def pairwise_js_divergence(probabilities: np.ndarray, bins: int = 100) -> float:
    """Mean symmetric JS divergence (natural log, range [0, ln 2]) for a member bank."""
    member_count = probabilities.shape[1]
    values = []
    for left, right in combinations(range(member_count), 2):
        edges = np.linspace(0.0, 1.0, bins + 1)
        left_hist = np.histogram(probabilities[:, left], bins=edges, density=False)[0].astype(float)
        right_hist = np.histogram(probabilities[:, right], bins=edges, density=False)[0].astype(float)
        left_hist /= max(left_hist.sum(), 1)
        right_hist /= max(right_hist.sum(), 1)
        mixture = 0.5 * (left_hist + right_hist)
        left_nonzero = left_hist > 0
        right_nonzero = right_hist > 0
        kl_left = float(np.sum(left_hist[left_nonzero] * np.log(left_hist[left_nonzero] / mixture[left_nonzero])))
        kl_right = float(np.sum(right_hist[right_nonzero] * np.log(right_hist[right_nonzero] / mixture[right_nonzero])))
        values.append(0.5 * (kl_left + kl_right))
    return float(np.mean(values)) if values else 0.0


def class_separation(predictions: np.ndarray, labels: np.ndarray) -> float:
    positive = predictions[labels > 0.5]
    negative = predictions[labels <= 0.5]
    denominator = np.sqrt(np.var(positive) + np.var(negative))
    if denominator == 0:
        return float("nan")
    return float((positive.mean() - negative.mean()) / denominator)


def greedy_selection(
    probabilities: np.ndarray,
    labels: np.ndarray,
    size: int,
    candidate_names: Sequence[str] | None = None,
) -> list[str]:
    names = list(candidate_names or [str(i) for i in range(probabilities.shape[1])])
    remaining = list(range(probabilities.shape[1]))
    selected: list[int] = []
    for _ in range(size):
        best_index, best_auc = None, -np.inf
        for candidate in remaining:
            trial = selected + [candidate]
            auc = auc_score(labels, probabilities[:, trial].mean(axis=1))
            if auc > best_auc:
                best_index, best_auc = candidate, auc
        selected.append(best_index)
        remaining.remove(best_index)
    return [names[index] for index in selected]


def random_subsets(member_names: Sequence[str], size: int, repeats: int, seed: int) -> list[list[str]]:
    if size > len(member_names):
        raise ValueError("subset size exceeds bank size")
    rng = np.random.default_rng(seed)
    subsets = []
    for repeat in range(repeats):
        indices = rng.choice(len(member_names), size=size, replace=False)
        subsets.append([member_names[int(index)] for index in indices])
    return subsets
