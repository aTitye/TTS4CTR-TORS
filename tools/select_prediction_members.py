#!/usr/bin/env python3
"""Select prediction members and compare pooled metrics."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts4ctr import config, metrics, predictions, reporting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-json", type=Path, required=True, help="prediction-bank manifest")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=config.output_root() / "member_selection.csv")
    parser.add_argument("--selection", choices=("random", "greedy", "top_auc", "js"), default="random")
    return parser.parse_args()


def evaluate_subset(probabilities: np.ndarray, labels: np.ndarray, members: list[str]) -> dict[str, float]:
    pooled = probabilities.mean(axis=1)
    return {
        "auc": metrics.auc_score(labels, pooled),
        "logloss": metrics.logloss_score(labels, np.clip(pooled, 1e-7, 1 - 1e-7)),
        "mean_member_auc": float(np.mean([metrics.auc_score(labels, probabilities[:, index]) for index in range(probabilities.shape[1])])),
        "mean_pair_js": predictions.pairwise_js_divergence(probabilities),
        "separation": predictions.class_separation(pooled, labels),
    }


def choose_members(
    selection: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    size: int,
    rng: np.random.Generator,
) -> list[str]:
    if size > len(names):
        raise ValueError("scale exceeds number of available members")
    if selection == "random":
        indices = rng.choice(len(names), size=size, replace=False)
        return [names[int(index)] for index in indices]
    if selection == "greedy":
        return predictions.greedy_selection(probabilities, labels, size, names)
    if selection == "top_auc":
        individual = [metrics.auc_score(labels, probabilities[:, index]) for index in range(probabilities.shape[1])]
        indices = np.argsort(individual)[-size:]
        return [names[int(index)] for index in indices]
    if selection == "js":
        divergences = []
        for index in range(probabilities.shape[1]):
            others = [other for other in range(probabilities.shape[1]) if other != index]
            if others:
                divergences.append(float(np.mean([
                    predictions.pairwise_js_divergence(probabilities[:, [index, other]])
                    for other in others
                ])))
            else:
                divergences.append(0.0)
        indices = np.argsort(divergences)[-size:]
        return [names[int(index)] for index in indices]
    raise ValueError(f"unknown selection strategy: {selection}")


def main() -> None:
    args = parse_args()
    with open(args.bank_json, encoding="utf-8") as stream:
        import json
        manifest = json.load(stream)
    bank = predictions.PredictionBank(
        name=manifest["name"],
        labels_path=Path(manifest["labels_path"]),
        prediction_paths=tuple((item["member"], Path(item["path"])) for item in manifest["members"]),
    )
    probabilities, labels, member_names = predictions.load_predictions(bank)
    rng = np.random.default_rng(args.seed)
    rows = []
    for scale in args.scales:
        sums = None
        for repeat in range(args.repeats):
            subset = choose_members(args.selection, probabilities, labels, member_names, scale, rng)
            indices = [member_names.index(name) for name in subset]
            result = evaluate_subset(probabilities[:, indices], labels, subset)
            sums = {key: sums[key] + value if sums is not None else value for key, value in result.items()}
        rows.append({"scale": scale, "repeats": args.repeats, **{key: value / args.repeats for key, value in sums.items()}})
    reporting.write_rows(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
