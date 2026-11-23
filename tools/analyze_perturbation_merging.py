#!/usr/bin/env python3
"""Compare parent predictions with merged perturbed predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts4ctr import config, metrics, predictions, reporting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True, help="parent prediction [.npy]")
    parser.add_argument("--members", nargs="+", type=Path, required=True, help="perturbed prediction [.npy] files")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--mode", choices=("probability", "logit"), default="logit")
    parser.add_argument("--output", type=Path, default=config.output_root() / "perturbation_merge.json")
    return parser.parse_args()


def summarize(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auc": metrics.auc_score(labels, probabilities),
        "logloss": metrics.logloss_score(labels, np.clip(probabilities, 1e-7, 1 - 1e-7)),
        "brier": metrics.brier_score(labels, probabilities),
    }


def main() -> None:
    args = parse_args()
    labels = predictions.load_labels(args.labels)
    parent = np.asarray(np.load(args.parent, mmap_mode="r")).reshape(-1)
    members = np.stack([np.asarray(np.load(path, mmap_mode="r")).reshape(-1) for path in args.members], axis=1)
    if members.shape[0] != labels.shape[0] or parent.shape[0] != labels.shape[0]:
        raise ValueError("labels and predictions have different lengths")

    rows = [{"arm": "parent", **summarize(labels, parent)}]
    values = members if args.mode == "probability" else predictions.to_logits(members)
    for count in range(1, members.shape[1] + 1):
        pooled = values[:, :count].mean(axis=1) if args.mode == "probability" else metrics.sigmoid(values[:, :count].mean(axis=1))
        rows.append({"arm": f"merge_{count}", **summarize(labels, pooled)})
    reporting.write_json(args.output, {
        "mode": args.mode,
        "members": len(args.members),
        "rows": rows,
        "parent_quality_gain": {
            metric: rows[-1][metric] - rows[0][metric]
            for metric in ("auc", "logloss", "brier")
        },
    })
    print(args.output)


if __name__ == "__main__":
    main()
