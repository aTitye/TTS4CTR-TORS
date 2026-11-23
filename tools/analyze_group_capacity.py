#!/usr/bin/env python3
"""Compare branch, group, and system metrics at a fixed group size."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts4ctr import config, metrics, predictions, reporting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="prediction-bank manifest")
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--output", type=Path, default=config.output_root() / "group_capacity.csv")
    parser.add_argument("--pooling", choices=("probability", "logit"), default="logit")
    return parser.parse_args()


def pool(members: np.ndarray, mode: str) -> np.ndarray:
    return members.mean(axis=1) if mode == "probability" else metrics.sigmoid(members.mean(axis=1))


def main() -> None:
    args = parse_args()
    with open(args.manifest, encoding="utf-8") as stream:
        payload = json.load(stream)
    bank = predictions.PredictionBank(
        name=payload["name"],
        labels_path=Path(payload["labels_path"]),
        prediction_paths=tuple((item["member"], Path(item["path"])) for item in payload["members"]),
    )
    probabilities, labels, names = predictions.load_predictions(bank)
    values = probabilities if args.pooling == "probability" else predictions.to_logits(probabilities)
    group_count = len(names)
    if group_size := args.group_size:
        if group_count % group_size:
            raise ValueError("member count is not divisible by the requested group size")
    groups = [list(range(start, start + args.group_size)) for start in range(0, group_count, args.group_size)]

    branch_losses = [metrics.logloss_score(labels, np.clip(probabilities[:, index], 1e-7, 1 - 1e-7)) for index in range(group_count)]
    group_losses = []
    for members in groups:
        pooled = pool(values[:, members], args.pooling)
        group_losses.append(metrics.logloss_score(labels, np.clip(pooled, 1e-7, 1 - 1e-7)))
    final_loss = float(np.mean(group_losses))
    branch_auc = [metrics.auc_score(labels, probabilities[:, index]) for index in range(group_count)]

    basic_mean = metrics.logloss_score(labels, np.clip(probabilities.mean(axis=1), 1e-7, 1 - 1e-7))
    basic_branch_loss = float(np.mean(branch_losses))
    group_branch_loss = float(np.mean(branch_losses))
    within_group = basic_branch_loss - final_loss
    between_group = final_loss - final_loss
    row = {
        "members": group_count,
        "group_size": args.group_size,
        "groups": len(groups),
        "mean_branch_auc": float(np.mean(branch_auc)),
        "branch_logloss": group_branch_loss,
        "group_logloss": float(np.mean(group_losses)),
        "system_logloss": final_loss,
        "branch_cost_vs_basic": group_branch_loss - basic_branch_loss,
        "within_group_benefit": within_group,
        "between_group_benefit": between_group,
    }
    reporting.write_rows(args.output, [row])
    print(args.output)


if __name__ == "__main__":
    main()
