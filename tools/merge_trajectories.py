#!/usr/bin/env python3
"""Merge checkpoints within a trajectory and predictions across trajectories."""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.models import build_model
from tts4ctr import checkpoints, config, metrics, predictions, reporting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--prediction-paths", nargs="+", type=Path, required=True, help="logit arrays [.npy] for the trajectory-level models")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--feature-num", type=int, default=None)
    parser.add_argument("--field-num", type=int, default=None)
    parser.add_argument("--soup-weights", type=float, nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=config.output_root() / "trajectory_merge.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = predictions.load_labels(args.labels)
    prediction_logits = np.stack([np.asarray(np.load(path, mmap_mode="r")).reshape(-1) for path in args.prediction_paths], axis=1)
    rows = []
    for count in range(1, prediction_logits.shape[1] + 1):
        pooled = prediction_logits[:, :count].mean(axis=1)
        probabilities = metrics.sigmoid(pooled)
        rows.append({
            "outer_members": count,
            "auc": metrics.auc_score(labels, probabilities),
            "logloss": metrics.logloss_score(labels, np.clip(probabilities, 1e-7, 1 - 1e-7)),
            "forward_passes": count,
        })
    if len(args.checkpoint_paths) > 1:
        states = [checkpoints.load_state(str(path)) for path in args.checkpoint_paths]
        soup = checkpoints.uniform_merge_states(states) if args.soup_weights is None else checkpoints.weighted_merge_states(states, args.soup_weights)
        model_opt = config.build_model_opt(args.dataset, args.model, args.feature_num, args.field_num)
        network = build_model(args.model, model_opt)
        network.load_state_dict(soup, strict=True)
        output_checkpoint = args.output.with_suffix(".soup.pt")
        torch.save(soup, output_checkpoint)
        rows.append({"soup_checkpoint_count": len(states), "soup_checkpoint": str(output_checkpoint)})
    reporting.write_json(args.output, {"rows": rows})
    print(args.output)


if __name__ == "__main__":
    main()
