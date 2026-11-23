#!/usr/bin/env python3
"""Analyze saved prediction banks for pooling, stability, calibration, and selection.

Examples:
  python tools/analyze_prediction_bank.py --tasks manifest topology pooling stability calibration \
    --split test --repeats 32 --output-dir outputs/analysis/test
  python tools/analyze_prediction_bank.py --tasks manifest topology selector \
    --split validation --selector-test --output-dir outputs/analysis/selection
"""

from __future__ import annotations

import argparse
import math
import re
import zlib
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn import metrics


DEFAULT_DATASETS = ("avazu", "criteo", "kdd12")
DEFAULT_MODELS = ("wukong", "rankmixer", "finalmlp")
DEFAULT_GROUP_SIZES = (1, 2, 4, 8, 16, 32)
EPS = 1e-7


@dataclass(frozen=True)
class PredictionBank:
    dataset: str
    model: str
    group_size: int
    split: str
    labels_path: Path
    prediction_paths: tuple[tuple[int, Path], ...]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(seed for seed, _ in self.prediction_paths)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-root", type=Path, default=Path(os.environ.get("TTS4CTR_SAVE_ROOT", "outputs/predictions")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("TTS4CTR_OUTPUT_ROOT", "outputs/analysis")))
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("manifest", "topology", "pooling", "stability", "calibration", "mechanism", "selector"),
        default=("manifest", "topology", "pooling", "stability", "calibration", "mechanism"),
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--budgets", nargs="+", type=int, default=(16,))
    parser.add_argument("--group-sizes", nargs="+", type=int, default=DEFAULT_GROUP_SIZES)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--stability-scales", nargs="+", type=int, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument(
        "--pilot-groups",
        type=int,
        default=2,
        help="Validation groups used by the moment selector for each candidate topology.",
    )
    parser.add_argument("--selector-test", action="store_true")
    parser.add_argument(
        "--mechanism-cache",
        type=Path,
        default=Path(os.environ.get("TTS4CTR_MECHANISM_CACHE", "outputs/mechanism_cache.pkl")),
        help="Existing selection/separation cache used for size-controlled P0 analysis.",
    )
    parser.add_argument("--pooling", nargs="+", choices=("probability", "logit"), default=("probability", "logit"))
    return parser.parse_args()


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return (base_seed + zlib.crc32(text)) % (2**32 - 1)


def label_path(save_root: Path, dataset: str, split: str) -> Path:
    suffix = "full_true.npy" if split == "test" else "validation_full_true.npy"
    return save_root / dataset / f"{dataset}_{suffix}"


def prediction_dir(save_root: Path, dataset: str, model: str, group_size: int) -> Path:
    if group_size > 1:
        return save_root / dataset / "group" / f"{model}_{group_size}x_Group" / "naive"
    candidates = sorted((save_root / dataset / "basic" / model).glob("dropout-*"))
    return (candidates[0] if candidates else save_root / dataset / "basic" / model / "dropout-0.0") / "naive"


def prediction_pattern(split: str) -> str:
    return "full_pred_seed*.npy" if split == "test" else "full_pred_validation_seed*.npy"


def parse_seed(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1)) if match else None


def discover_bank(save_root: Path, dataset: str, model: str, group_size: int, split: str) -> PredictionBank | None:
    labels = label_path(save_root, dataset, split)
    pred_dir = prediction_dir(save_root, dataset, model, group_size)
    paths: dict[int, Path] = {}
    for path in pred_dir.glob(prediction_pattern(split)):
        seed = parse_seed(path)
        if seed is not None:
            paths[seed] = path
    if not labels.exists() or not paths:
        return None
    return PredictionBank(dataset, model, group_size, split, labels, tuple(sorted(paths.items())))


def load_labels(bank: PredictionBank) -> np.ndarray:
    return np.asarray(np.load(bank.labels_path, mmap_mode="r")).reshape(-1).astype(np.float64, copy=False)


def load_predictions(bank: PredictionBank, requested_seeds: Sequence[int] | None = None) -> np.ndarray:
    wanted = set(requested_seeds) if requested_seeds is not None else None
    arrays = [
        np.asarray(np.load(path, mmap_mode="r")).reshape(-1)
        for seed, path in bank.prediction_paths
        if wanted is None or seed in wanted
    ]
    if not arrays:
        raise ValueError(f"No prediction files selected for {bank.dataset}/{bank.model}/g={bank.group_size}.")
    return np.column_stack(arrays).astype(np.float64, copy=False)


def validate_bank(bank: PredictionBank) -> dict[str, object]:
    labels = np.load(bank.labels_path, mmap_mode="r")
    prediction = np.load(bank.prediction_paths[0][1], mmap_mode="r")
    if labels.size != prediction.size:
        raise ValueError(
            f"Shape mismatch for {bank.dataset}/{bank.model}/g={bank.group_size}: "
            f"{labels.shape} labels vs {prediction.shape} predictions"
        )
    return {
        "dataset": bank.dataset,
        "model": bank.model,
        "group_size": bank.group_size,
        "split": bank.split,
        "labels_path": str(bank.labels_path),
        "labels_shape": list(labels.shape),
        "prediction_dir": str(bank.prediction_paths[0][1].parent),
        "prediction_count": len(bank.prediction_paths),
        "seeds": ",".join(str(seed) for seed in bank.seeds),
        "prediction_shape": list(prediction.shape),
        "prediction_dtype": str(prediction.dtype),
        "prediction_bytes": int(bank.prediction_paths[0][1].stat().st_size),
    }


def feasible_group_sizes(budget: int, candidates: Iterable[int]) -> list[int]:
    return [g for g in sorted(set(candidates)) if g > 0 and budget % g == 0]


def choose_subsets(seeds: Sequence[int], size: int, repeats: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    if size > len(seeds):
        return []
    if size == len(seeds):
        return [tuple(seeds)]
    if math.comb(len(seeds), size) <= repeats:
        return list(combinations(seeds, size))
    selected: set[tuple[int, ...]] = set()
    values = np.asarray(seeds)
    while len(selected) < repeats:
        selected.add(tuple(sorted(rng.choice(values, size=size, replace=False).tolist())))
    return sorted(selected)


def probability_mean(predictions: np.ndarray) -> np.ndarray:
    return predictions.mean(axis=1)


def logit_mean(predictions: np.ndarray) -> np.ndarray:
    clipped = np.clip(predictions, EPS, 1.0 - EPS)
    logits = np.log(clipped) - np.log1p(-clipped)
    mean_logits = logits.mean(axis=1)
    return 1.0 / (1.0 + np.exp(-mean_logits))


def expected_calibration_error(labels: np.ndarray, predictions: np.ndarray, bins: int = 15, mode: str = "uniform") -> float:
    values = np.clip(predictions, EPS, 1.0 - EPS)
    if mode == "uniform":
        edges = np.linspace(0.0, 1.0, bins + 1)
    elif mode == "quantile":
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"Unsupported ECE mode: {mode}")
    assignment = np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)
    ece = 0.0
    for index in range(len(edges) - 1):
        mask = assignment == index
        if mask.any():
            ece += float(mask.mean() * abs(values[mask].mean() - labels[mask].mean()))
    return ece


def metric_row(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    values = np.clip(predictions, EPS, 1.0 - EPS)
    return {
        "auc": float(metrics.roc_auc_score(labels, values)),
        "logloss": float(metrics.log_loss(labels, values, labels=(0, 1))),
        "brier": float(np.mean((labels - values) ** 2)),
        "ece_uniform_15": expected_calibration_error(labels, values, mode="uniform"),
        "ece_quantile_15": expected_calibration_error(labels, values, mode="quantile"),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [name for name in ("split", "dataset", "model", "budget", "group_size", "num_groups", "pooling") if name in frame]
    metrics_to_summarize = ("auc", "logloss", "brier", "ece_uniform_15", "ece_quantile_15")
    summary = frame.groupby(keys, dropna=False).agg(
        trials=("trial", "count"),
        **{f"{metric}_mean": (metric, "mean") for metric in metrics_to_summarize},
        **{f"{metric}_std": (metric, "std") for metric in metrics_to_summarize},
        **{f"{metric}_min": (metric, "min") for metric in metrics_to_summarize},
        **{f"{metric}_max": (metric, "max") for metric in metrics_to_summarize},
    ).reset_index()
    for metric in metrics_to_summarize:
        summary[f"{metric}_std"] = summary[f"{metric}_std"].fillna(0.0)
    return summary.sort_values(keys).reset_index(drop=True)


def discover_banks(args: argparse.Namespace) -> list[PredictionBank]:
    banks: list[PredictionBank] = []
    for dataset in args.datasets:
        for model in args.models:
            for group_size in args.group_sizes:
                bank = discover_bank(args.save_root, dataset, model, group_size, args.split)
                if bank is not None:
                    banks.append(bank)
    return banks


def make_index(banks: Sequence[PredictionBank]) -> dict[tuple[str, str, int], PredictionBank]:
    return {(bank.dataset, bank.model, bank.group_size): bank for bank in banks}


def write_manifest(args: argparse.Namespace, banks: Sequence[PredictionBank]) -> pd.DataFrame:
    manifest = pd.DataFrame(validate_bank(bank) for bank in banks)
    manifest = manifest.sort_values(["dataset", "model", "group_size"]).reset_index(drop=True)
    manifest.to_csv(args.output_dir / f"prediction_manifest_{args.split}.csv", index=False)
    return manifest


def run_topology(args: argparse.Namespace, banks: Sequence[PredictionBank], pooling_analysis: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = make_index(banks)
    topology_rows: list[dict[str, object]] = []
    pooling_rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for model in args.models:
            for budget in args.budgets:
                for group_size in feasible_group_sizes(budget, args.group_sizes):
                    bank = indexed.get((dataset, model, group_size))
                    if bank is None:
                        continue
                    num_groups = budget // group_size
                    if len(bank.seeds) < num_groups:
                        continue
                    labels = load_labels(bank)
                    predictions = load_predictions(bank)
                    if labels.size != predictions.shape[0]:
                        raise ValueError(f"Prediction length mismatch for {dataset}/{model}/g={group_size}.")
                    column_by_seed = {seed: column for column, seed in enumerate(bank.seeds)}
                    rng = np.random.default_rng(stable_seed(args.seed, args.split, dataset, model, budget, group_size))
                    for trial, subset in enumerate(choose_subsets(bank.seeds, num_groups, args.repeats, rng)):
                        selected = predictions[:, [column_by_seed[seed] for seed in subset]]
                        common = {
                            "split": args.split,
                            "dataset": dataset,
                            "model": model,
                            "budget": budget,
                            "group_size": group_size,
                            "num_groups": num_groups,
                            "trial": trial,
                            "member_seeds": ",".join(str(seed) for seed in subset),
                        }
                        probability = probability_mean(selected)
                        topology_rows.append({**common, "pooling": "probability", **metric_row(labels, probability)})
                        if pooling_analysis:
                            for pooling in args.pooling:
                                merged = probability if pooling == "probability" else logit_mean(selected)
                                pooling_rows.append({**common, "pooling": pooling, **metric_row(labels, merged)})

    trials = pd.DataFrame(topology_rows)
    if trials.empty:
        raise RuntimeError("No feasible topology trials. Check selected models, split and save root.")
    trials.to_csv(args.output_dir / f"topology_trials_{args.split}.csv", index=False)
    summary = summarize(trials)
    summary.to_csv(args.output_dir / f"topology_summary_{args.split}.csv", index=False)
    if pooling_analysis:
        pooling_trials = pd.DataFrame(pooling_rows)
        pooling_trials.to_csv(args.output_dir / f"pooling_trials_{args.split}.csv", index=False)
        summarize(pooling_trials).to_csv(args.output_dir / f"pooling_summary_{args.split}.csv", index=False)
    return trials, summary


def run_stability(args: argparse.Namespace, banks: Sequence[PredictionBank]) -> pd.DataFrame:
    indexed = make_index(banks)
    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for model in args.models:
            bank = indexed.get((dataset, model, 1))
            if bank is None:
                continue
            labels, predictions = load_labels(bank), load_predictions(bank)
            positions = {seed: index for index, seed in enumerate(bank.seeds)}
            for scale in args.stability_scales:
                if scale > len(bank.seeds):
                    continue
                rng = np.random.default_rng(stable_seed(args.seed, "stability", args.split, dataset, model, scale))
                for trial, subset in enumerate(choose_subsets(bank.seeds, scale, args.repeats, rng)):
                    selected = predictions[:, [positions[seed] for seed in subset]]
                    rows.append(
                        {
                            "split": args.split,
                            "dataset": dataset,
                            "model": model,
                            "budget": scale,
                            "group_size": 1,
                            "num_groups": scale,
                            "trial": trial,
                            "member_seeds": ",".join(str(seed) for seed in subset),
                            "pooling": "probability",
                            **metric_row(labels, probability_mean(selected)),
                        }
                    )
    trials = pd.DataFrame(rows)
    trials.to_csv(args.output_dir / f"resampling_trials_{args.split}.csv", index=False)
    summary = summarize(trials)
    if not summary.empty:
        summary["auc_variance"] = summary["auc_std"] ** 2
        summary["logloss_variance"] = summary["logloss_std"] ** 2
    summary.to_csv(args.output_dir / f"resampling_summary_{args.split}.csv", index=False)
    return summary


def run_calibration(args: argparse.Namespace, banks: Sequence[PredictionBank]) -> pd.DataFrame:
    indexed = make_index(banks)
    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for model in args.models:
            bank = indexed.get((dataset, model, 1))
            if bank is None:
                continue
            labels, predictions = load_labels(bank), load_predictions(bank)
            for budget in args.budgets:
                if budget > len(bank.seeds):
                    continue
                rng = np.random.default_rng(stable_seed(args.seed, "calibration", args.split, dataset, model, budget))
                subset = choose_subsets(bank.seeds, budget, 1, rng)[0]
                columns = [bank.seeds.index(seed) for seed in subset]
                selected = predictions[:, columns]
                for pooling, merged in (
                    ("single_fixed_seed", selected[:, 0]),
                    ("probability", probability_mean(selected)),
                    ("logit", logit_mean(selected)),
                ):
                    rows.append(
                        {
                            "split": args.split,
                            "dataset": dataset,
                            "model": model,
                            "budget": budget,
                            "pooling": pooling,
                            "member_seeds": ",".join(str(seed) for seed in subset),
                            **metric_row(labels, merged),
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / f"calibration_{args.split}.csv", index=False)
    return result


def run_mechanism(args: argparse.Namespace) -> pd.DataFrame:
    """Recompute AUC--separation associations after controlling ensemble size."""
    with args.mechanism_cache.open("rb") as file:
        cache = pickle.load(file)
    data = pd.DataFrame(cache["all_results_with_sep"])
    if data.empty:
        raise RuntimeError(f"No mechanism rows in {args.mechanism_cache}.")

    rows: list[dict[str, object]] = []
    for size, group in data.groupby("size"):
        variable = group["auc"].nunique() > 1 and group["separation"].nunique() > 1
        rows.append(
            {
                "scope": "within_size",
                "ensemble_size": int(size),
                "rows": len(group),
                "unique_subsets": group["indices"].nunique(),
                "pearson_auc_separation": group["auc"].corr(group["separation"], method="pearson") if variable else np.nan,
                "spearman_auc_separation": group["auc"].corr(group["separation"], method="spearman") if variable else np.nan,
            }
        )

    demeaned = data.groupby("size").filter(
        lambda group: group["auc"].nunique() > 1 and group["separation"].nunique() > 1
    ).copy()
    demeaned["auc_demeaned"] = demeaned["auc"] - demeaned.groupby("size")["auc"].transform("mean")
    demeaned["separation_demeaned"] = demeaned["separation"] - demeaned.groupby("size")["separation"].transform("mean")
    rows.extend(
        [
            {
                "scope": "pooled_raw",
                "ensemble_size": "all",
                "rows": len(data),
                "unique_subsets": data["indices"].nunique(),
                "pearson_auc_separation": data["auc"].corr(data["separation"], method="pearson"),
                "spearman_auc_separation": data["auc"].corr(data["separation"], method="spearman"),
            },
            {
                "scope": "pooled_size_demeaned_nonconstant",
                "ensemble_size": "all",
                "rows": len(demeaned),
                "unique_subsets": demeaned["indices"].nunique(),
                "pearson_auc_separation": demeaned["auc_demeaned"].corr(demeaned["separation_demeaned"], method="pearson"),
                "spearman_auc_separation": demeaned["auc_demeaned"].corr(demeaned["separation_demeaned"], method="spearman"),
            },
        ]
    )

    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "mechanism_size_controlled_correlations.csv", index=False)
    data.to_csv(args.output_dir / "mechanism_cached_points.csv", index=False)
    return result


def moment_score(labels: np.ndarray, predictions: np.ndarray, target_groups: int) -> dict[str, float]:
    """Class-conditional separation extrapolated to target_groups."""
    statistics: dict[int, tuple[float, float, float]] = {}
    member_count = predictions.shape[1]
    for label in (0, 1):
        values = predictions[labels == label]
        mean = float(values.mean())
        sigma2 = float(np.var(values, axis=0, ddof=1).mean())
        variance_of_mean = float(np.var(values.mean(axis=1), ddof=1))
        if member_count == 1 or sigma2 <= 0.0:
            rho = 1.0
        else:
            rho = float(np.clip((member_count * variance_of_mean / sigma2 - 1.0) / (member_count - 1.0), -1.0 / (member_count - 1.0), 1.0))
        projected = sigma2 * (rho + (1.0 - rho) / target_groups)
        statistics[label] = (mean, projected, rho)
    score = (statistics[1][0] - statistics[0][0]) / math.sqrt(max(statistics[0][1] + statistics[1][1], EPS))
    return {
        "selector_score": float(score),
        "mu_negative": statistics[0][0],
        "mu_positive": statistics[1][0],
        "projected_variance_negative": statistics[0][1],
        "projected_variance_positive": statistics[1][1],
        "rho_negative": statistics[0][2],
        "rho_positive": statistics[1][2],
    }


def run_selector(args: argparse.Namespace, banks: Sequence[PredictionBank]) -> pd.DataFrame:
    if args.split != "validation":
        raise ValueError("The selector must be fit on validation predictions.")
    indexed = make_index(banks)
    candidate_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for model in args.models:
            for budget in args.budgets:
                candidates: list[dict[str, object]] = []
                for group_size in feasible_group_sizes(budget, args.group_sizes):
                    bank = indexed.get((dataset, model, group_size))
                    if bank is None:
                        continue
                    num_groups = budget // group_size
                    pilot_seeds = bank.seeds[: min(args.pilot_groups, len(bank.seeds))]
                    if not pilot_seeds:
                        continue
                    labels, predictions = load_labels(bank), load_predictions(bank, pilot_seeds)
                    rng = np.random.default_rng(stable_seed(args.seed, "selector-members", dataset, model, budget, group_size))
                    pilot_members = choose_subsets(pilot_seeds, min(num_groups, len(pilot_seeds)), 1, rng)[0]
                    member_positions = [pilot_seeds.index(seed) for seed in pilot_members]
                    record = {
                        "split": "validation",
                        "dataset": dataset,
                        "model": model,
                        "budget": budget,
                        "group_size": group_size,
                        "num_groups": num_groups,
                        "pilot_groups": len(pilot_seeds),
                        "pilot_member_seeds": ",".join(str(seed) for seed in pilot_members),
                        "final_members_available": len(bank.seeds) >= num_groups,
                        **moment_score(labels, predictions, num_groups),
                        **{f"validation_{name}": value for name, value in metric_row(labels, probability_mean(predictions[:, member_positions])).items()},
                    }
                    candidate_rows.append(record)
                    candidates.append(record)
                if candidates:
                    selected_rows.append({**max(candidates, key=lambda item: float(item["selector_score"])), "selection_rule": "moment_separation"})

    candidates = pd.DataFrame(candidate_rows)
    selected = pd.DataFrame(selected_rows)
    candidates.to_csv(args.output_dir / "selector_scores_validation.csv", index=False)
    selected.to_csv(args.output_dir / "selector_selected_validation.csv", index=False)
    if args.selector_test and not selected.empty:
        test_rows: list[dict[str, object]] = []
        for record in selected.to_dict(orient="records"):
            bank = discover_bank(args.save_root, record["dataset"], record["model"], int(record["group_size"]), "test")
            if bank is None:
                continue
            num_groups = int(record["num_groups"])
            if len(bank.seeds) < num_groups:
                continue
            rng = np.random.default_rng(
                stable_seed(args.seed, "selector-final-test-members", record["dataset"], record["model"], record["budget"], record["group_size"])
            )
            members = choose_subsets(bank.seeds, num_groups, 1, rng)[0]
            test_metrics = metric_row(load_labels(bank), probability_mean(load_predictions(bank, members)))
            test_rows.append(
                {
                    **record,
                    "test_member_seeds": ",".join(str(seed) for seed in members),
                    **{f"test_{name}": value for name, value in test_metrics.items()},
                }
            )
        pd.DataFrame(test_rows).to_csv(args.output_dir / "selector_frozen_test.csv", index=False)
    return selected


def main() -> None:
    args = parse_args()
    args.save_root = args.save_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "command": " ".join(sys.argv),
        "save_root": str(args.save_root),
        "split": args.split,
        "datasets": args.datasets,
        "models": args.models,
        "budgets": args.budgets,
        "group_sizes": args.group_sizes,
        "repeats": args.repeats,
        "seed": args.seed,
        "pilot_groups": args.pilot_groups,
        "tasks": args.tasks,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    banks = discover_banks(args)
    if not banks:
        raise RuntimeError(f"No matching prediction banks below {args.save_root}.")
    if "manifest" in args.tasks:
        print(f"Manifest banks: {len(write_manifest(args, banks))}")
    if "topology" in args.tasks or "pooling" in args.tasks:
        _, summary = run_topology(args, banks, pooling_analysis="pooling" in args.tasks)
        print(f"Topology summary rows: {len(summary)}")
    if "stability" in args.tasks:
        print(f"Resampling summary rows: {len(run_stability(args, banks))}")
    if "calibration" in args.tasks:
        print(f"Calibration rows: {len(run_calibration(args, banks))}")
    if "mechanism" in args.tasks:
        print(f"Mechanism correlation rows: {len(run_mechanism(args))}")
    if "selector" in args.tasks:
        print(f"Selector choices: {len(run_selector(args, banks))}")


if __name__ == "__main__":
    main()
