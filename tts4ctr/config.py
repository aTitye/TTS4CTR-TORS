"""Shared configuration helpers for the public TORS framework."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import configs

DATASET_ALIASES = {"kdd": "kdd12"}


def canonical_dataset(dataset: str) -> str:
    return DATASET_ALIASES.get(dataset.lower(), dataset.lower())


def data_dir(dataset: str, data_dir: str | os.PathLike[str] | None = None) -> Path:
    dataset = canonical_dataset(dataset)
    if data_dir is not None:
        return Path(data_dir)
    configured = configs.DATASET_CONFIGS[dataset]["data_path"]
    return Path(os.environ.get("TTS4CTR_DATA_ROOT", ".")) / configured


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def base_config(dataset: str, model: str, **overrides: Any) -> dict[str, Any]:
    dataset = canonical_dataset(dataset)
    model = model.lower()
    config = dict(configs.BASE_MODEL_CONFIGS[dataset][model])
    config.update(overrides)
    return config


def build_model_opt(
    dataset: str,
    model: str,
    feature_num: int | None = None,
    field_num: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = canonical_dataset(dataset)
    values = base_config(dataset, model, **(overrides or {}))
    dataset_cfg = configs.DATASET_CONFIGS[dataset]
    return {
        "latent_dim": values["dim"],
        "feat_num": int(feature_num or dataset_cfg["num_features"]),
        "field_num": int(field_num or dataset_cfg["num_fields"]),
        "mlp_dropout": values.get("mlp_dropout", 0.0),
        "use_bn": values.get("mlp_bn", False),
        "mlp_dims": values.get("mlp_dims", [1024, 512, 256]),
        "num_layers": values.get("num_layers", 3),
        "num_embed_lcb": values.get("num_embed_lcb", 16),
        "num_embed_fmb": values.get("num_embed_fmb", 16),
        "rank_fmb": values.get("rank_fmb", 24),
        "fm_mlp_dims": values.get("fm_mlp_dims", [512, 512]),
        "num_L": values.get("num_L", 3),
        "expansion_rate": values.get("expansion_rate", 2),
        "model_dims": values.get("model_dims", {}),
    }


def group_models(topology: str | Sequence[str]) -> list[str]:
    if isinstance(topology, str):
        if "+" in topology:
            return topology.split("+")
        if topology in configs.GROUP_CONFIGS:
            return list(configs.GROUP_CONFIGS[topology])
        return [topology]
    return [model.lower() for model in topology]


def output_root() -> Path:
    return Path(os.environ.get("TTS4CTR_OUTPUT_ROOT", "outputs"))
