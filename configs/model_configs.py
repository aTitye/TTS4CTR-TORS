"""Public dataset, model, and group-topology configurations.

Paths are intentionally relative placeholders; override them through
``DATASET_CONFIGS`` in downstream scripts or pass ``--data_dir`` explicitly.
"""
from __future__ import annotations

from typing import Dict

DATASET_CONFIGS: Dict[str, Dict] = {
    "avazu": {"data_path": "data/avazu/record", "num_features": 4_500_000, "num_fields": 24},
    "criteo": {"data_path": "data/criteo/record", "num_features": 6_900_000, "num_fields": 39},
    "kdd12": {"data_path": "data/kdd12/record", "num_features": 6_100_000, "num_fields": 11},
}

BASE_MODELS = ["fnn", "finalmlp", "dcn", "dcnv2", "deepfm", "ipnn", "wukong", "rankmixer"]

BASE_MODEL_CONFIGS: Dict[str, Dict[str, Dict]] = {
    "avazu": {
        "fnn": {"dim": 16, "bsize": 4096, "lr": 3e-4, "l2": 3e-6, "optim": "adam", "mlp_dropout": 0.0},
        "finalmlp": {"dim": 16, "bsize": 4096, "lr": 3e-3, "l2": 0.0, "optim": "adam", "mlp_dropout": 0.0},
        "wukong": {"dim": 16, "bsize": 4096, "lr": 3e-4, "l2": 1e-6, "optim": "adam", "mlp_dropout": 0.0},
        "rankmixer": {"dim": 24, "bsize": 4096, "lr": 1e-4, "l2": 0.0, "optim": "adam", "mlp_dropout": 0.0, "num_L": 3, "expansion_rate": 2},
    },
    "criteo": {
        "fnn": {"dim": 16, "bsize": 4096, "lr": 3e-4, "l2": 3e-6, "optim": "adam", "mlp_dropout": 0.0},
        "finalmlp": {"dim": 16, "bsize": 4096, "lr": 3e-3, "l2": 1e-7, "optim": "adam", "mlp_dropout": 0.0},
        "wukong": {"dim": 16, "bsize": 4096, "lr": 3e-4, "l2": 3e-6, "optim": "adam", "mlp_dropout": 0.0},
        "rankmixer": {"dim": 39, "bsize": 4096, "lr": 1e-3, "l2": 1e-6, "optim": "adam", "mlp_dropout": 0.0, "num_L": 3, "expansion_rate": 2},
    },
    "kdd12": {
        "fnn": {"dim": 16, "bsize": 4096, "lr": 1e-3, "l2": 0.0, "optim": "adam", "mlp_dropout": 0.0},
        "finalmlp": {"dim": 16, "bsize": 4096, "lr": 1e-3, "l2": 0.0, "optim": "adam", "mlp_dropout": 0.0},
        "wukong": {"dim": 16, "bsize": 4096, "lr": 3e-4, "l2": 0.0, "optim": "adam", "mlp_dropout": 0.0},
        "rankmixer": {"dim": 33, "bsize": 4096, "lr": 3e-4, "l2": 0.0, "optim": "adam", "mlp_dropout": 0.0, "num_L": 3, "expansion_rate": 2},
    },
}

# Dataset-specific settings for classical CTR baseline architectures.
_BASELINE_CONFIGS: Dict[str, Dict[str, Dict]] = {
    "avazu": {
        "dcn": {"dim": 16, "bsize": 4096, "lr": 3e-4, "l2": 3e-6, "optim": "adam", "mlp_dropout": 0.0, "cross": 3},
        "dcnv2": {"dim": 16, "bsize": 4096, "lr": 3e-3, "l2": 0.0, "optim": "adam", "mlp_dropout": 0.0, "cross": 3},
        "deepfm": {"dim": 16, "bsize": 4096, "lr": 3e-4, "l2": 1e-6, "optim": "adam", "mlp_dropout": 0.0},
        "ipnn": {"dim": 16, "bsize": 4096, "lr": 3e-4, "l2": 3e-6, "optim": "adam", "mlp_dropout": 0.0},
    },
    "criteo": {
        "dcn": {"dim": 16, "bsize": 4096, "lr": 3e-4, "l2": 3e-6, "optim": "adam", "mlp_dropout": 0.0, "cross": 3},
        "dcnv2": {"dim": 16, "bsize": 4096, "lr": 1e-3, "l2": 3e-6, "optim": "adam", "mlp_dropout": 0.0, "cross": 3},
        "deepfm": {"dim": 16, "bsize": 4096, "lr": 3e-4, "l2": 3e-5, "optim": "adam", "mlp_dropout": 0.0},
        "ipnn": {"dim": 16, "bsize": 4096, "lr": 3e-4, "l2": 3e-6, "optim": "adam", "mlp_dropout": 0.0},
    },
    "kdd12": {
        "dcn": {"dim": 16, "bsize": 4096, "lr": 1e-3, "l2": 0.0, "optim": "adam", "mlp_dropout": 0.0, "cross": 3},
        "dcnv2": {"dim": 16, "bsize": 4096, "lr": 1e-3, "l2": 0.0, "optim": "adam", "mlp_dropout": 0.0, "cross": 3},
        "deepfm": {"dim": 16, "bsize": 4096, "lr": 1e-3, "l2": 0.0, "optim": "adam", "mlp_dropout": 0.0},
        "ipnn": {"dim": 16, "bsize": 4096, "lr": 1e-3, "l2": 0.0, "optim": "adam", "mlp_dropout": 0.0},
    },
}
for _dataset, _models in _BASELINE_CONFIGS.items():
    BASE_MODEL_CONFIGS[_dataset].update(_models)

HETEROGENEOUS_TOPOLOGIES = {
    "w-r": ("wukong", "rankmixer"),
}

# Values in the paper's Table "setup_group_configs": 1x, 2x, 4x, 8x, 16x.
GROUP_CONFIG_SCALES: Dict[str, Dict[str, Dict[str, int]]] = {
    "avazu": {
        "rankmixer": {"scales": [1, 2, 4, 8, 16]},
        "wukong": {"scales": [1, 2, 2, 4, 4]},
        "w-r": {"scales": [1, 2, 4, 4, 8]},
    },
    "criteo": {
        "rankmixer": {"scales": [1, 1, 1, 1, 1]},
        "wukong": {"scales": [1, 1, 1, 2, 2]},
        "w-r": {"scales": [1, 1, 1, 1, 1]},
    },
    "kdd12": {
        "rankmixer": {"scales": [1, 2, 2, 4, 8]},
        "wukong": {"scales": [1, 1, 2, 4, 4]},
        "w-r": {"scales": [1, 1, 1, 2, 4]},
    },
}

# Backward-compatible aliases used by the compact training/evaluation entries.
GROUP_CONFIGS = {
    "w-r_Group": ["wukong", "rankmixer"],
}
