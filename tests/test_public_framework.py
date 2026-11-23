import json

import numpy as np
import torch

from modules.models import build_model
from tts4ctr import checkpoints, metrics, predictions


def test_auc_and_pooling_math():
    labels = np.array([0, 0, 1, 1], dtype=np.float64)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(metrics.auc_score(labels, scores) - 1.0) < 1e-12
    members = np.array([[0.1, 0.3], [0.2, 0.4]])
    logits = predictions.to_logits(members)
    pooled = metrics.sigmoid(logits.mean(axis=1))
    expected = 1.0 / (1.0 + np.exp(-((np.log(members / (1 - members))).mean(axis=1))))
    assert np.allclose(pooled, expected, atol=1e-12)


def test_quality_decomposition_identity():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=500).astype(float)
    members = rng.random((500, 4))
    result = metrics.quality_ambiguity(members, labels)
    assert abs(result["signed_residual"]) < 1e-12


def test_uniform_state_merge_shapes():
    net = build_model("wukong", {
        "latent_dim": 4, "feat_num": 20, "field_num": 4, "mlp_dropout": 0.0,
        "use_bn": False, "mlp_dims": [8, 4], "num_layers": 1,
        "num_embed_lcb": 3, "num_embed_fmb": 3, "rank_fmb": 2,
        "fm_mlp_dims": [4], "num_L": 1, "expansion_rate": 1,
    })
    states = [{key: value.clone() for key, value in net.state_dict().items()} for _ in range(2)]
    merged = checkpoints.uniform_merge_states(states)
    assert all(merged[key].shape == states[0][key].shape for key in merged)
    expected = torch.stack([state["embedding.embedding"] for state in states]).mean(dim=0)
    assert torch.allclose(merged["embedding.embedding"], expected)


def test_prediction_manifest_roundtrip(tmp_path):
    labels = np.array([0, 1, 0, 1], dtype=float)
    members = np.array([[0.1, 0.3], [0.2, 0.4], [0.3, 0.5], [0.4, 0.6]])
    label_path = tmp_path / "labels.npy"
    member_paths = []
    for index in range(2):
        path = tmp_path / f"member-{index}.npy"
        np.save(path, members[:, index])
        member_paths.append(path)
    np.save(label_path, labels)
    bank = predictions.PredictionBank("tiny", label_path, tuple((f"m{i}", p) for i, p in enumerate(member_paths)))
    values, loaded_labels, names = predictions.load_predictions(bank)
    assert values.shape == (4, 2)
    assert names == ["m0", "m1"]
    assert np.array_equal(loaded_labels, labels)
