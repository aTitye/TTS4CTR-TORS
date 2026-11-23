import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _bank(tmp_path):
    labels = np.array([0, 0, 1, 1], dtype=float)
    members = np.array([
        [0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.8, 0.9, 0.6, 0.7], [0.9, 0.8, 0.7, 0.6]
    ])
    np.save(tmp_path / "labels.npy", labels)
    members_manifest = []
    for index in range(members.shape[1]):
        path = tmp_path / f"m{index}.npy"
        np.save(path, members[:, index])
        members_manifest.append({"member": f"m{index}", "path": str(path)})
    manifest = {"name": "tiny", "labels_path": str(tmp_path / "labels.npy"), "members": members_manifest}
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, tmp_path / "labels.npy"


def test_member_selection_cli(tmp_path):
    bank, labels = _bank(tmp_path)
    output = tmp_path / "members.csv"
    subprocess.run([sys.executable, str(ROOT / "tools/select_prediction_members.py"), "--bank-json", str(bank), "--scales", "2", "--repeats", "2", "--output", str(output)], check=True)
    assert output.exists()


def test_group_capacity_cli(tmp_path):
    bank, labels = _bank(tmp_path)
    output = tmp_path / "capacity.csv"
    subprocess.run([sys.executable, str(ROOT / "tools/analyze_group_capacity.py"), "--manifest", str(bank), "--group-size", "2", "--output", str(output)], check=True)
    assert output.exists()


def test_perturbation_merging_cli(tmp_path):
    bank, labels = _bank(tmp_path)
    output = tmp_path / "perturbation.json"
    subprocess.run([sys.executable, str(ROOT / "tools/analyze_perturbation_merging.py"), "--parent", str(bank.parent / "m0.npy"), "--members", str(bank.parent / "m1.npy"), str(bank.parent / "m2.npy"), "--labels", str(labels), "--output", str(output)], check=True)
    payload = json.loads(output.read_text())
    assert len(payload["rows"]) == 3
