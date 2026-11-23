"""State-dict merging utilities for compatible model checkpoints."""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping, Sequence

import torch

StateDict = dict[str, torch.Tensor]


def uniform_merge_states(states: Iterable[Mapping[str, torch.Tensor]]) -> StateDict:
    states = list(states)
    if not states:
        raise ValueError("at least one state dict is required")
    first = states[0]
    keys = list(first.keys())
    if any(list(state.keys()) != keys for state in states[1:]):
        raise ValueError("state dicts have different parameter orders")
    merged = OrderedDict()
    for key in keys:
        reference = first[key]
        if not torch.is_tensor(reference):
            merged[key] = first[key]
            continue
        if any(state[key].shape != reference.shape for state in states[1:]):
            raise ValueError(f"shape mismatch at {key}; parameter merging is not compatible")
        merged[key] = torch.stack([state[key].detach().cpu().float() for state in states]).mean(dim=0).to(reference.dtype)
    return merged


def weighted_merge_states(
    states: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[float]
) -> StateDict:
    if len(states) != len(weights):
        raise ValueError("states and weights must have equal length")
    if not states:
        raise ValueError("at least one state dict is required")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("merge weights must be positive")
    normalized = [float(weight) / total for weight in weights]
    first = states[0]
    keys = list(first.keys())
    if any(list(state.keys()) != keys for state in states[1:]):
        raise ValueError("state dicts have different parameter orders")
    merged = OrderedDict()
    for key in keys:
        reference = first[key]
        if not torch.is_tensor(reference):
            merged[key] = first[key]
            continue
        if any(state[key].shape != reference.shape for state in states[1:]):
            raise ValueError(f"shape mismatch at {key}; parameter merging is not compatible")
        merged[key] = sum(
            weight * state[key].detach().cpu().float() for weight, state in zip(normalized, states)
        ).to(reference.dtype)
    return merged


def load_state(path: str) -> StateDict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint {path} does not contain a state dict")
    return payload
