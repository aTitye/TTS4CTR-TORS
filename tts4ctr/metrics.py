"""Metrics and exact decomposition mathematics for the theory_2 protocol.

Implements (docs/analysis/theory_2.md section 4):
- AUC (exact, tie-aware), clipped LogLoss, raw Brier;
- weighted quality-ambiguity decomposition  R = Q - A  (identity, 4.1);
- finite-bank moments Q_a, C_a, H_a and the E_S R(J) = C_a + H_a/J curve (4.2);
- 3x3 ranking transition accounting with half-credit ties (4.4).

All accumulations are float64.  Identity checks must reach 1e-10 absolute.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn import metrics as skmetrics

EPS_DEFAULT = 1e-7
IDENTITY_TOL = 1e-10


# --------------------------------------------------------------------------- #
# base metrics
# --------------------------------------------------------------------------- #
def clip_probs(p: np.ndarray, eps: float = EPS_DEFAULT) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    """Exact AUC with tie handling (average-rank formulation).

    Equivalent to sklearn's roc_auc_score (trapezoidal on the ROC curve)
    to float64 round-off; kept in-house because the trial grid calls it
    millions of times on 4M-15M row arrays and this avoids sklearn's
    per-threshold overhead.  Validated against sklearn in tests.
    """
    y = np.asarray(y_true)
    s = np.asarray(score)
    n_pos = float(y.sum())
    n_neg = float(y.shape[0] - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    xs = s[order]
    ys = y[order]
    # tie groups: consecutive equal scores share the average 1-based rank
    change = np.nonzero(np.diff(xs))[0]
    ends = np.concatenate([change, [xs.shape[0] - 1]])
    starts = np.concatenate([[0], change + 1])
    avg_rank = (starts + ends) / 2.0 + 1.0          # 1-based average rank
    pos_rank_sum = float(np.repeat(avg_rank, ends - starts + 1)[ys == 1].sum())
    g = n_pos * (n_pos + 1.0) / 2.0
    return float((pos_rank_sum - g) / (n_pos * n_neg))


def logloss_score(y_true: np.ndarray, p: np.ndarray,
                  eps: float = EPS_DEFAULT) -> float:
    """Mean BCE after frozen-eps clipping (logloss_prob_clipped)."""
    pc = clip_probs(p, eps)
    return float(-np.mean(y_true * np.log(pc) + (1.0 - y_true) * np.log1p(-pc)))


def raw_bce_logits(y_true: np.ndarray, z: np.ndarray) -> float:
    """BCE on native logits without clipping: mean(logaddexp(0,z)-y*z)."""
    return float(np.mean(np.logaddexp(0.0, z) - y_true * z))


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """Brier on RAW probabilities (no clipping)."""
    return float(np.mean((p - y_true) ** 2))


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #
def pool_probabilities(members: np.ndarray,
                       weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted probability mean.  members: [n, k]."""
    if weights is None:
        return members.mean(axis=1)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return members @ w


def pool_logits(members: np.ndarray,
                weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted logit mean.  members: [n, k] native logits."""
    if weights is None:
        return members.mean(axis=1)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return members @ w


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray, eps: float = EPS_DEFAULT) -> np.ndarray:
    """Reconstructed logits from probabilities (mark reconstructed_logits)."""
    pc = clip_probs(p, eps)
    return np.log(pc) - np.log1p(-pc)


# --------------------------------------------------------------------------- #
# 4.1 weighted quality-ambiguity decomposition (linear probability pooling)
# --------------------------------------------------------------------------- #
def quality_ambiguity(members: np.ndarray, y_true: np.ndarray,
                      weights: np.ndarray | None = None) -> dict:
    """Exact decomposition of the pooled risk for ONE fixed member set.

    R(pbar) = Q_S - A_S with Q_S = sum_j w_j E (p_j - y)^2 and
    A_S = E sum_j w_j (p_j - pbar)^2.  Returns signed residual; callers assert
    |residual| <= 1e-10 for linear probability pooling (not for logit pooling).
    """
    n, k = members.shape
    w = (np.full(k, 1.0 / k) if weights is None
         else np.asarray(weights, dtype=np.float64))
    w = w / w.sum()
    e2 = (members - y_true[:, None]) ** 2
    q = float(w @ e2.mean(axis=0))
    pbar = members @ w
    a = float(np.mean(w @ (members - pbar[:, None]).T ** 2)) if k > 1 else 0.0
    pooled_risk = brier_score(y_true, pbar)
    return {"Q": q, "A": a, "R": pooled_risk,
            "signed_residual": pooled_risk - (q - a)}


# --------------------------------------------------------------------------- #
# 4.2 finite-bank moments via single-pass u / v accumulation
# --------------------------------------------------------------------------- #
def bank_moments(members: np.ndarray, y_true: np.ndarray,
                 block_rows: int = 65_536) -> dict:
    """Q_a, C_a, H_a for a bank of N_a member probability columns [n, k].

    Streaming: per row block accumulate u = sum_i e_i and v = sum_i e_i^2:
        Q = E[v]/N,  C = E[u^2 - v] / (N (N-1)),  H = Q - C.
    C is a residual cross-moment (may be negative), not a bias.
    """
    n, k = members.shape
    if k < 2:
        return {"N": int(k), "Q": brier_score(y_true, members[:, 0]),
                "C": None, "H": None, "note": "N=1: C not identifiable"}
    u2_acc = 0.0
    v_acc = 0.0
    for sl in _blocks(n, block_rows):
        e = members[sl] - y_true[sl][:, None]
        u = e.sum(axis=1)                     # per-row u = sum_i e_i
        u2_acc += float((u ** 2).sum())       # sum over rows of u^2
        v_acc += float((e ** 2).sum())
    q = v_acc / (n * k)
    c = (u2_acc - v_acc) / (n * k * (k - 1))
    return {"N": int(k), "Q": q, "C": c, "H": q - c}


def finite_bank_curve(q: float, c: float, j_values: Sequence) -> list:
    """Exact expected subset risk E_S R(J) = C + H/J (uniform averaging)."""
    h = q - c
    return [{"J": int(j), "expected_brier": c + h / j} for j in j_values]


# --------------------------------------------------------------------------- #
# 4.4 ranking rescue/damage accounting (3x3 transition, half-credit ties)
# --------------------------------------------------------------------------- #
def h_score(d: np.ndarray) -> np.ndarray:
    """h(D) = 1[D>0] + 0.5*1[D==0]."""
    return (d > 0).astype(np.float64) + 0.5 * (d == 0).astype(np.float64)


def transition_counts(d_ref_pos: np.ndarray, d_ref_neg: np.ndarray,
                      d_a: np.ndarray, d_b: np.ndarray) -> dict:
    """3x3 (neg, zero, pos) x (neg, zero, pos) transition table for A->B.

    All inputs are per-pair score differences (B - A) on the same pairs:
    d_ref_pos on the positive side, d_ref_neg on the negative side; d_a/d_b
    are the generic difference columns used for the closure check.
    Returns counts, pair-level delta AUC and the closure absolute error.
    """
    def cat(d: np.ndarray) -> np.ndarray:
        out = np.full(d.shape, 2, dtype=np.int64)   # pos
        out[d == 0] = 1                             # zero
        out[d < 0] = 0                              # neg
        return out

    ci, cj = cat(d_ref_neg), cat(d_ref_pos)
    counts = np.zeros((3, 3), dtype=np.int64)
    np.add.at(counts, (ci, cj), 1)
    # bin_delta[row=neg_cat, col=pos_cat] = h(pos_cat) - h(neg_cat)
    bin_delta = np.array([[0.0, 0.5, 1.0],
                          [-0.5, 0.0, 0.5],
                          [-1.0, -0.5, 0.0]])
    lhs = float((counts * bin_delta).sum())
    rhs = float(h_score(d_ref_pos).sum() - h_score(d_ref_neg).sum())
    return {
        "counts": counts.tolist(),
        "closure_abs_error": abs(lhs - rhs),
        "pair_auc_delta": float(h_score(d_ref_pos).mean()
                                - h_score(d_ref_neg).mean()),
    }


def _blocks(n: int, block_rows: int):
    start = 0
    while start < n:
        stop = min(start + block_rows, n)
        yield slice(start, stop)
        start = stop
