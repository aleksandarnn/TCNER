"""Confidence analysis: entropy, calibration, temperature scaling, selective prediction."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import entropy as _entropy


def ovr_proba(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """sklearn's one-vs-rest probability: normalised sigmoids of the per-venue log-odds."""
    s = expit(logits / temperature)
    return s / s.sum(axis=1, keepdims=True)


def entropy(proba: np.ndarray, normalised: bool = True) -> np.ndarray:
    """Predictive entropy; divided by log(K) when ``normalised`` so it lies in [0, 1]."""
    h = _entropy(proba, axis=1)
    return h / np.log(proba.shape[1]) if normalised else h


def nll(proba, y_idx) -> float:
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-12, None))))


def brier(proba, y_idx) -> float:
    onehot = np.eye(proba.shape[1])[y_idx]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def reliability(proba, y_idx, n_bins: int = 15):
    """Top-label reliability table and expected calibration error (Naeini et al., 2015)."""
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_idx
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            rows.append((edges[b], edges[b + 1], int(m.sum()), conf[m].mean(), correct[m].mean()))
    return float(ece), np.array(rows, dtype=float)


def ece(proba, y_idx, n_bins: int = 15) -> float:
    return reliability(proba, y_idx, n_bins)[0]


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    """Temperature that minimises NLL (Guo et al., 2017), fitted on held-out logits."""
    res = minimize_scalar(
        lambda logt: nll(ovr_proba(logits, np.exp(logt)), y_idx),
        bounds=(np.log(0.05), np.log(20)), method="bounded",
    )
    return float(np.exp(res.x))


def risk_coverage(confidence: np.ndarray, correct: np.ndarray):
    """Selective prediction: answer only the most confident fraction of titles.

    Returns (coverage, accuracy on answered titles) sorted by decreasing confidence,
    and the area under the risk-coverage curve (AURC, lower is better).
    """
    order = np.argsort(-confidence, kind="stable")
    hits = correct[order].astype(float)
    n = np.arange(1, len(hits) + 1)
    acc = np.cumsum(hits) / n
    coverage = n / len(hits)
    aurc = float(np.mean(1 - acc))
    return coverage, acc, aurc


def coverage_at_accuracy(coverage, acc, target: float) -> float:
    ok = np.flatnonzero(acc >= target)
    return float(coverage[ok].max()) if ok.size else 0.0


def em_prior_adjustment(proba: np.ndarray, train_prior: np.ndarray, n_iter: int = 100,
                        tol: float = 1e-8):
    """Saerens et al. (2002): re-estimate class priors on unlabelled test inputs and
    adjust the posteriors accordingly. Returns (adjusted proba, estimated priors)."""
    prior = train_prior.copy()
    for _ in range(n_iter):
        w = proba * (prior / train_prior)
        adj = w / w.sum(axis=1, keepdims=True)
        new = adj.mean(axis=0)
        if np.abs(new - prior).max() < tol:
            prior = new
            break
        prior = new
    w = proba * (prior / train_prior)
    return w / w.sum(axis=1, keepdims=True), prior
