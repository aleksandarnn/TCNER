"""Metrics used to evaluate explanations (Co-12 properties made measurable)."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .explain import RemovalGame

# --------------------------------------------------------------------------------------
# Agreement between two attribution vectors (Krishna et al., 2022)
# --------------------------------------------------------------------------------------


def rank_corr(a, b) -> float:
    """Spearman correlation of two attribution vectors (NaN when undefined)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(spearmanr(a, b).statistic)


def top_k(a, k: int) -> set[int]:
    a = np.asarray(a, float)
    return set(np.argsort(-np.abs(a), kind="stable")[: min(k, len(a))].tolist())


def feature_agreement(a, b, k: int = 3) -> float:
    """Fraction of the top-k words (by |attribution|) that two explanations share."""
    k = min(k, len(a))
    return len(top_k(a, k) & top_k(b, k)) / k if k else np.nan


def sign_agreement(a, b, k: int = 3) -> float:
    """Like :func:`feature_agreement`, but a shared word must also have the same sign."""
    k = min(k, len(a))
    common = top_k(a, k) & top_k(b, k)
    same = [i for i in common if np.sign(a[i]) == np.sign(b[i])]
    return len(same) / k if k else np.nan


def top1_match(a, b) -> float:
    return float(int(np.argmax(a)) == int(np.argmax(b)))


def jaccard(s: set, t: set) -> float:
    return len(s & t) / len(s | t) if (s | t) else 1.0


def mean_pairwise(vectors, fn) -> float:
    vals = [fn(vectors[i], vectors[j]) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    return float(np.nanmean(vals)) if vals else np.nan


# --------------------------------------------------------------------------------------
# Faithfulness by removal (ERASER: DeYoung et al., 2020)
# --------------------------------------------------------------------------------------


def comprehensiveness(game: RemovalGame, attr, target: int, k: int) -> float:
    """Drop in P(target) after deleting the k words with the highest attribution."""
    order = np.argsort(-np.asarray(attr), kind="stable")[: min(k, game.n)]
    mask = np.ones((2, game.n), dtype=bool)
    mask[1, order] = False
    proba, _ = game(mask)
    return float(proba[0, target] - proba[1, target])


def sufficiency(game: RemovalGame, attr, target: int, k: int) -> float:
    """Drop in P(target) when *only* the top-k words are kept (lower = more sufficient)."""
    order = np.argsort(-np.asarray(attr), kind="stable")[: min(k, game.n)]
    mask = np.zeros((2, game.n), dtype=bool)
    mask[0] = True
    mask[1, order] = True
    proba, _ = game(mask)
    return float(proba[0, target] - proba[1, target])


def random_comprehensiveness(game: RemovalGame, target: int, k: int, n_draws: int = 20,
                             seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    return float(np.mean([
        comprehensiveness(game, rng.random(game.n), target, k) for _ in range(n_draws)
    ]))


# --------------------------------------------------------------------------------------
# Compactness
# --------------------------------------------------------------------------------------


def words_for_mass(attr, mass: float = 0.8) -> int:
    """Number of words needed to cover ``mass`` of the total |attribution|."""
    a = np.sort(np.abs(np.asarray(attr)))[::-1]
    if a.sum() == 0:
        return 0
    return int(np.searchsorted(np.cumsum(a) / a.sum(), mass) + 1)
