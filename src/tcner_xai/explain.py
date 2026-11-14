"""Local explanation methods for the venue classifier.

All word-level explainers share one *interpretable representation*: the distinct
words (lemmas) of the cleaned title the model actually sees. Removing a word
deletes all of its occurrences before the text is re-vectorised. This defines a
cooperative game

    v(S) = model output for venue k when only the words in S are kept,

and every word-level method below is a different way of summarising that game:

=================  ===================================================================
``exact_shapley``  exact Shapley values of v (ground truth; titles are short, so all
                   2^n coalitions can be enumerated)
``lime``           LIME (Ribeiro et al., 2016): weighted linear fit to random samples
``kernel_shap``    KernelSHAP (Lundberg & Lee, 2017): sampled estimate of Shapley
``occlusion``      leave-one-out: v(N) - v(N \\ {i})
``linear_shap``    what the original project used: SHAP for a linear model in TF-IDF
                   feature space, phi_j = w_j (x_j - E[x_j]), mapped back to words
``coef_x_input``   w_j * x_j in TF-IDF space, mapped back to words
=================  ===================================================================

Anchors (rules) and counterfactuals (minimal edits) are provided as well.
"""

from __future__ import annotations

import itertools
import time
from math import factorial

import numpy as np

from .model import VenueClassifier

MAX_EXACT_PLAYERS = 15  # 2^15 = 32,768 model calls, about a second per title


def word_types(clean: str) -> list[str]:
    """Distinct words of a cleaned title in order of first appearance."""
    return list(dict.fromkeys(clean.split()))


# --------------------------------------------------------------------------------------
# The removal game
# --------------------------------------------------------------------------------------


class RemovalGame:
    """Evaluate the model on a title with some of its words removed.

    ``masks`` are boolean arrays of shape (m, n_players); True = word kept.
    Calls return ``(proba, logit)``, each of shape (m, n_classes).
    """

    def __init__(self, model: VenueClassifier, clean: str):
        self.model = model
        self.clean = clean
        self.tokens = clean.split()
        self.players = word_types(clean)
        index = {w: i for i, w in enumerate(self.players)}
        self._token_player = np.array([index[t] for t in self.tokens], dtype=int)
        self.n_calls = 0

    @property
    def n(self) -> int:
        return len(self.players)

    def texts(self, masks) -> list[str]:
        masks = np.atleast_2d(np.asarray(masks, dtype=bool))
        keep = masks[:, self._token_player]
        return [" ".join(t for t, k in zip(self.tokens, row) if k) for row in keep]

    def __call__(self, masks):
        texts = self.texts(masks)
        self.n_calls += len(texts)
        X = self.model.transform(texts)
        return self.model.clf.predict_proba(X), self.model.clf.decision_function(X)

    def without(self, drop) -> str:
        mask = np.ones(self.n, dtype=bool)
        mask[list(drop)] = False
        return self.texts(mask[None])[0]


def all_masks(n: int) -> np.ndarray:
    """Row m is the coalition whose bit i is set iff player i is present."""
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)


def shapley_from_table(values: np.ndarray, n: int) -> np.ndarray:
    """Exact Shapley values from a value table indexed by coalition bitmask.

    ``values`` has shape (2^n, k); returns (n, k).
    """
    idx = np.arange(2**n)
    sizes = np.array([bin(i).count("1") for i in idx])
    weight = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
    phi = np.empty((n, values.shape[1]))
    for i in range(n):
        S = idx[((idx >> i) & 1) == 0]
        phi[i] = weight[sizes[S]] @ (values[S | (1 << i)] - values[S])
    return phi


def exact_shapley(game: RemovalGame) -> dict:
    """Exact Shapley values for every venue, in probability and log-odds space."""
    if game.n > MAX_EXACT_PLAYERS:
        raise ValueError(f"{game.n} words > MAX_EXACT_PLAYERS={MAX_EXACT_PLAYERS}")
    t0 = time.perf_counter()
    proba, logit = game(all_masks(game.n))
    out = {
        "proba": shapley_from_table(proba, game.n),
        "logit": shapley_from_table(logit, game.n),
        "v_full": proba[-1],
        "v_empty": proba[0],
        "logit_full": logit[-1],
        "logit_empty": logit[0],
        "seconds": time.perf_counter() - t0,
    }
    return out


def occlusion(game: RemovalGame) -> dict:
    """Leave-one-out attributions for every venue (n + 1 model calls)."""
    masks = np.vstack([np.ones(game.n, bool), ~np.eye(game.n, dtype=bool)])
    proba, logit = game(masks)
    return {"proba": proba[0] - proba[1:], "logit": logit[0] - logit[1:]}


# --------------------------------------------------------------------------------------
# LIME and KernelSHAP on the same game
# --------------------------------------------------------------------------------------


def lime_explain(model: VenueClassifier, clean: str, labels, num_samples: int = 5000,
                 seed: int = 0, kernel_width: float = 25.0):
    """Run LIME on a cleaned title. Returns ``({label: attributions}, score, explanation)``.

    ``bow=True`` + whitespace splitting makes LIME's perturbations identical to
    the removal game (a word is removed everywhere it occurs).
    """
    from lime.lime_text import LimeTextExplainer

    players = word_types(clean)
    explainer = LimeTextExplainer(
        class_names=list(model.classes_), bow=True, split_expression=r"\s+",
        random_state=seed, kernel_width=kernel_width,
    )
    exp = explainer.explain_instance(
        clean, model.proba_clean, labels=tuple(labels), num_features=len(players),
        num_samples=num_samples,
    )
    pos = {w: i for i, w in enumerate(players)}
    out = {}
    for lab in labels:
        phi = np.zeros(len(players))
        for fid, w in exp.as_map()[lab]:
            phi[pos[exp.domain_mapper.indexed_string.word(fid)]] = w
        out[lab] = phi
    return out, float(exp.score), exp


def kernel_shap(game: RemovalGame, labels, nsamples: int, seed: int = 0) -> dict:
    """KernelSHAP estimate of the Shapley values of the removal game (probability space)."""
    import shap

    labels = list(labels)

    def f(Z):
        proba, _ = game(Z > 0.5)
        return proba[:, labels]

    np.random.seed(seed)
    explainer = shap.KernelExplainer(f, np.zeros((1, game.n)))
    sv = explainer.shap_values(np.ones((1, game.n)), nsamples=nsamples, l1_reg=False,
                               silent=True)
    sv = np.asarray(sv)
    if sv.shape == (len(labels), 1, game.n):  # older shap: list of (1, n) arrays
        sv = sv[:, 0, :].T
    else:  # newer shap: (1, n, n_outputs)
        sv = sv.reshape(game.n, len(labels))
    return {lab: sv[:, j] for j, lab in enumerate(labels)}


# --------------------------------------------------------------------------------------
# Feature-space attributions (the original project's SHAP) mapped back to words
# --------------------------------------------------------------------------------------


class FeatureSpaceExplainer:
    """Closed-form SHAP for a linear model and coefficient x input, in TF-IDF space.

    For a linear model with an independent ("interventional") masker, SHAP's
    LinearExplainer returns exactly ``phi_j = w_kj (x_j - mean_j)`` in log-odds
    space; we compute it directly (the notebook checks it against
    ``shap.LinearExplainer``). N-gram attributions are split equally over the
    words they contain. Attributions of n-grams *absent* from the title cannot
    be assigned to any word; their share is reported separately because it
    shows how much of a feature-space SHAP explanation is about missing words.
    """

    def __init__(self, model: VenueClassifier, X_background):
        self.model = model
        self.W = model.clf.coef_  # (K, d)
        self.mean = np.asarray(X_background.mean(axis=0)).ravel()
        self.names = model.feature_names_
        self.tokenize = model.vectorizer.build_tokenizer()

    def explain(self, clean: str, kind: str = "linear_shap") -> dict:
        players = word_types(clean)
        owner: dict[str, list[int]] = {}
        for i, w in enumerate(players):
            for tok in self.tokenize(w):
                owner.setdefault(tok.lower(), []).append(i)

        x = self.model.transform([clean])
        dense = x.toarray().ravel()
        baseline = self.mean if kind == "linear_shap" else np.zeros_like(self.mean)
        contrib = self.W * (dense - baseline)  # (K, d)

        phi = np.zeros((len(players), self.W.shape[0]))
        present = x.indices
        for j in present:
            words = [i for tok in self.names[j].split(" ") for i in owner.get(tok, [])]
            if words:
                for i in words:
                    phi[i] += contrib[:, j] / len(words)
        absent = np.ones(contrib.shape[1], dtype=bool)
        absent[present] = False
        total = np.abs(contrib).sum(axis=1)
        absent_share = np.abs(contrib[:, absent]).sum(axis=1) / np.where(total > 0, total, 1)
        top_absent = [
            [(self.names[j], float(contrib[k, j]))
             for j in np.flatnonzero(absent)[np.argsort(-np.abs(contrib[k, absent]))[:5]]]
            for k in range(contrib.shape[0])
        ]
        return {"logit": phi, "absent_share": absent_share, "top_absent": top_absent}


# --------------------------------------------------------------------------------------
# Anchors
# --------------------------------------------------------------------------------------


def make_anchor_explainer(model: VenueClassifier, seed: int = 0):
    """alibi AnchorText on cleaned text. Removed words are replaced by an unknown token."""
    import warnings

    import spacy

    with warnings.catch_warnings():  # alibi -> transformers -> tqdm warns about ipywidgets
        warnings.simplefilter("ignore")
        from alibi.explainers import AnchorText

    nlp = spacy.load("en_core_web_sm")

    def predictor(texts):
        return model.proba_clean(list(texts)).argmax(1)

    return AnchorText(predictor=predictor, sampling_strategy="unknown", nlp=nlp, seed=seed)


def explain_anchor(anchor_explainer, clean: str, threshold: float = 0.95) -> dict:
    """Run alibi AnchorText.

    Note: alibi's ``coverage`` is measured on alibi's own perturbation samples
    (roughly the chance that every anchor word survives random masking), not on
    real titles. Use :func:`anchor_dataset_stats` for coverage on real data.
    """
    t0 = time.perf_counter()
    exp = anchor_explainer.explain(clean, threshold=threshold)
    return {
        "anchor": list(exp.anchor),
        "precision": float(np.atleast_1d(exp.precision)[0]),
        "coverage": float(np.atleast_1d(exp.coverage)[0]),
        "seconds": time.perf_counter() - t0,
    }


def anchor_dataset_stats(anchor_words, clean_texts, predicted_labels, label) -> dict:
    """Coverage and precision of an anchor rule on a real set of (cleaned) titles.

    coverage  = share of titles that contain every anchor word
    precision = share of those titles the model assigns to ``label``
    """
    words = set(anchor_words)
    if not words:
        return {"data coverage": 1.0, "data precision": float(np.mean(np.asarray(predicted_labels) == label))}
    covered = np.array([words <= set(t.split()) for t in clean_texts])
    prec = float(np.mean(np.asarray(predicted_labels)[covered] == label)) if covered.any() else np.nan
    return {"data coverage": float(covered.mean()), "data precision": prec, "n covered": int(covered.sum())}


# --------------------------------------------------------------------------------------
# Counterfactuals (contrastive explanations)
# --------------------------------------------------------------------------------------


def deletion_counterfactual(model: VenueClassifier, clean: str, target: int | None = None,
                            max_size: int = 3) -> dict | None:
    """Smallest set of words whose removal changes the prediction (to ``target`` if given).

    Exhaustive over subsets of increasing size, so the result is truly minimal.
    """
    game = RemovalGame(model, clean)
    pred = int(game(np.ones((1, game.n), bool))[0].argmax())
    for size in range(1, min(max_size, game.n) + 1):
        combos = list(itertools.combinations(range(game.n), size))
        masks = np.ones((len(combos), game.n), dtype=bool)
        for r, c in enumerate(combos):
            masks[r, list(c)] = False
        proba, _ = game(masks)
        new = proba.argmax(1)
        ok = (new != pred) if target is None else (new == target)
        if ok.any():
            cand = np.flatnonzero(ok)
            score = proba[cand, new[cand]] if target is None else proba[cand, target]
            best = cand[np.argmax(score)]
            return {
                "removed": [game.players[i] for i in combos[best]],
                "new_label": model.classes_[new[best]],
                "new_proba": float(proba[best, new[best]]),
                "size": size,
            }
    return None


def insertion_counterfactual(model: VenueClassifier, clean: str, target: int,
                             top_m: int = 300) -> dict:
    """Single vocabulary word that, appended to the title, most raises venue ``target``."""
    proba = model.proba_clean([clean])[0]
    pred = int(proba.argmax())
    names = model.feature_names_
    unigram = np.array([" " not in n for n in names])
    delta = model.clf.coef_[target] - model.clf.coef_[pred]
    cand = np.flatnonzero(unigram)[np.argsort(-delta[unigram])[:top_m]]
    texts = [f"{clean} {names[j]}" for j in cand]
    P = model.proba_clean(texts)
    flips = P.argmax(1) == target
    best = int(np.argmax(P[:, target]))
    return {
        "word": names[cand[best]],
        "new_proba": float(P[best, target]),
        "flips": bool(flips[best]),
        "n_single_word_flips": int(flips.sum()),
    }
