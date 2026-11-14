"""Ground-truth benchmark of local explainers (notebook 03).

For each title we compute the exact Shapley values of the removal game and then
score every explainer against them, plus stability, continuity, contrastivity,
completeness and cost measurements. One function per title so the work can be
spread over processes with joblib.
"""

from __future__ import annotations

import time

import numpy as np

from . import metrics as M
from .explain import (FeatureSpaceExplainer, RemovalGame, exact_shapley, kernel_shap,
                      lime_explain, occlusion)
from .model import VenueClassifier

BUDGETS = (16, 64, 256, 1000, 5000)  # model evaluations per explanation
STABILITY_SEEDS = (0, 1, 2, 3, 4)
STABILITY_BUDGETS = (64, 1000)  # same budgets for LIME and KernelSHAP
CONTINUITY_BUDGET = 1000
NULL_WORD = "the"  # an English stop word: the vectoriser drops it, so it cannot matter


def _score(attr, gt, game, target, k_list=(1, 2, 3)) -> dict:
    out = {
        "spearman": M.rank_corr(attr, gt),
        "top1": M.top1_match(attr, gt),
        "agree@3": M.feature_agreement(attr, gt, 3),
        "sign@3": M.sign_agreement(attr, gt, 3),
    }
    for k in k_list:
        out[f"comp@{k}"] = M.comprehensiveness(game, attr, target, k)
    out["suff@1"] = M.sufficiency(game, attr, target, 1)
    return out


def benchmark_title(model: VenueClassifier, fs: FeatureSpaceExplainer, clean: str,
                    title_id: int, seed: int = 0) -> list[dict]:
    """All measurements for one title. Returns one row per (method, setting)."""
    import warnings

    warnings.filterwarnings("ignore")  # shap/lime warn about small sample budgets by design
    game = RemovalGame(model, clean)
    n = game.n
    t0 = time.perf_counter()
    gt = exact_shapley(game)
    gt_time = time.perf_counter() - t0
    proba_full = gt["v_full"]
    order = np.argsort(-proba_full)
    k, runner_up = int(order[0]), int(order[1])
    gt_p, gt_l = gt["proba"][:, k], gt["logit"][:, k]
    base = dict(title_id=title_id, n_words=n, pred=int(k), p_pred=float(proba_full[k]))
    rows = []

    def add(method, setting, attr, ref, seconds, calls, **extra):
        row = dict(base, method=method, setting=setting, seconds=seconds, model_calls=calls)
        row.update(_score(attr, ref, game, k))
        row["words@80%"] = M.words_for_mass(attr)
        row.update(extra)
        rows.append(row)

    # Ground truth itself (for comprehensiveness reference values and cost)
    efficiency_gap = abs(gt_p.sum() - (proba_full[k] - gt["v_empty"][k]))
    add("Exact Shapley", "exact", gt_p, gt_p, gt_time, 2 ** n,
        efficiency_gap=efficiency_gap,
        contrast_spearman=M.rank_corr(gt_p, gt["proba"][:, runner_up]))

    # Random baseline for comprehensiveness
    rng = np.random.default_rng(seed + title_id)
    rand = rng.random(n)
    add("Random", "-", rand, gt_p, 0.0, 0)

    # Occlusion (leave-one-out)
    t0 = time.perf_counter()
    occ = occlusion(game)
    add("Occlusion", "n+1 calls", occ["proba"][:, k], gt_p, time.perf_counter() - t0, n + 1,
        contrast_spearman=M.rank_corr(occ["proba"][:, k], occ["proba"][:, runner_up]))

    # LIME and KernelSHAP at equal budgets
    first = {"LIME": {}, "KernelSHAP": {}}  # seed-0 explanations, reused for stability
    for b in BUDGETS:
        t0 = time.perf_counter()
        lime_attr, r2, _ = lime_explain(model, clean, labels=[k, runner_up], num_samples=b, seed=seed)
        dt = time.perf_counter() - t0
        lime_sum = lime_attr[k].sum()
        add("LIME", b, lime_attr[k], gt_p, dt, b, lime_r2=r2,
            efficiency_gap=abs(lime_sum - (proba_full[k] - gt["v_empty"][k])),
            contrast_spearman=M.rank_corr(lime_attr[k], lime_attr[runner_up]))
        first["LIME"][b] = lime_attr[k]
        if b <= 1000:
            t0 = time.perf_counter()
            ks = kernel_shap(game, [k, runner_up], nsamples=b, seed=seed)
            dt = time.perf_counter() - t0
            add("KernelSHAP", b, ks[k], gt_p, dt, min(b, 2 ** n),
                efficiency_gap=abs(ks[k].sum() - (proba_full[k] - gt["v_empty"][k])),
                contrast_spearman=M.rank_corr(ks[k], ks[runner_up]))
            first["KernelSHAP"][b] = ks[k]

    # Feature-space methods (log-odds space, compared with log-odds ground truth)
    for name, kind in [("Linear SHAP", "linear_shap"), ("Coef x input", "coef_x_input")]:
        t0 = time.perf_counter()
        fx = fs.explain(clean, kind)
        add(name, "closed form", fx["logit"][:, k], gt_l, time.perf_counter() - t0, 0,
            absent_share=float(fx["absent_share"][k]) if kind == "linear_shap" else 0.0,
            contrast_spearman=M.rank_corr(fx["logit"][:, k], fx["logit"][:, runner_up]))

    # ---- Consistency: same explainer, different random seeds ----------------------
    stab = dict(base, method="stability")
    top3_jaccard = lambda a, b: M.jaccard(M.top_k(a, 3), M.top_k(b, 3))  # noqa: E731
    for budget in STABILITY_BUDGETS:
        lime_runs = [first["LIME"][budget]] + [
            lime_explain(model, clean, labels=[k], num_samples=budget, seed=s)[0][k]
            for s in STABILITY_SEEDS[1:]
        ]
        ks_runs = [first["KernelSHAP"][budget]] + [
            kernel_shap(game, [k], nsamples=budget, seed=s)[k] for s in STABILITY_SEEDS[1:]
        ]
        for name, runs in [("LIME", lime_runs), ("KernelSHAP", ks_runs)]:
            stab[f"{name}@{budget} top3 jaccard"] = M.mean_pairwise(runs, top3_jaccard)
            stab[f"{name}@{budget} spearman"] = M.mean_pairwise(runs, M.rank_corr)
            stab[f"{name}@{budget} rel. std"] = float(
                np.mean(np.std(runs, axis=0)) / (np.mean(np.abs(runs)) + 1e-12))

    # ---- Continuity: insert a stop word the model provably ignores -----------------
    tokens = clean.split()
    pos = int(rng.integers(0, len(tokens) + 1))
    perturbed = " ".join(tokens[:pos] + [NULL_WORD] + tokens[pos:])
    pgame = RemovalGame(model, perturbed)
    assert np.allclose(pgame(np.ones((1, pgame.n), bool))[0], proba_full)
    keep = [pgame.players.index(w) for w in game.players]
    null = pgame.players.index(NULL_WORD)
    p_gt = exact_shapley(pgame)["proba"][:, k]
    p_lime = lime_explain(model, perturbed, labels=[k], num_samples=CONTINUITY_BUDGET, seed=seed)[0][k]
    p_ks = kernel_shap(pgame, [k], nsamples=CONTINUITY_BUDGET, seed=seed)[k]
    for name, before, after in [("Exact Shapley", gt_p, p_gt),
                                ("LIME", first["LIME"][CONTINUITY_BUDGET], p_lime),
                                ("KernelSHAP", first["KernelSHAP"][CONTINUITY_BUDGET], p_ks)]:
        stab[f"{name} continuity spearman"] = M.rank_corr(before, after[keep])
        stab[f"{name} null-word share"] = float(abs(after[null]) / (np.abs(after).sum() + 1e-12))
    rows.append(stab)
    return rows
