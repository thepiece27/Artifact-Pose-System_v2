"""Statistics shared by eval_pose / eval_reloc / eval_change (seeded, scipy).

All bootstrap procedures take an explicit seed; nothing here touches global
RNG state.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence

import numpy as np
from scipy import stats


def describe(values: Sequence[float]) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if not len(v):
        return {"n": 0, "median": None, "mean": None, "rmse": None, "p95": None, "max": None}
    return {"n": int(len(v)), "median": float(np.median(v)), "mean": float(np.mean(v)),
            "rmse": float(np.sqrt(np.mean(v ** 2))), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}


def bootstrap_ci(values: Sequence[float], stat: Callable[[np.ndarray], float] = np.median, *,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 2027) -> tuple | None:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if len(v) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.apply_along_axis(stat, 1, v[idx])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def rmse(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(v) ** 2)))


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (dict key -> p)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[key] = running
    return out


def friedman_wilcoxon(table: dict[str, Sequence[float]]) -> dict:
    """Paired comparison of methods on the same units (rows aligned).

    ``table``: method -> list of values (same length/order, finite only rows
    should be passed). Returns Friedman chi2/p and pairwise Wilcoxon
    signed-rank with Holm correction.
    """
    names = list(table)
    data = np.asarray([table[n] for n in names], dtype=np.float64)
    out = {"methods": names, "n_units": int(data.shape[1]) if data.ndim == 2 else 0}
    if len(names) < 2 or data.shape[1] < 3:
        out["note"] = "not enough paired units"
        return out
    if len(names) >= 3:
        chi2, p = stats.friedmanchisquare(*data)
        out["friedman"] = {"chi2": float(chi2), "p": float(p)}
    raw = {}
    for a, b in combinations(range(len(names)), 2):
        diff = data[a] - data[b]
        if np.allclose(diff, 0):
            raw[f"{names[a]}|{names[b]}"] = 1.0
            continue
        raw[f"{names[a]}|{names[b]}"] = float(stats.wilcoxon(data[a], data[b], zero_method="wilcox").pvalue)
    adj = holm(raw)
    out["wilcoxon_holm"] = [{"a": k.split("|")[0], "b": k.split("|")[1], "p": raw[k], "p_holm": adj[k],
                             "median_diff": float(np.median(data[names.index(k.split("|")[0])]
                                                            - data[names.index(k.split("|")[1])]))}
                            for k in raw]
    return out


def mcnemar(success_a: Sequence[bool], success_b: Sequence[bool]) -> dict:
    a, b = np.asarray(success_a, bool), np.asarray(success_b, bool)
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"a_only": n10, "b_only": n01, "p_exact": p}


def roc_auc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float | None:
    pos, neg = np.asarray(scores_pos, float), np.asarray(scores_neg, float)
    if not len(pos) or not len(neg):
        return None
    # Mann-Whitney U statistic / (n_pos n_neg), ties count 1/2.
    ranks = stats.rankdata(np.r_[pos, neg])
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def tpr_at_fpr(scores_pos, scores_neg, fpr: float = 0.05) -> dict:
    pos, neg = np.sort(np.asarray(scores_pos, float)), np.sort(np.asarray(scores_neg, float))
    if not len(pos) or not len(neg):
        return {"threshold": None, "tpr": None, "fpr": None}
    # Smallest threshold whose FPR (score > t) <= fpr.
    k = int(np.floor(fpr * len(neg)))
    thr = neg[len(neg) - 1 - k] if k < len(neg) else -np.inf
    return {"threshold": float(thr), "tpr": float(np.mean(pos > thr)), "fpr": float(np.mean(neg > thr))}


def coverage(errors: np.ndarray, sigmas: np.ndarray, k: float = 2.0) -> float | None:
    e, s = np.asarray(errors, float), np.asarray(sigmas, float)
    ok = np.isfinite(e) & np.isfinite(s) & (s > 0)
    if not ok.any():
        return None
    return float(np.mean(np.abs(e[ok]) <= k * s[ok]))


def cluster_bootstrap_ci(clusters: dict, stat: Callable[[np.ndarray], float] = np.median, *, n_boot: int = 2000,
                         alpha: float = 0.05, seed: int = 2027) -> tuple | None:
    """Two-stage cluster bootstrap (R5): resample clusters (sessions) with
    replacement, then resample the units inside every selected cluster with
    replacement, pool and compute ``stat``. Units of one session share the
    golden model and are correlated, so this CI is the honest one."""
    groups = [np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=np.float64)
              for v in clusters.values()]
    groups = [g for g in groups if len(g)]
    if not groups or sum(len(g) for g in groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.integers(0, len(groups), size=len(groups))
        pooled = np.concatenate([groups[i][rng.integers(0, len(groups[i]), size=len(groups[i]))] for i in picks])
        boots[b] = stat(pooled)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def wilcoxon_contrasts(table: dict[str, Sequence[float]], contrasts: list[tuple[str, str]]) -> list[dict]:
    """Paired Wilcoxon for pre-registered contrasts (a - b), Holm over the set."""
    raw = {}
    for a, b in contrasts:
        if a not in table or b not in table:
            continue
        x, y = np.asarray(table[a], float), np.asarray(table[b], float)
        if len(x) < 3:
            continue
        p = 1.0 if np.allclose(x, y) else float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)
        raw[f"{a}|{b}"] = (p, float(np.median(x - y)), len(x))
    adj = holm({k: v[0] for k, v in raw.items()})
    return [{"contrast": f"{k.split('|')[0]}-{k.split('|')[1]}", "a": k.split("|")[0], "b": k.split("|")[1],
             "n_units": v[2], "median_diff": v[1], "p": v[0], "p_holm": adj[k]} for k, v in raw.items()]
