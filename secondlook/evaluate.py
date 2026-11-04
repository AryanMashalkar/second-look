"""Metrics: AUC with paired bootstrap, lending-policy curves and fairness."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def paired_bootstrap_auc(y, scores: dict, n_boot=1000, seed=0) -> dict:
    """AUC per model with 95% CIs, plus CIs for each model's difference to the first one."""
    y = np.asarray(y)
    names = list(scores)
    arr = {k: np.asarray(v) for k, v in scores.items()}
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in names}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for k in names:
            boot[k].append(roc_auc_score(yb, arr[k][idx]))
    out = {}
    ref = names[0]
    for k in names:
        b = np.array(boot[k])
        out[k] = {
            "auc": float(roc_auc_score(y, arr[k])),
            "ci_low": float(np.percentile(b, 2.5)),
            "ci_high": float(np.percentile(b, 97.5)),
        }
        if arr[k].min() >= 0 and arr[k].max() <= 1:
            out[k]["brier"] = float(brier_score_loss(y, arr[k]))
        if k != ref:
            d = b - np.array(boot[ref])
            out[k]["delta_vs_" + ref] = {
                "mean": float(out[k]["auc"] - out[ref]["auc"]),
                "ci_low": float(np.percentile(d, 2.5)),
                "ci_high": float(np.percentile(d, 97.5)),
                "share_positive": float((d > 0).mean()),
            }
    return out


def policy_curve(y, risk, points=101) -> pd.DataFrame:
    """Approve the lowest-risk share of applicants; report the bad rate among approvals."""
    y = np.asarray(y)
    order = np.argsort(np.asarray(risk), kind="stable")
    bad_sorted = y[order]
    cum_bad = np.cumsum(bad_sorted)
    n = len(y)
    rows = []
    for rate in np.linspace(0.01, 1.0, points - 1):
        k = max(1, int(round(rate * n)))
        rows.append({"approval_rate": k / n, "bad_rate": cum_bad[k - 1] / k, "good_approved": k - cum_bad[k - 1]})
    return pd.DataFrame(rows)


def matched_policy(y, baseline_risk, model_risk, baseline_rate: float) -> dict:
    """At the baseline's bad rate, how many more applicants (and repayers) can the model approve?"""
    y = np.asarray(y)
    n = len(y)
    k_base = int(round(baseline_rate * n))
    order_b = np.argsort(np.asarray(baseline_risk), kind="stable")
    base_bad = y[order_b][:k_base].sum()
    base_bad_rate = base_bad / k_base

    order_m = np.argsort(np.asarray(model_risk), kind="stable")
    cum_bad = np.cumsum(y[order_m])
    ks = np.arange(1, n + 1)
    ok = np.where(cum_bad / ks <= base_bad_rate + 1e-12)[0]
    k_model = int(ks[ok].max()) if len(ok) else 0
    model_bad = int(cum_bad[k_model - 1]) if k_model else 0
    per = 10_000 / n
    return {
        "baseline_approval_rate": k_base / n,
        "baseline_bad_rate": float(base_bad_rate),
        "model_approval_rate": k_model / n,
        "model_bad_rate": float(model_bad / k_model) if k_model else 0.0,
        "extra_approved_per_10k": float((k_model - k_base) * per),
        "extra_repayers_per_10k": float(((k_model - model_bad) - (k_base - base_bad)) * per),
    }


def fixed_policy(y, baseline_risk, model_risk, baseline_thr: float, model_thr: float) -> dict:
    """Apply cut-offs chosen elsewhere (on validation) and report what they do on this population."""
    y = np.asarray(y)
    n = len(y)
    base_ok = np.asarray(baseline_risk) <= baseline_thr
    model_ok = np.asarray(model_risk) <= model_thr
    k_base, k_model = int(base_ok.sum()), int(model_ok.sum())
    base_bad, model_bad = int(y[base_ok].sum()), int(y[model_ok].sum())
    per = 10_000 / n
    return {
        "baseline_approval_rate": k_base / n,
        "baseline_bad_rate": base_bad / k_base if k_base else 0.0,
        "model_approval_rate": k_model / n,
        "model_bad_rate": model_bad / k_model if k_model else 0.0,
        "extra_approved_per_10k": float((k_model - k_base) * per),
        "extra_repayers_per_10k": float(((k_model - model_bad) - (k_base - base_bad)) * per),
    }


def bootstrap_policy(y, baseline_ok, model_ok, n_boot=1000, seed=0) -> dict:
    """95% CIs for extra approvals and extra repayers per 10,000 under fixed decisions."""
    y, b, m = np.asarray(y), np.asarray(baseline_ok), np.asarray(model_ok)
    n = len(y)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    extra = (m[idx].sum(1) - b[idx].sum(1)) * 10_000 / n
    repaid = ((m[idx] & (y[idx] == 0)).sum(1) - (b[idx] & (y[idx] == 0)).sum(1)) * 10_000 / n
    return {
        "extra_approved_ci": [float(np.percentile(extra, 2.5)), float(np.percentile(extra, 97.5))],
        "extra_repayers_ci": [float(np.percentile(repaid, 2.5)), float(np.percentile(repaid, 97.5))],
    }


def parity_cutoffs(risk, groups, y, max_bad_rate: float) -> dict:
    """Per-group cut-offs that approve every group at the same rate, as high as the bad-rate limit allows."""
    risk, groups, y = np.asarray(risk), np.asarray(groups), np.asarray(y)
    best = None
    for rate in np.arange(0.50, 0.995, 0.0025):
        thr = {g: float(np.quantile(risk[groups == g], rate)) for g in np.unique(groups)}
        ok = risk <= np.array([thr[g] for g in groups])
        if y[ok].mean() <= max_bad_rate + 1e-12:
            best = (float(rate), thr)
    return {"rate": best[0], "thresholds": best[1]}


def threshold_for_rate(risk, rate: float) -> float:
    """Risk cut-off that approves `rate` of this population."""
    return float(np.quantile(np.asarray(risk), rate))


def fairness_table(df: pd.DataFrame, group_col: str, risk_cols: dict, thresholds: dict) -> pd.DataFrame:
    """Approval rate overall and among actual repayers (equal opportunity), per group and model."""
    rows = []
    for model, col in risk_cols.items():
        approved = df[col] <= thresholds[model]
        for group, part in df.groupby(group_col, observed=True):
            a = approved.loc[part.index]
            good = part["TARGET"] == 0
            rows.append(
                {
                    "model": model,
                    "attribute": group_col,
                    "group": str(group),
                    "n": int(len(part)),
                    "approval_rate": float(a.mean()),
                    "repayer_approval_rate": float(a[good].mean()) if good.any() else float("nan"),
                    "bad_rate_approved": float(part.loc[a, "TARGET"].mean()) if a.any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def fairness_gaps(table: pd.DataFrame, min_n=200) -> pd.DataFrame:
    """Largest gap between groups (with at least min_n rows) for each model and attribute."""
    t = table[table["n"] >= min_n]
    g = t.groupby(["model", "attribute"]).agg(
        approval_gap=("approval_rate", lambda s: s.max() - s.min()),
        repayer_approval_gap=("repayer_approval_rate", lambda s: s.max() - s.min()),
    )
    return g.reset_index()
