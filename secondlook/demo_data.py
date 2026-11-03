"""Synthetic applicant profiles for the public demo.

The Home Credit competition data may not be redistributed, so the live app never
ships real rows. Each demo profile is assembled from blocks of fields (loan terms,
work, home, social circle, lender history, external scores), every block taken
from a *different* randomly chosen training applicant, then jittered. No profile
matches a real person. All reported metrics come from the real held-out test set.
"""

import numpy as np
import pandas as pd

from secondlook import config
from secondlook.features import add_ratios

BLOCKS = {
    "terms": ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_CONTRACT_TYPE"],
    "work": [
        "DAYS_EMPLOYED",
        "EMPLOYED_YEARS_MISSING",
        "NAME_INCOME_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "NAME_EDUCATION_TYPE",
        "FLAG_WORK_PHONE",
        "REG_CITY_NOT_WORK_CITY",
    ],
    "home": [
        "NAME_HOUSING_TYPE",
        "REGION_POPULATION_RELATIVE",
        "REGION_RATING_CLIENT_W_CITY",
        "DAYS_REGISTRATION",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "OWN_CAR_AGE",
        "CNT_CHILDREN",
        "FLAG_PHONE",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
        "FLAG_DOCUMENT_3",
    ],
    "social": ["OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"],
    "history": config.HISTORY,
    "external": config.EXTERNAL,
}
MONEY = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
DAYS = ["DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE", "PREV_DAYS_LAST_DECISION"]


def synthesize(source: pd.DataFrame, n: int, seed=0) -> pd.DataFrame:
    """Block-shuffled, jittered profiles drawn from `source`."""
    rng = np.random.default_rng(seed)
    parts = []
    for cols in BLOCKS.values():
        idx = rng.integers(0, len(source), n)
        parts.append(source[cols].iloc[idx].reset_index(drop=True))
    out = pd.concat(parts, axis=1)
    scale = rng.uniform(0.93, 1.07, (n, len(MONEY)))
    out[MONEY] = (out[MONEY].to_numpy(dtype=float) * scale / 500).round() * 500
    out["AMT_CREDIT"] = out["AMT_CREDIT"].clip(lower=25_000)
    out["AMT_INCOME_TOTAL"] = out["AMT_INCOME_TOTAL"].clip(lower=25_000)
    out["AMT_ANNUITY"] = out["AMT_ANNUITY"].clip(lower=1_500)
    for col in DAYS:
        shift = rng.integers(-30, 31, n)
        out[col] = np.minimum(out[col].to_numpy(dtype=float) + shift, 0)
    for col in config.EXTERNAL:
        out[col] = (out[col] + rng.normal(0, 0.02, n)).clip(0.01, 0.99).round(3)
    add_ratios(out)
    return out


def make_demo_profiles(source: pd.DataFrame, second_look, baseline, n_pool=6000, seed=0) -> pd.DataFrame:
    """A review queue mixing the cases a loan officer would actually see."""
    pool = synthesize(source, n_pool, seed)
    pool["risk_baseline"] = baseline.risk(pool)
    pool["risk_second_look"] = second_look.risk(pool)
    base_ok = pool["risk_baseline"] <= baseline.threshold
    sl_ok = pool["risk_second_look"] <= second_look.threshold
    groups = [
        (~base_ok & sl_ok, 18, "Second look: approve"),
        (base_ok & ~sl_ok, 6, "Second look: caution"),
        (base_ok & sl_ok, 8, "Both approve"),
        (~base_ok & ~sl_ok, 8, "Both decline"),
    ]
    picked = []
    for mask, k, label in groups:
        part = pool[mask]
        if len(part):
            picked.append(part.sample(min(k, len(part)), random_state=seed).assign(queue_reason=label))
    demo = pd.concat(picked).sample(frac=1, random_state=seed).reset_index(drop=True)
    demo.insert(0, "profile_id", [f"SL-{1001 + i}" for i in range(len(demo))])
    return demo
