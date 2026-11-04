"""Train the baseline, Second Look and combined models, evaluate them and write artifacts.

Run:  python -m secondlook.train
"""

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from secondlook import config, evaluate
from secondlook.demo_data import make_demo_profiles
from secondlook.features import load_raw
from secondlook.model import Baseline, SecondLook

PROCESSED = config.ROOT / "data" / "processed" / "applicants.parquet"

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "seed": config.SEED,
}


def load_table() -> pd.DataFrame:
    if PROCESSED.exists():
        return pd.read_parquet(PROCESSED)
    table = load_raw()
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(PROCESSED)
    return table


def audit(table: pd.DataFrame) -> dict:
    thin = table[table["HAS_BUREAU"] == 0]
    return {
        "applicants": int(len(table)),
        "default_rate": float(table["TARGET"].mean()),
        "no_bureau_applicants": int(len(thin)),
        "no_bureau_share": float(len(thin) / len(table)),
        "no_bureau_default_rate": float(thin["TARGET"].mean()),
        "bureau_default_rate": float(table.loc[table["HAS_BUREAU"] == 1, "TARGET"].mean()),
        "ext_source_missing_no_bureau": {c: float(thin[c].isna().mean()) for c in config.EXTERNAL},
        "ext_source_all_missing_no_bureau": float(thin[config.EXTERNAL].isna().all(axis=1).mean()),
        "no_bureau_with_lender_history": float((thin["PREV_COUNT"] > 0).mean()),
    }


def fit_baseline(train: pd.DataFrame) -> Baseline:
    medians = train[config.EXTERNAL].median().tolist()
    base = Baseline([0.0] * 6, 0.0, medians, 0.0)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(base.design(train), train["TARGET"])
    base.coef = lr.coef_[0].tolist()
    base.intercept = float(lr.intercept_[0])
    return base


def fit_lgb(train, valid, features):
    dtrain = lgb.Dataset(train[features], train["TARGET"], categorical_feature=config.APP_CATEGORICAL)
    dvalid = lgb.Dataset(valid[features], valid["TARGET"], reference=dtrain)
    params = {**LGB_PARAMS, "monotone_constraints": [config.MONOTONE.get(f, 0) for f in features]}
    return lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )


def main():
    t0 = time.time()
    config.ARTIFACTS.mkdir(exist_ok=True)
    table = load_table()
    data_audit = audit(table)
    print("audit", json.dumps(data_audit, indent=1))

    strat = table["TARGET"].astype(str) + "_" + table["HAS_BUREAU"].astype(str)
    train_full, test = train_test_split(table, test_size=config.TEST_SIZE, stratify=strat, random_state=config.SEED)
    train, valid = train_test_split(
        train_full, test_size=config.VALID_SIZE, stratify=strat.loc[train_full.index], random_state=config.SEED
    )
    print(f"split train={len(train)} valid={len(valid)} test={len(test)}")

    baseline = fit_baseline(train)
    sl_booster = fit_lgb(train, valid, config.SECOND_LOOK_FEATURES)
    comb_booster = fit_lgb(train, valid, config.COMBINED_FEATURES)
    # Ablation: the same model without the lender's own history, to show where the lift comes from.
    app_only_features = [f for f in config.SECOND_LOOK_FEATURES if f not in config.HISTORY]
    app_booster = fit_lgb(train, valid, app_only_features)
    print(f"trees second_look={sl_booster.best_iteration} combined={comb_booster.best_iteration}")

    cats = {c: [str(v) for v in train[c].cat.categories] for c in config.APP_CATEGORICAL}
    second_look = SecondLook(sl_booster, config.SECOND_LOOK_FEATURES, cats, threshold=0.0)

    test = test.copy()
    test["risk_baseline"] = baseline.risk(test)
    test["risk_second_look"] = second_look.risk(test)
    test["risk_combined"] = comb_booster.predict(test[config.COMBINED_FEATURES], num_iteration=comb_booster.best_iteration)
    test["risk_application_only"] = app_booster.predict(test[app_only_features], num_iteration=app_booster.best_iteration)
    thin = test[test["HAS_BUREAU"] == 0]

    names = {
        "baseline": "risk_baseline",
        "second_look": "risk_second_look",
        "application_only": "risk_application_only",
        "combined": "risk_combined",
    }
    results = {"audit": data_audit, "split": {"train": len(train), "valid": len(valid), "test": len(test), "test_no_bureau": len(thin)}}
    for label, part in (("no_bureau", thin), ("all", test)):
        aucs = evaluate.paired_bootstrap_auc(part["TARGET"], {k: part[v] for k, v in names.items()}, n_boot=1000)
        policy = {
            k: evaluate.matched_policy(part["TARGET"], part["risk_baseline"], part[names[k]], config.BASELINE_APPROVAL_RATE)
            for k in ("second_look", "application_only", "combined")
        }
        results[label] = {"auc": aucs, "policy": policy}

    # Operating cut-offs are chosen on the no-bureau validation split, never on test: the baseline approves
    # BASELINE_APPROVAL_RATE, Second Look the largest share with no higher default rate than the baseline.
    valid_thin = valid[valid["HAS_BUREAU"] == 0]
    v_base = baseline.risk(valid_thin)
    v_sl = second_look.risk(valid_thin)
    v_comb = comb_booster.predict(valid_thin[config.COMBINED_FEATURES], num_iteration=comb_booster.best_iteration)
    base_thr = evaluate.threshold_for_rate(v_base, config.BASELINE_APPROVAL_RATE)
    sl_rate = evaluate.matched_policy(valid_thin["TARGET"], v_base, v_sl, config.BASELINE_APPROVAL_RATE)["model_approval_rate"]
    sl_thr = evaluate.threshold_for_rate(v_sl, sl_rate)
    comb_rate = evaluate.matched_policy(valid_thin["TARGET"], v_base, v_comb, config.BASELINE_APPROVAL_RATE)["model_approval_rate"]
    comb_thr = evaluate.threshold_for_rate(v_comb, comb_rate)
    baseline.threshold = base_thr
    second_look.threshold = sl_thr
    # What those fixed cut-offs actually do on the untouched test applicants.
    results["no_bureau"]["policy_holdout"] = {
        "second_look": evaluate.fixed_policy(thin["TARGET"], thin["risk_baseline"], thin["risk_second_look"], base_thr, sl_thr),
        "combined": evaluate.fixed_policy(thin["TARGET"], thin["risk_baseline"], thin["risk_combined"], base_thr, comb_thr),
    }
    results["thresholds"] = {"baseline": base_thr, "second_look": sl_thr, "combined": comb_thr}

    # Who changes decision, and how did they actually do?
    base_ok = thin["risk_baseline"] <= base_thr
    sl_ok = thin["risk_second_look"] <= sl_thr
    swaps = {}
    for name, mask in {
        "both_approve": base_ok & sl_ok,
        "second_look_only": ~base_ok & sl_ok,
        "baseline_only": base_ok & ~sl_ok,
        "both_decline": ~base_ok & ~sl_ok,
    }.items():
        swaps[name] = {"n": int(mask.sum()), "default_rate": float(thin.loc[mask, "TARGET"].mean()) if mask.any() else None}
    results["decision_changes_no_bureau"] = swaps

    # Fairness on the no-bureau test population (gender XNA rows are too few to audit).
    audit_df = thin[thin["CODE_GENDER"].isin(["F", "M"])]
    thresholds = {"baseline": base_thr, "second_look": sl_thr}
    risk_cols = {"baseline": "risk_baseline", "second_look": "risk_second_look"}
    fair = pd.concat(
        [evaluate.fairness_table(audit_df, col, risk_cols, thresholds) for col in config.AUDIT_COLUMNS],
        ignore_index=True,
    )
    fair.to_csv(config.ARTIFACTS / "fairness.csv", index=False)
    results["fairness_gaps"] = evaluate.fairness_gaps(fair).to_dict(orient="records")

    # Global importance: mean absolute contribution on no-bureau test applicants.
    sample = thin.sample(min(5000, len(thin)), random_state=config.SEED)
    contrib = sl_booster.predict(second_look.prepare(sample), pred_contrib=True)[:, :-1]
    importance = pd.Series(np.abs(contrib).mean(axis=0), index=config.SECOND_LOOK_FEATURES).sort_values(ascending=False)
    results["importance"] = [
        {"feature": f, "label": config.LABELS.get(f, f), "mean_abs_contribution": float(v)} for f, v in importance.head(15).items()
    ]

    curves = []
    for label, part in (("no_bureau", thin), ("all", test)):
        for model, col in names.items():
            c = evaluate.policy_curve(part["TARGET"], part[col])
            c["model"], c["population"] = model, label
            curves.append(c)
    pd.concat(curves).to_csv(config.ARTIFACTS / "policy_curves.csv", index=False)

    # Artifacts the app needs.
    sl_booster.save_model(str(config.ARTIFACTS / "second_look.txt"), num_iteration=sl_booster.best_iteration)
    (config.ARTIFACTS / "second_look_meta.json").write_text(
        json.dumps({"features": config.SECOND_LOOK_FEATURES, "categories": cats, "threshold": sl_thr}, indent=1)
    )
    (config.ARTIFACTS / "baseline.json").write_text(
        json.dumps({"coef": baseline.coef, "intercept": baseline.intercept, "medians": baseline.medians, "threshold": base_thr}, indent=1)
    )
    demo = make_demo_profiles(train[train["HAS_BUREAU"] == 0], second_look, baseline, seed=config.SEED)
    demo.to_csv(config.ARTIFACTS / "demo_profiles.csv", index=False)

    results["params"] = {**LGB_PARAMS, "trees_second_look": sl_booster.best_iteration, "trees_combined": comb_booster.best_iteration}
    results["runtime_seconds"] = round(time.time() - t0, 1)
    (config.ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: results[k] for k in ("no_bureau", "decision_changes_no_bureau", "fairness_gaps")}, indent=1))


if __name__ == "__main__":
    main()
