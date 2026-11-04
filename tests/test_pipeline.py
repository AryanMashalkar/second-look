import json

import numpy as np
import pandas as pd
import pytest

from secondlook import config, evaluate
from secondlook.features import DAYS_EMPLOYED_SENTINEL, build_table, installment_features, previous_features


def tiny_application():
    rows = []
    for i in range(1, 5):
        row = {c: 1 for c in config.APP_NUMERIC}
        row.update({c: "x" for c in config.APP_CATEGORICAL})
        row.update({c: 0.5 for c in config.EXTERNAL})
        row.update(
            SK_ID_CURR=i,
            TARGET=i % 2,
            AMT_INCOME_TOTAL=120_000,
            AMT_CREDIT=240_000,
            AMT_ANNUITY=12_000,
            AMT_GOODS_PRICE=216_000,
            DAYS_EMPLOYED=-1000 if i != 2 else DAYS_EMPLOYED_SENTINEL,
            DAYS_BIRTH=-365.25 * (25 + 15 * i),
            CODE_GENDER="F",
        )
        rows.append(row)
    return pd.DataFrame(rows)


def tiny_previous():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 3],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved", "Refused"],
            "DAYS_DECISION": [-100, -50, -400, -10],
            "AMT_APPLICATION": [100, 50, 200, 10],
            "AMT_CREDIT": [100, 0, 100, 10],
        }
    )


def tiny_installments():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 1],
            "DAYS_INSTALMENT": [-100, -70, -40, -10],
            "DAYS_ENTRY_PAYMENT": [-100, -60, -45, -10],
            "AMT_INSTALMENT": [100, 100, 100, 100],
            "AMT_PAYMENT": [100, 100, 100, 50],
        }
    )


def test_previous_features_counts_and_shares():
    p = previous_features(tiny_previous())
    assert p.loc[1, "PREV_COUNT"] == 3
    assert p.loc[1, "PREV_REFUSED_SHARE"] == pytest.approx(1 / 3)
    assert p.loc[1, "PREV_DAYS_LAST_DECISION"] == -50
    # granted amount of 0 must not create an infinite ratio
    assert np.isfinite(p.loc[1, "PREV_APPLIED_TO_GRANTED"])


def test_installment_features_late_and_underpaid():
    i = installment_features(tiny_installments())
    assert i.loc[1, "INST_COUNT"] == 4
    assert i.loc[1, "INST_LATE_SHARE"] == pytest.approx(0.25)  # only the -70/-60 instalment is late
    assert i.loc[1, "INST_MEAN_DAYS_LATE"] == pytest.approx(2.5)
    assert i.loc[1, "INST_UNDERPAID_SHARE"] == pytest.approx(0.25)


def test_build_table_flags_bureau_and_cleans_sentinel():
    t = build_table(tiny_application(), pd.Series([1, 4]), tiny_previous(), tiny_installments()).set_index("SK_ID_CURR")
    assert t["HAS_BUREAU"].to_dict() == {1: 1, 2: 0, 3: 0, 4: 1}
    assert np.isnan(t.loc[2, "DAYS_EMPLOYED"]) and t.loc[2, "EMPLOYED_YEARS_MISSING"] == 1
    assert t.loc[1, "TERM_MONTHS"] == pytest.approx(20)  # monthly annuity
    assert t.loc[1, "ANNUITY_TO_INCOME"] == pytest.approx(0.1)
    assert t.loc[4, "PREV_COUNT"] == 0  # no history means zero, not missing
    assert list(t["AGE_BAND"]) == ["30-44", "45-59", "60+", "60+"]  # ages 40, 55, 70, 85


def test_matched_policy_rewards_a_better_ranking():
    y = np.array([0] * 80 + [1] * 20)
    perfect = np.arange(100)  # all defaulters ranked riskiest
    random_ = np.random.default_rng(0).permutation(100)
    out = evaluate.matched_policy(y, random_, perfect, 0.5)
    assert out["model_bad_rate"] <= out["baseline_bad_rate"] + 1e-12
    assert out["extra_approved_per_10k"] > 0


def test_fixed_policy_counts_what_the_cutoffs_approve():
    y = np.array([0, 0, 1, 0, 1, 1])
    base = np.array([0.1, 0.5, 0.2, 0.6, 0.7, 0.9])  # cut-off 0.3 approves rows 0 and 2
    model = np.array([0.1, 0.2, 0.8, 0.3, 0.9, 0.9])  # cut-off 0.3 approves rows 0, 1 and 3
    out = evaluate.fixed_policy(y, base, model, 0.3, 0.3)
    assert out["baseline_bad_rate"] == pytest.approx(0.5)
    assert out["model_bad_rate"] == 0
    assert out["extra_approved_per_10k"] == pytest.approx(10_000 / 6)
    assert out["extra_repayers_per_10k"] == pytest.approx(2 * 10_000 / 6)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    good = y + rng.normal(0, 0.8, 500)
    noise = rng.normal(0, 1, 500)
    out = evaluate.paired_bootstrap_auc(y, {"noise": noise, "good": good}, n_boot=200)
    assert out["good"]["ci_low"] <= out["good"]["auc"] <= out["good"]["ci_high"]
    assert out["good"]["delta_vs_noise"]["ci_low"] > 0


artifacts_ready = (config.ARTIFACTS / "second_look.txt").exists()


@pytest.mark.skipif(not artifacts_ready, reason="run python -m secondlook.train first")
class TestArtifacts:
    def setup_class(self):
        from secondlook.model import Baseline, SecondLook

        self.model = SecondLook.load()
        self.baseline = Baseline.load()
        self.demo = pd.read_csv(config.ARTIFACTS / "demo_profiles.csv")
        self.metrics = json.loads((config.ARTIFACTS / "metrics.json").read_text())

    def test_demo_scores_are_recomputed_by_the_shipped_model(self):
        assert np.allclose(self.model.risk(self.demo), self.demo["risk_second_look"])
        assert np.allclose(self.baseline.risk(self.demo), self.demo["risk_baseline"])

    def test_reported_thresholds_match_the_shipped_models(self):
        assert self.model.threshold == pytest.approx(self.metrics["thresholds"]["second_look"])
        assert self.baseline.threshold == pytest.approx(self.metrics["thresholds"]["baseline"])

    def test_model_never_uses_external_or_protected_fields(self):
        banned = set(config.EXTERNAL) | {"CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS", "AGE_BAND"}
        assert banned.isdisjoint(self.model.booster.feature_name())

    def test_recourse_only_changes_loan_terms_and_reaches_threshold(self):
        declined = self.demo[self.demo["risk_second_look"] > self.model.threshold]
        found = 0
        for i in range(len(declined)):
            row = declined.iloc[[i]]
            rc = self.model.recourse(row)
            if rc["found"]:
                found += 1
                assert rc["new_risk"] <= self.model.threshold < rc["base_risk"]
                assert 0.5 <= rc["amount_factor"] <= 1 and 0.75 <= rc["term_factor"] <= 1.5
        assert found > 0

    def test_more_late_payments_never_lower_risk(self):
        row = self.demo.iloc[[0]].copy()
        risks = []
        for share in (0.0, 0.2, 0.5, 1.0):
            row["INST_LATE_SHARE"] = share
            risks.append(float(self.model.risk(row)[0]))
        assert risks == sorted(risks)

    def test_reasons_sum_to_model_output(self):
        row = self.demo.iloc[[3]]
        contrib = self.model.booster.predict(self.model.prepare(row), pred_contrib=True)[0]
        p = 1 / (1 + np.exp(-contrib.sum()))
        assert p == pytest.approx(float(self.model.risk(row)[0]))
