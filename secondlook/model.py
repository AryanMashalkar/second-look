"""Scoring, reason codes and loan-term recourse for a trained Second Look model."""

import json
from dataclasses import dataclass
from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd

from secondlook import config
from secondlook.features import add_ratios


@dataclass
class SecondLook:
    booster: lgb.Booster
    features: list
    categories: dict  # column -> list of levels seen in training
    threshold: float  # maximum risk that is approved under the reference policy

    @classmethod
    def load(cls, directory=config.ARTIFACTS) -> "SecondLook":
        meta = json.loads((directory / "second_look_meta.json").read_text())
        booster = lgb.Booster(model_file=str(directory / "second_look.txt"))
        return cls(booster, meta["features"], meta["categories"], meta["threshold"])

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.features].copy()
        for col, levels in self.categories.items():
            x[col] = pd.Categorical(x[col].astype("object"), categories=levels)
        return x

    def risk(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted probability of default."""
        return self.booster.predict(self.prepare(df))

    def reasons(self, row: pd.DataFrame, top=4) -> pd.DataFrame:
        """Per-feature contributions (log-odds) for one applicant, largest first."""
        contrib = self.booster.predict(self.prepare(row), pred_contrib=True)[0][:-1]
        values = row.iloc[0]
        out = pd.DataFrame(
            {
                "feature": self.features,
                "label": [config.LABELS.get(f, f) for f in self.features],
                "value": [values[f] for f in self.features],
                "contribution": contrib,
            }
        )
        out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index)
        raises = out[out["contribution"] > 0].head(top)
        lowers = out[out["contribution"] < 0].head(top)
        return pd.concat([raises.assign(direction="raises risk"), lowers.assign(direction="lowers risk")])

    def recourse(self, row: pd.DataFrame, max_term_months=72):
        """Smallest change to the loan terms (amount, term) that brings risk under the threshold.

        Only lender-negotiable terms are changed; who the applicant is stays fixed.
        The result describes what the model responds to, not a causal guarantee.
        """
        base_risk = float(self.risk(row)[0])
        if base_risk <= self.threshold:
            return None
        options = []
        for amount, term in product([1.0, 0.9, 0.8, 0.7, 0.6, 0.5], [0.75, 1.0, 1.25, 1.5]):
            if amount == 1.0 and term == 1.0:
                continue
            options.append((round((1 - amount) + 0.5 * abs(term - 1), 4), amount, term))
        options.sort()
        candidates = []
        for _, amount, term in options:
            r = row.copy()
            r["AMT_CREDIT"] = row["AMT_CREDIT"] * amount
            r["AMT_ANNUITY"] = row["AMT_ANNUITY"] * amount / term
            add_ratios(r)
            if float(r["TERM_MONTHS"].iloc[0]) > max_term_months:
                continue
            candidates.append((amount, term, r))
        if not candidates:
            return {"found": False, "base_risk": base_risk}
        batch = pd.concat([c[2] for c in candidates], ignore_index=True)
        risks = self.risk(batch)
        for (amount, term, r), risk in zip(candidates, risks):
            if risk <= self.threshold:
                return {
                    "found": True,
                    "base_risk": base_risk,
                    "new_risk": float(risk),
                    "amount_factor": amount,
                    "term_factor": term,
                    "new_credit": float(r["AMT_CREDIT"].iloc[0]),
                    "new_annuity": float(r["AMT_ANNUITY"].iloc[0]),
                    "new_term_months": float(r["TERM_MONTHS"].iloc[0]),
                }
        return {"found": False, "base_risk": base_risk}


@dataclass
class Baseline:
    """Logistic regression on the external scores, the 'opaque score' a lender would rely on."""

    coef: list
    intercept: float
    medians: list
    threshold: float

    @classmethod
    def load(cls, directory=config.ARTIFACTS) -> "Baseline":
        meta = json.loads((directory / "baseline.json").read_text())
        return cls(meta["coef"], meta["intercept"], meta["medians"], meta["threshold"])

    def design(self, df: pd.DataFrame) -> np.ndarray:
        ext = df[config.EXTERNAL].to_numpy(dtype=float)
        missing = np.isnan(ext).astype(float)
        filled = np.where(np.isnan(ext), np.asarray(self.medians), ext)
        return np.hstack([filled, missing])

    def risk(self, df: pd.DataFrame) -> np.ndarray:
        z = self.design(df) @ np.asarray(self.coef) + self.intercept
        return 1 / (1 + np.exp(-z))
