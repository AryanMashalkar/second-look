"""Turn the raw Home Credit tables into one row per applicant."""

import numpy as np
import pandas as pd

from secondlook import config

DAYS_EMPLOYED_SENTINEL = 365243  # Home Credit's code for "not employed / pensioner"


def application_features(app: pd.DataFrame) -> pd.DataFrame:
    """Clean application fields and add affordability ratios."""
    out = app.copy()
    sentinel = out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    out["EMPLOYED_YEARS_MISSING"] = sentinel.astype(int)
    out.loc[sentinel, "DAYS_EMPLOYED"] = np.nan
    add_ratios(out)
    age_years = -out["DAYS_BIRTH"] / 365.25
    out["AGE_BAND"] = pd.cut(
        age_years, bins=[0, 30, 45, 60, 120], labels=["<30", "30-44", "45-59", "60+"], right=False
    ).astype(str)
    return out


def add_ratios(df: pd.DataFrame) -> None:
    """Affordability ratios. Recomputed whenever loan terms change (recourse)."""
    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_TO_INCOME"] = 12 * df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]  # annuity is monthly, income yearly
    df["TERM_MONTHS"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["GOODS_TO_CREDIT"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]


def previous_features(prev: pd.DataFrame) -> pd.DataFrame:
    """Summarise earlier applications with the same lender."""
    p = prev[["SK_ID_CURR", "NAME_CONTRACT_STATUS", "DAYS_DECISION", "AMT_APPLICATION", "AMT_CREDIT"]].copy()
    p["approved"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["refused"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    p["applied_to_granted"] = (p["AMT_APPLICATION"] / p["AMT_CREDIT"]).replace([np.inf, -np.inf], np.nan)
    g = p.groupby("SK_ID_CURR")
    out = pd.DataFrame(
        {
            "PREV_COUNT": g.size(),
            "PREV_APPROVED": g["approved"].sum(),
            "PREV_REFUSED": g["refused"].sum(),
            "PREV_DAYS_LAST_DECISION": g["DAYS_DECISION"].max(),
            "PREV_APPLIED_TO_GRANTED": g["applied_to_granted"].mean(),
        }
    )
    out["PREV_REFUSED_SHARE"] = out["PREV_REFUSED"] / out["PREV_COUNT"]
    return out


def installment_features(inst: pd.DataFrame) -> pd.DataFrame:
    """Summarise repayment behaviour on earlier loans with the same lender."""
    i = inst[["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]].dropna()
    days_late = (i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]).clip(lower=0)
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": i["SK_ID_CURR"].to_numpy(),
            "late": (days_late > 0).astype(int).to_numpy(),
            "days_late": days_late.to_numpy(),
            "underpaid": (i["AMT_PAYMENT"] < i["AMT_INSTALMENT"] - 1).astype(int).to_numpy(),
        }
    )
    g = frame.groupby("SK_ID_CURR")
    return pd.DataFrame(
        {
            "INST_COUNT": g.size(),
            "INST_LATE_SHARE": g["late"].mean(),
            "INST_MEAN_DAYS_LATE": g["days_late"].mean(),
            "INST_UNDERPAID_SHARE": g["underpaid"].mean(),
        }
    )


def build_table(app: pd.DataFrame, bureau_ids: pd.Series, prev: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """One row per applicant with every model feature, the target and audit columns."""
    table = application_features(app)
    table = table.join(previous_features(prev), on="SK_ID_CURR")
    table = table.join(installment_features(inst), on="SK_ID_CURR")
    for col in ("PREV_COUNT", "PREV_APPROVED", "PREV_REFUSED", "INST_COUNT"):
        table[col] = table[col].fillna(0)
    table["HAS_BUREAU"] = table["SK_ID_CURR"].isin(set(bureau_ids)).astype(int)
    for col in config.APP_CATEGORICAL:
        table[col] = table[col].astype("category")
    keep = (
        ["SK_ID_CURR", "TARGET", "HAS_BUREAU"]
        + config.COMBINED_FEATURES
        + ["CODE_GENDER", "AGE_BAND", "DAYS_BIRTH"]
    )
    return table[keep]


def load_raw(raw_dir=config.RAW_DIR) -> pd.DataFrame:
    """Read the four raw CSVs and build the applicant table."""
    app = pd.read_csv(raw_dir / "application_train.csv")
    bureau_ids = pd.read_csv(raw_dir / "bureau.csv", usecols=["SK_ID_CURR"])["SK_ID_CURR"]
    prev = pd.read_csv(
        raw_dir / "previous_application.csv",
        usecols=["SK_ID_CURR", "NAME_CONTRACT_STATUS", "DAYS_DECISION", "AMT_APPLICATION", "AMT_CREDIT"],
    )
    inst = pd.read_csv(
        raw_dir / "installments_payments.csv",
        usecols=["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
    )
    return build_table(app, bureau_ids, prev, inst)
