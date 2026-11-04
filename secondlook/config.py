"""Paths, feature lists and constants shared by training, evaluation and the app."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"

SEED = 42
TEST_SIZE = 0.20
VALID_SIZE = 0.15

# Policy used for the headline comparison: the baseline approves this share of
# thin-file applicants, and Second Look must match its bad rate among approvals.
BASELINE_APPROVAL_RATE = 0.80

# Application fields the Second Look model may use. Deliberately excluded:
#   EXT_SOURCE_1/2/3   - opaque external scores (that is the baseline)
#   AMT_REQ_CREDIT_BUREAU_*  - credit bureau enquiries
#   CODE_GENDER, DAYS_BIRTH, NAME_FAMILY_STATUS - protected or sensitive; audited, not used
APP_NUMERIC = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT_W_CITY",
    "OWN_CAR_AGE",
    "CNT_CHILDREN",
    "FLAG_DOCUMENT_3",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "REG_CITY_NOT_WORK_CITY",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
]
APP_CATEGORICAL = [
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]
DERIVED = [
    "CREDIT_TO_INCOME",
    "ANNUITY_TO_INCOME",
    "TERM_MONTHS",
    "GOODS_TO_CREDIT",
    "EMPLOYED_YEARS_MISSING",
]
HISTORY = [
    "PREV_COUNT",
    "PREV_APPROVED",
    "PREV_REFUSED",
    "PREV_REFUSED_SHARE",
    "PREV_DAYS_LAST_DECISION",
    "PREV_APPLIED_TO_GRANTED",
    "INST_COUNT",
    "INST_LATE_SHARE",
    "INST_MEAN_DAYS_LATE",
    "INST_UNDERPAID_SHARE",
]
EXTERNAL = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

SECOND_LOOK_FEATURES = APP_NUMERIC + APP_CATEGORICAL + DERIVED + HISTORY
COMBINED_FEATURES = SECOND_LOOK_FEATURES + EXTERNAL

# Attributes audited for fairness but never fed to Second Look.
AUDIT_COLUMNS = ["CODE_GENDER", "AGE_BAND"]

# Directions the model is forced to respect (+1: higher value can only raise risk).
# Keeps reason codes and recourse consistent with common sense.
MONOTONE = {
    "CREDIT_TO_INCOME": 1,
    "ANNUITY_TO_INCOME": 1,
    "DEF_30_CNT_SOCIAL_CIRCLE": 1,
    "DEF_60_CNT_SOCIAL_CIRCLE": 1,
    "PREV_REFUSED_SHARE": 1,
    "INST_LATE_SHARE": 1,
    "INST_MEAN_DAYS_LATE": 1,
    "INST_UNDERPAID_SHARE": 1,
}

# Plain-English names for reason codes.
LABELS = {
    "AMT_INCOME_TOTAL": "Declared income",
    "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Monthly repayment",
    "AMT_GOODS_PRICE": "Price of goods financed",
    "DAYS_EMPLOYED": "Time in current job",
    "DAYS_REGISTRATION": "Time since address registration",
    "DAYS_ID_PUBLISH": "Time since ID document was issued",
    "DAYS_LAST_PHONE_CHANGE": "Time since phone number changed",
    "REGION_POPULATION_RELATIVE": "Population density of home region",
    "REGION_RATING_CLIENT_W_CITY": "Lender's rating of home region and city",
    "OWN_CAR_AGE": "Age of own car",
    "CNT_CHILDREN": "Number of children",
    "FLAG_DOCUMENT_3": "Supplied document 3",
    "FLAG_WORK_PHONE": "Gave a work phone",
    "FLAG_PHONE": "Gave a home phone",
    "REG_CITY_NOT_WORK_CITY": "Works in a different city from home",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Contacts with any lending history",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Contacts 30+ days late",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Contacts 60+ days late",
    "NAME_CONTRACT_TYPE": "Loan type",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_EDUCATION_TYPE": "Education",
    "NAME_HOUSING_TYPE": "Housing",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Employer type",
    "FLAG_OWN_CAR": "Owns a car",
    "FLAG_OWN_REALTY": "Owns property",
    "CREDIT_TO_INCOME": "Loan size relative to income",
    "ANNUITY_TO_INCOME": "Repayment burden relative to income",
    "TERM_MONTHS": "Loan term",
    "GOODS_TO_CREDIT": "Goods price relative to loan",
    "EMPLOYED_YEARS_MISSING": "No employment length (pensioner or unemployed)",
    "PREV_COUNT": "Earlier applications with this lender",
    "PREV_APPROVED": "Earlier applications approved",
    "PREV_REFUSED": "Earlier applications refused",
    "PREV_REFUSED_SHARE": "Share of earlier applications refused",
    "PREV_DAYS_LAST_DECISION": "Time since last decision with this lender",
    "PREV_APPLIED_TO_GRANTED": "Amount asked vs. granted on earlier loans",
    "INST_COUNT": "Instalments paid on earlier loans",
    "INST_LATE_SHARE": "Share of earlier instalments paid late",
    "INST_MEAN_DAYS_LATE": "Average days late on earlier instalments",
    "INST_UNDERPAID_SHARE": "Share of earlier instalments underpaid",
    "EXT_SOURCE_1": "External score 1",
    "EXT_SOURCE_2": "External score 2",
    "EXT_SOURCE_3": "External score 3",
}
