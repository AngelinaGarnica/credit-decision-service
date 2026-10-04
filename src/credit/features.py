"""Model inputs: the single source of truth for feature names and order, shared by
training and serving."""

import pandas as pd

TARGET = "defaulted"

FEATURES = [
    "revolving_utilization",
    "age",
    "past_due_30_59",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "past_due_90",
    "real_estate_loans",
    "past_due_60_89",
    "dependents",
]


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model features in a fixed order; missing values stay as NaN."""
    return df[FEATURES].astype(float)
