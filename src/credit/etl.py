"""Load the Give Me Some Credit CSV, clean it, and write it to the `applicants` table.

Run with: python -m credit.etl
"""

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from credit.config import get_settings
from credit.db import Applicant, get_engine

logger = logging.getLogger(__name__)

# Kaggle column name -> our column name.
COLUMNS = {
    "SeriousDlqin2yrs": "defaulted",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "past_due_90",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
    "NumberOfDependents": "dependents",
}

PAST_DUE_COLUMNS = ["past_due_30_59", "past_due_60_89", "past_due_90"]

# 96 and 98 in the past-due columns are reporting codes, not real counts.
PAST_DUE_SPECIAL_CODES = [96, 98]

# Below this, the age is a data error (the dataset has a 0), not a young applicant.
MIN_VALID_AGE = 18


def load_raw(path: Path) -> pd.DataFrame:
    """Read the CSV; its unnamed first column is the applicant id."""
    raw = pd.read_csv(path, index_col=0)
    return raw.rename_axis("applicant_id").reset_index()


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and drop invalid rows. Missing income and dependents stay missing."""
    df = raw.rename(columns=COLUMNS)

    invalid_age = df["age"] < MIN_VALID_AGE
    special_code = df[PAST_DUE_COLUMNS].isin(PAST_DUE_SPECIAL_CODES).any(axis=1)
    logger.info(
        "Dropping %d rows with invalid age and %d with past-due codes 96/98",
        invalid_age.sum(),
        special_code.sum(),
    )
    df = df[~invalid_age & ~special_code].copy()

    df["dependents"] = df["dependents"].astype("Int64")  # nullable integer
    return df


def write(df: pd.DataFrame, engine: Engine) -> None:
    """Rebuild the `applicants` table from scratch, so the ETL can be re-run safely."""
    table = Applicant.__table__
    table.drop(engine, checkfirst=True)
    table.create(engine)
    df.to_sql(table.name, engine, if_exists="append", index=False, chunksize=2000, method="multi")


def run() -> None:
    settings = get_settings()
    df = clean(load_raw(settings.raw_data_path))
    write(df, get_engine())
    logger.info("Wrote %d applicants (default rate %.2f%%)", len(df), 100 * df["defaulted"].mean())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
