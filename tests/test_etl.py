import pandas as pd
import pytest
from sqlalchemy import create_engine

from credit.etl import COLUMNS, clean, write


def raw_row(**overrides) -> dict:
    """One row with Kaggle column names and valid values."""
    row = {
        "applicant_id": 1,
        "SeriousDlqin2yrs": 0,
        "RevolvingUtilizationOfUnsecuredLines": 0.3,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.4,
        "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2.0,
    }
    return row | overrides


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            raw_row(applicant_id=1),
            raw_row(applicant_id=2, age=0),
            raw_row(applicant_id=3, **{"NumberOfTimes90DaysLate": 98}),
            raw_row(applicant_id=4, MonthlyIncome=None, NumberOfDependents=None),
        ]
    )


def test_clean_renames_columns(raw):
    assert set(clean(raw).columns) == {"applicant_id", *COLUMNS.values()}


def test_clean_drops_invalid_age_and_past_due_codes(raw):
    assert clean(raw)["applicant_id"].tolist() == [1, 4]


def test_clean_keeps_missing_income_and_dependents(raw):
    row = clean(raw).set_index("applicant_id").loc[4]

    assert pd.isna(row["monthly_income"])
    assert pd.isna(row["dependents"])


def test_write_replaces_table_contents(raw):
    engine = create_engine("sqlite://")
    df = clean(raw)

    write(df, engine)
    write(df, engine)  # re-running must not duplicate rows

    assert len(pd.read_sql_table("applicants", engine)) == len(df)
