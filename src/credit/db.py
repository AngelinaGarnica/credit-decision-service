"""Database engine and table models."""

from functools import lru_cache

from sqlalchemy import Engine, Float, Integer, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from credit.config import get_settings


class Base(DeclarativeBase):
    pass


class Applicant(Base):
    """One historical applicant from the training dataset, after cleaning.

    Missing income and dependents are kept as NULL on purpose: missingness is a
    signal for both the policy rules and the model.
    """

    __tablename__ = "applicants"

    applicant_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    defaulted: Mapped[int] = mapped_column(Integer)  # 1 = 90+ days past due within 2 years
    revolving_utilization: Mapped[float] = mapped_column(Float)
    age: Mapped[int] = mapped_column(Integer)
    past_due_30_59: Mapped[int] = mapped_column(Integer)
    debt_ratio: Mapped[float] = mapped_column(Float)
    monthly_income: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_credit_lines: Mapped[int] = mapped_column(Integer)
    past_due_90: Mapped[int] = mapped_column(Integer)
    real_estate_loans: Mapped[int] = mapped_column(Integer)
    past_due_60_89: Mapped[int] = mapped_column(Integer)
    dependents: Mapped[int | None] = mapped_column(Integer, nullable=True)


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide engine for the configured database."""
    return create_engine(get_settings().database_url)
