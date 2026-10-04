"""Database engine and table models."""

from collections.abc import Iterator
from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import JSON, DateTime, Engine, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from credit.config import get_settings
from credit.schemas import Application, DecisionResult


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


class DecisionLog(Base):
    """One decision served by the API, with its input, for audit and monitoring.

    JSON (not JSONB) keeps the table portable to SQLite for tests.
    """

    __tablename__ = "decisions"

    decision_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    application: Mapped[dict] = mapped_column(JSON)
    decision: Mapped[str] = mapped_column(String(16))
    probability_of_default: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason_codes: Mapped[list] = mapped_column(JSON)
    policy_version: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(64))


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide engine for the configured database."""
    return create_engine(get_settings().database_url)


def create_decisions_table(engine: Engine) -> None:
    DecisionLog.__table__.create(engine, checkfirst=True)


def get_session() -> Iterator[Session]:
    """One session per request (a FastAPI dependency)."""
    with Session(get_engine()) as session:
        yield session


def log_decision(session: Session, application: Application, result: DecisionResult) -> None:
    session.add(DecisionLog(application=application.model_dump(), **result.model_dump()))
    session.commit()
