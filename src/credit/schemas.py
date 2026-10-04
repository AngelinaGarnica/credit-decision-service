"""Request and response shapes, shared by the decision logic and the API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_APPLICATION = {
    "revolving_utilization": 0.35,
    "age": 41,
    "past_due_30_59": 0,
    "debt_ratio": 0.42,
    "monthly_income": 5200.0,
    "open_credit_lines": 7,
    "past_due_90": 0,
    "real_estate_loans": 1,
    "past_due_60_89": 0,
    "dependents": 2,
}


class Application(BaseModel):
    """One credit application. Field names match the model features.

    Validation only rejects impossible values; business limits belong to `policy.py`.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_APPLICATION]})

    revolving_utilization: float = Field(ge=0, description="Unsecured balances / credit limits")
    age: int = Field(ge=0, le=120)
    past_due_30_59: int = Field(ge=0, description="Times 30-59 days past due in 2 years")
    debt_ratio: float = Field(ge=0, description="Monthly debt payments / monthly income")
    monthly_income: float | None = Field(default=None, ge=0)
    open_credit_lines: int = Field(ge=0)
    past_due_90: int = Field(ge=0, description="Times 90+ days past due")
    real_estate_loans: int = Field(ge=0)
    past_due_60_89: int = Field(ge=0, description="Times 60-89 days past due in 2 years")
    dependents: int | None = Field(default=None, ge=0)


class DecisionResult(BaseModel):
    """The decision and everything needed to explain and audit it later."""

    decision: Literal["approve", "decline"]
    probability_of_default: float | None  # None when a policy rule declined without scoring
    reason_codes: list[str]
    policy_version: str
    model_version: str
