from dataclasses import dataclass

import pandas as pd
import pytest

from credit.schemas import EXAMPLE_APPLICATION, Application


@dataclass
class FakeRiskModel:
    """Stands in for `RiskModel`: a fixed PD and reasons, and counts how often it scores."""

    probability: float = 0.05
    threshold: float = 0.10
    model_version: str = "test-model"
    times_scored: int = 0

    def probability_of_default(self, features: pd.DataFrame) -> float:
        self.times_scored += 1
        return self.probability

    def reason_codes(self, features: pd.DataFrame) -> list[str]:
        return ["debt_ratio", "revolving_utilization", "age"]


@pytest.fixture
def risk_model() -> FakeRiskModel:
    return FakeRiskModel()


@pytest.fixture
def application() -> Application:
    return Application(**EXAMPLE_APPLICATION)
