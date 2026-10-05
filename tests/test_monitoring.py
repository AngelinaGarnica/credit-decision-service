from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from credit import monitoring
from credit.monitoring import (
    SUMMARY_MODEL,
    business_summary,
    drift_status,
    policy_rules,
    psi,
    rule_failure_rates,
    simulate_new_month,
    summary_or_none,
)
from credit.schemas import EXAMPLE_APPLICATION, Application

rng = np.random.default_rng(0)


def test_psi_is_near_zero_for_the_same_distribution():
    values = pd.Series(rng.normal(size=5000))

    assert psi(values, values.sample(frac=1, random_state=1)) == pytest.approx(0, abs=1e-6)


def test_psi_flags_a_shifted_distribution_as_major():
    reference = pd.Series(rng.normal(0, 1, size=5000))
    shifted = pd.Series(rng.normal(1, 1, size=5000))

    assert drift_status(psi(reference, shifted)) == "major"


def test_psi_detects_a_change_in_missingness():
    reference = pd.Series(np.r_[rng.normal(size=4500), [np.nan] * 500])  # 10% missing
    more_missing = pd.Series(np.r_[rng.normal(size=3000), [np.nan] * 2000])  # 40% missing

    assert psi(reference, more_missing) > 0.25


def test_psi_separates_zero_from_nonzero_in_zero_heavy_counts():
    reference = pd.Series([0] * 950 + [1] * 50)  # every decile is 0
    more_delinquent = pd.Series([0] * 700 + [1] * 300)

    assert psi(reference, more_delinquent) > 0.25


def test_simulated_month_oversamples_applicants_without_income():
    income = rng.uniform(1000, 9000, size=10_000)
    income[:2000] = np.nan  # 20% missing, as in the training data
    applicants = pd.DataFrame({"monthly_income": income})

    new_month = simulate_new_month(applicants, size=2000, seed=0)

    assert len(new_month) == 2000
    assert new_month["monthly_income"].isna().mean() > 0.35


class FakeAnthropic:
    """Records the request and returns a canned response, like `client.beta.messages`."""

    def __init__(self, stop_reason="end_turn", text="All stable."):
        self.request = None
        self.response = SimpleNamespace(
            model=SUMMARY_MODEL,
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="text", text=text)],
        )
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.request = request
        return self.response


def test_business_summary_sends_the_metrics_and_returns_the_text():
    client = FakeAnthropic(text="Approvals fell.")

    summary = business_summary({"new_month": {"approval_rate": 0.71}}, client)

    assert summary == "Approvals fell."
    assert client.request["model"] == SUMMARY_MODEL
    assert '"approval_rate": 0.71' in client.request["messages"][0]["content"]


def test_business_summary_raises_on_refusal():
    with pytest.raises(RuntimeError):
        business_summary({}, FakeAnthropic(stop_reason="refusal"))


def test_summary_is_skipped_when_claude_is_unavailable(monkeypatch):
    def unavailable(metrics, client):
        raise RuntimeError("The summary request was refused")

    monkeypatch.setattr(monitoring, "business_summary", unavailable)

    assert summary_or_none({}, api_key="test-key") is None


def test_policy_rules_are_described_from_their_docstrings():
    assert policy_rules()["income_required"] == (
        "Decline applications with zero or missing monthly income."
    )


def test_rule_failure_rates_count_each_rule_separately():
    applications = [
        Application(**EXAMPLE_APPLICATION),
        Application(**EXAMPLE_APPLICATION | {"monthly_income": None}),
        Application(**EXAMPLE_APPLICATION | {"monthly_income": None, "age": 17}),
        Application(**EXAMPLE_APPLICATION | {"monthly_income": 0}),
    ]

    assert rule_failure_rates(applications) == {"minimum_age": 0.25, "income_required": 0.75}
