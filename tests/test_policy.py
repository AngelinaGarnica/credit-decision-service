import pytest

from credit import policy


def always_fails(application):
    return "always_fails"


def always_passes(application):
    return None


def test_failed_rules_returns_reason_codes_of_failed_rules(monkeypatch, application):
    monkeypatch.setattr(policy, "RULES", [always_fails, always_passes])

    assert policy.failed_rules(application) == ["always_fails"]


def test_failed_rules_is_empty_when_every_rule_passes(monkeypatch, application):
    monkeypatch.setattr(policy, "RULES", [always_passes])

    assert policy.failed_rules(application) == []


@pytest.mark.parametrize(("age", "expected"), [(17, "below_minimum_age"), (18, None), (41, None)])
def test_minimum_age(application, age, expected):
    assert policy.minimum_age(application.model_copy(update={"age": age})) == expected


@pytest.mark.parametrize(
    ("monthly_income", "expected"), [(None, "no_income"), (0, "no_income"), (1, None), (5200, None)]
)
def test_income_required(application, monthly_income, expected):
    applicant = application.model_copy(update={"monthly_income": monthly_income})

    assert policy.income_required(applicant) == expected


def test_failed_rules_reports_every_failed_rule(application):
    applicant = application.model_copy(update={"age": 16, "monthly_income": None})

    assert policy.failed_rules(applicant) == ["below_minimum_age", "no_income"]
