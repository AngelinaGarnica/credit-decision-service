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


# TODO(owner): one test per rule (minimum_age, income_required), including the boundary values.
