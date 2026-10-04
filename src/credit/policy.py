"""Hard policy rules, checked before the model. Any failed rule declines without scoring.

Bump POLICY_VERSION whenever a rule is added, removed or changed: it is logged with
every decision, so each one can be traced to the rules in force at the time.
"""

from collections.abc import Callable

from credit.schemas import Application

POLICY_VERSION = "1.0.0"

# A rule returns its reason code when the application fails it, or None when it passes.
Rule = Callable[[Application], str | None]


def minimum_age(application: Application) -> str | None:
    """Decline applicants below the minimum age to borrow.

    Returns "below_minimum_age" when the rule fails, None otherwise.
    """
    # TODO(owner): implement the rule.
    raise NotImplementedError


def income_required(application: Application) -> str | None:
    """Decline applications with zero or missing monthly income.

    Returns "no_income" when the rule fails, None otherwise.
    """
    # TODO(owner): implement the rule.
    raise NotImplementedError


RULES: list[Rule] = [minimum_age, income_required]


def failed_rules(application: Application) -> list[str]:
    """Reason codes of every rule the application fails, in RULES order."""
    return [code for rule in RULES if (code := rule(application)) is not None]
