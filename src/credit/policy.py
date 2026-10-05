"""Hard policy rules, checked before the model. Any failed rule declines without scoring.

Bump POLICY_VERSION whenever a rule is added, removed or changed: it is logged with
every decision, so each one can be traced to the rules in force at the time.
"""

from collections.abc import Callable

from credit.schemas import Application

POLICY_VERSION = "1.0.0"

MIN_AGE = 18

# A rule returns its reason code when the application fails it, or None when it passes.
Rule = Callable[[Application], str | None]


def minimum_age(application: Application) -> str | None:
    """Decline applicants below the minimum age to borrow.

    A legal rule, not a risk one: the training data has no applicant under 21.
    Returns "below_minimum_age" when the rule fails, None otherwise.
    """
    if application.age < MIN_AGE:
        return "below_minimum_age"
    return None


def income_required(application: Application) -> str | None:
    """Decline applications with zero or missing monthly income.

    An affordability rule, not a risk one: without income, repayment capacity cannot be
    verified, even though this segment defaults less than average in the training data.
    Returns "no_income" when the rule fails, None otherwise.
    """
    if application.monthly_income is None or application.monthly_income == 0:
        return "no_income"
    return None


RULES: list[Rule] = [minimum_age, income_required]


def failed_rules(application: Application) -> list[str]:
    """Reason codes of every rule the application fails, in RULES order."""
    return [code for rule in RULES if (code := rule(application)) is not None]
