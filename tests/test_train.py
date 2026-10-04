import numpy as np
import pandas as pd
import pytest

from credit.train import UnitEconomics, choose_threshold, ks_statistic, outcome_at, profit_curve

ECONOMICS = UnitEconomics(principal=100.0, fee_rate=0.1, loss_given_default=1.0)  # +10 / -100


def test_ks_is_one_for_perfect_separation():
    y = pd.Series([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_breakeven_pd_gives_zero_expected_profit():
    p = ECONOMICS.breakeven_pd

    assert p == pytest.approx(10 / 110)
    assert (1 - p) * ECONOMICS.gain_per_good - p * ECONOMICS.loss_per_bad == pytest.approx(0)


def test_outcome_at_counts_only_approved_applicants():
    y = pd.Series([0, 0, 0, 1])
    pd_hat = np.array([0.05, 0.10, 0.50, 0.08])

    outcome = outcome_at(y, pd_hat, threshold=0.2, economics=ECONOMICS)

    # Approved: two goods (+10 each) and one bad (-100), over 4 applicants.
    assert outcome["approval_rate"] == 0.75
    assert outcome["bad_rate_approved"] == pytest.approx(1 / 3)
    assert outcome["profit_per_applicant"] == pytest.approx((2 * 10 - 100) / 4)


def test_profit_curve_has_one_row_per_cutoff():
    curve = profit_curve(pd.Series([0, 1]), np.array([0.1, 0.9]), ECONOMICS)

    assert len(curve) == 199
    assert curve["approval_rate"].is_monotonic_increasing


def test_choose_threshold_takes_the_lowest_cutoff_with_maximum_profit():
    curve = pd.DataFrame(
        {
            "threshold": [0.20, 0.05, 0.10, 0.15],
            "profit_per_applicant": [3.0, 1.0, 5.0, 5.0],  # 0.10 and 0.15 tie for the maximum
        }
    )

    assert choose_threshold(curve) == 0.10
