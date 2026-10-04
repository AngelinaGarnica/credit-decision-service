"""Train the risk model, calibrate it, choose the approval threshold, and save everything.

Run with: python -m credit.train

Splits: train (fit models) / validation (calibrate, choose threshold) / test (report only).
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from credit.config import get_settings
from credit.db import Applicant, get_engine
from credit.features import FEATURES, TARGET, feature_matrix
from credit.plotting import PALETTE_MAIN, PALETTE_NEUTRAL, PALETTE_POS, apply_style

logger = logging.getLogger(__name__)

SEED = 42


@dataclass(frozen=True)
class UnitEconomics:
    """Illustrative economics of one advance. Change the assumptions here, not the code."""

    principal: float = 500.0
    fee_rate: float = 0.10  # revenue on an advance that is repaid
    loss_given_default: float = 1.0  # share of the principal lost when it is not

    @property
    def gain_per_good(self) -> float:
        return self.principal * self.fee_rate

    @property
    def loss_per_bad(self) -> float:
        return self.principal * self.loss_given_default

    @property
    def breakeven_pd(self) -> float:
        """PD at which approving has zero expected profit: (1 - p) * gain = p * loss.

        With calibrated probabilities, the profit-maximizing cutoff should land close to it.
        """
        return self.gain_per_good / (self.gain_per_good + self.loss_per_bad)


# --- Data --------------------------------------------------------------------


def load_applicants() -> pd.DataFrame:
    return pd.read_sql_table(Applicant.__tablename__, get_engine())


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """60 / 20 / 20, stratified on the target."""
    train, rest = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=SEED)
    valid, test = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=SEED)
    return train, valid, test


# --- Models ------------------------------------------------------------------


def train_logistic_baseline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Linear baseline: median imputation (plus missing flags), scaling, logistic regression."""
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    return model.fit(X, y)


def train_lightgbm(X: pd.DataFrame, y: pd.Series) -> LGBMClassifier:
    """Gradient boosting with sensible defaults; handles missing values natively."""
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1)
    return model.fit(X, y)


def raw_scores(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def fit_calibrator(scores: np.ndarray, y: pd.Series) -> IsotonicRegression:
    """Map raw scores to probabilities of default. Monotone, so the ranking is unchanged."""
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(scores, y)


# --- Metrics -----------------------------------------------------------------


def ks_statistic(y: pd.Series, scores: np.ndarray) -> float:
    """Max distance between the score distributions of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y, scores)
    return float(np.max(tpr - fpr))


def evaluate(y: pd.Series, pd_hat: np.ndarray) -> dict[str, float]:
    return {
        "auc": round(float(roc_auc_score(y, pd_hat)), 4),
        "ks": round(ks_statistic(y, pd_hat), 4),
        "brier": round(float(brier_score_loss(y, pd_hat)), 4),
    }


# --- Threshold ---------------------------------------------------------------


def outcome_at(
    y: pd.Series, pd_hat: np.ndarray, threshold: float, economics: UnitEconomics
) -> dict[str, float]:
    """Realized results of approving everyone with PD < threshold."""
    y = np.asarray(y)
    approved = pd_hat < threshold
    goods = np.sum(approved & (y == 0))
    bads = np.sum(approved & (y == 1))
    profit = goods * economics.gain_per_good - bads * economics.loss_per_bad
    return {
        "threshold": round(float(threshold), 3),
        "approval_rate": float(approved.mean()),
        "bad_rate_approved": float(bads / approved.sum()) if approved.any() else 0.0,
        "profit_per_applicant": float(profit / len(y)),
    }


def profit_curve(y: pd.Series, pd_hat: np.ndarray, economics: UnitEconomics) -> pd.DataFrame:
    """`outcome_at` for every candidate cutoff from 0.005 to 0.995."""
    cutoffs = np.arange(0.005, 1.0, 0.005)
    return pd.DataFrame([outcome_at(y, pd_hat, t, economics) for t in cutoffs])


def choose_threshold(curve: pd.DataFrame) -> float:
    """Choose the approval cutoff from the validation profit curve.

    `curve` has one row per candidate cutoff, with columns `threshold`,
    `approval_rate`, `bad_rate_approved` and `profit_per_applicant`.
    An applicant is approved when their probability of default is below the cutoff.
    """
    # TODO(owner): implement the threshold choice.
    raise NotImplementedError


# --- Outputs -----------------------------------------------------------------


def plot_profit_curve(
    curve: pd.DataFrame, threshold: float, breakeven_pd: float, path: Path
) -> None:
    apply_style()
    fig, (ax_profit, ax_approval) = plt.subplots(1, 2)

    ax_profit.plot(curve["threshold"], curve["profit_per_applicant"], color=PALETTE_MAIN)
    ax_profit.axhline(0, color=PALETTE_NEUTRAL, linestyle="--", alpha=0.7)
    ax_profit.set_title("Profit per applicant by cutoff (validation)")
    ax_profit.set_xlabel("Cutoff: approve when PD < cutoff")
    ax_profit.set_ylabel("Profit per applicant ($)")

    ax_approval.plot(curve["threshold"], curve["approval_rate"], color=PALETTE_POS)
    ax_approval.set_title("Approval rate by cutoff (validation)")
    ax_approval.set_xlabel("Cutoff: approve when PD < cutoff")
    ax_approval.set_ylabel("Approval rate")

    for ax in (ax_profit, ax_approval):
        ax.axvline(threshold, color=PALETTE_NEUTRAL, linestyle=":", label=f"chosen = {threshold}")
        ax.axvline(
            breakeven_pd,
            color=PALETTE_NEUTRAL,
            linestyle="-.",
            label=f"breakeven PD = {breakeven_pd:.3f}",
        )
        ax.legend()

    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    settings = get_settings()
    economics = UnitEconomics()

    train, valid, test = split(load_applicants())
    X_train, X_valid, X_test = (feature_matrix(df) for df in (train, valid, test))
    y_train, y_valid, y_test = train[TARGET], valid[TARGET], test[TARGET]

    models = {
        "logistic_baseline": train_logistic_baseline(X_train, y_train),
        "lightgbm": train_lightgbm(X_train, y_train),
    }
    calibrators = {
        name: fit_calibrator(raw_scores(model, X_valid), y_valid) for name, model in models.items()
    }
    test_metrics = {
        name: evaluate(y_test, calibrators[name].predict(raw_scores(model, X_test)))
        for name, model in models.items()
    }
    logger.info("Test metrics: %s", test_metrics)

    # LightGBM is the served model: TreeSHAP gives per-applicant reason codes.
    # The baseline's test metrics are saved next to it, to justify the extra complexity.
    model, calibrator = models["lightgbm"], calibrators["lightgbm"]
    pd_valid = calibrator.predict(raw_scores(model, X_valid))
    pd_test = calibrator.predict(raw_scores(model, X_test))

    curve = profit_curve(y_valid, pd_valid, economics)
    threshold = choose_threshold(curve)
    test_at_threshold = outcome_at(y_test, pd_test, threshold, economics)
    logger.info("Chosen threshold on test: %s", test_at_threshold)
    logger.info("Breakeven PD (control): %.4f", economics.breakeven_pd)

    model_version = f"lgbm-{datetime.now(UTC):%Y%m%d%H%M}"
    metadata = {
        "model_version": model_version,
        "features": FEATURES,
        "threshold": threshold,
        "breakeven_pd": round(economics.breakeven_pd, 4),
        "decision_rule": "approve when probability_of_default < threshold",
        "unit_economics": asdict(economics),
        "test_metrics": test_metrics,
        "test_at_threshold": {k: round(v, 4) for k, v in test_at_threshold.items()},
        "rows": {"train": len(train), "validation": len(valid), "test": len(test)},
    }

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator}, settings.artifacts_dir / "model.joblib")
    (settings.artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (settings.reports_dir / "training_summary.json").write_text(json.dumps(metadata, indent=2))
    curve.to_csv(settings.reports_dir / "profit_curve.csv", index=False)
    plot_profit_curve(
        curve, threshold, economics.breakeven_pd, settings.reports_dir / "profit_curve.png"
    )
    logger.info(
        "Saved %s to %s and reports to %s",
        model_version,
        settings.artifacts_dir,
        settings.reports_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
