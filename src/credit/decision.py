"""Decide on one application: policy rules first, then the calibrated model, then the threshold."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

from credit import policy
from credit.features import feature_matrix
from credit.schemas import Application, DecisionResult

TOP_REASONS = 3


@dataclass(frozen=True)
class RiskModel:
    """The served model: LightGBM, its isotonic calibrator and the approval threshold."""

    model: LGBMClassifier
    calibrator: IsotonicRegression
    threshold: float
    model_version: str

    @classmethod
    def load(cls, artifacts_dir: Path) -> "RiskModel":
        """Load what `credit.train` wrote to `artifacts/`."""
        bundle = joblib.load(artifacts_dir / "model.joblib")
        metadata = json.loads((artifacts_dir / "metadata.json").read_text())
        return cls(
            model=bundle["model"],
            calibrator=bundle["calibrator"],
            threshold=metadata["threshold"],
            model_version=metadata["model_version"],
        )

    def probability_of_default(self, features: pd.DataFrame) -> float:
        raw_score = self.model.predict_proba(features)[:, 1]
        return float(self.calibrator.predict(raw_score)[0])

    def reason_codes(self, features: pd.DataFrame) -> list[str]:
        """The features that push this applicant's risk up the most, by SHAP value.

        LightGBM computes TreeSHAP natively: `pred_contrib` returns one contribution per
        feature (in log-odds), plus the base value in the last column.
        """
        contributions = self.model.predict(features, pred_contrib=True)[0, :-1]
        largest_first = np.argsort(contributions)[::-1][:TOP_REASONS]
        return [features.columns[i] for i in largest_first]


def decide(application: Application, risk_model: RiskModel) -> DecisionResult:
    """Decline on any failed policy rule; otherwise approve when PD < threshold."""
    failed_rules = policy.failed_rules(application)
    if failed_rules:
        return DecisionResult(
            decision="decline",
            probability_of_default=None,
            reason_codes=failed_rules,
            policy_version=policy.POLICY_VERSION,
            model_version=risk_model.model_version,
        )

    features = feature_matrix(pd.DataFrame([application.model_dump()]))
    probability_of_default = risk_model.probability_of_default(features)
    approved = probability_of_default < risk_model.threshold
    return DecisionResult(
        decision="approve" if approved else "decline",
        probability_of_default=probability_of_default,
        reason_codes=risk_model.reason_codes(features),
        policy_version=policy.POLICY_VERSION,
        model_version=risk_model.model_version,
    )
