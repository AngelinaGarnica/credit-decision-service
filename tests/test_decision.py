import json

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

from credit import policy
from credit.decision import TOP_REASONS, RiskModel, decide
from credit.features import FEATURES


@pytest.fixture
def no_rules(monkeypatch):
    monkeypatch.setattr(policy, "RULES", [])


def test_failed_rule_declines_without_scoring(monkeypatch, application, risk_model):
    monkeypatch.setattr(policy, "RULES", [lambda application: "no_income"])

    result = decide(application, risk_model)

    assert result.decision == "decline"
    assert result.reason_codes == ["no_income"]
    assert result.probability_of_default is None
    assert risk_model.times_scored == 0


def test_approves_below_threshold(no_rules, application, risk_model):
    risk_model.probability, risk_model.threshold = 0.05, 0.10

    result = decide(application, risk_model)

    assert result.decision == "approve"
    assert result.probability_of_default == 0.05
    assert result.reason_codes == ["debt_ratio", "revolving_utilization", "age"]


def test_declines_at_threshold(no_rules, application, risk_model):
    risk_model.probability, risk_model.threshold = 0.10, 0.10

    assert decide(application, risk_model).decision == "decline"


def test_result_carries_policy_and_model_versions(no_rules, application, risk_model):
    result = decide(application, risk_model)

    assert result.policy_version == policy.POLICY_VERSION
    assert result.model_version == "test-model"


# --- RiskModel with a real (tiny) LightGBM -------------------------------------


@pytest.fixture
def tiny_risk_model(tmp_path) -> RiskModel:
    """Train on synthetic data where only past_due_90 drives default, save like train.py."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 1, size=(1000, len(FEATURES))), columns=FEATURES)
    X["past_due_90"] = rng.integers(0, 4, size=1000)
    y = (X["past_due_90"] >= 2).astype(int)

    model = LGBMClassifier(n_estimators=20, verbose=-1).fit(X, y)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(X)[:, 1], y)
    joblib.dump({"model": model, "calibrator": calibrator}, tmp_path / "model.joblib")
    metadata = {"model_version": "lgbm-test", "threshold": 0.5}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    return RiskModel.load(tmp_path)


def test_risk_model_loads_threshold_and_version(tiny_risk_model):
    assert tiny_risk_model.threshold == 0.5
    assert tiny_risk_model.model_version == "lgbm-test"


def test_risk_model_ranks_the_driving_feature_first(no_rules, application, tiny_risk_model):
    risky = application.model_copy(update={"past_due_90": 3})

    result = decide(risky, tiny_risk_model)

    assert result.decision == "decline"
    assert 0 <= result.probability_of_default <= 1
    assert len(result.reason_codes) == TOP_REASONS
    assert result.reason_codes[0] == "past_due_90"
