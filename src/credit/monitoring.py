"""Monitor the decision engine: feature drift (PSI), approval rate, and a business summary.

Run with: python -m credit.monitoring

Compares the training sample with a simulated new month of applicants, runs both a baseline
and the new month through the same `decide` path the API uses, and asks Claude to explain the
results in business language. The metrics never depend on the LLM: without credentials, the
summary is skipped.
"""

import json
import logging
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd

from credit import policy
from credit.config import get_settings
from credit.decision import RiskModel, decide
from credit.features import FEATURES
from credit.schemas import Application
from credit.train import load_applicants, split

logger = logging.getLogger(__name__)

SEED = 7
SAMPLE_SIZE = 5000
PSI_BINS = 10
PSI_MODERATE, PSI_MAJOR = 0.10, 0.25  # industry rule of thumb
SUMMARY_MODEL = "claude-opus-5-5"

SUMMARY_INSTRUCTIONS = """\
You write the monthly monitoring note of a credit decision engine for risk and business \
managers who do not read code. You receive the monitoring metrics as JSON.

Write at most 200 words in Markdown with four short parts: a one-line headline; what changed \
in the applicant population (name the features with moderate or major drift); the business \
impact on approvals and risk; and one recommended next step. Use only numbers present in the \
JSON, and say "PSI" once with a plain explanation (how much a feature's distribution moved)."""


# --- Drift --------------------------------------------------------------------


def psi(expected: pd.Series, actual: pd.Series, bins: int = PSI_BINS) -> float:
    """Population Stability Index of `actual` against `expected`.

    Bins are the deciles of `expected`, so zero-heavy counts collapse into "0" vs "more".
    Missing values get their own bin: a change in missingness is drift too.
    """
    deciles = np.nanquantile(expected, np.linspace(0, 1, bins + 1)[1:-1])
    edges = np.concatenate([[-np.inf], np.unique(deciles), [np.inf]])

    def shares(values: pd.Series) -> np.ndarray:
        counts = pd.cut(values, edges).value_counts(sort=False).tolist() + [values.isna().sum()]
        return np.clip(np.array(counts) / len(values), 1e-4, None)  # avoid log(0)

    expected_share, actual_share = shares(expected), shares(actual)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def drift_status(value: float) -> str:
    if value >= PSI_MAJOR:
        return "major"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI and status per model feature, largest drift first."""
    rows = [
        {"feature": feature, "psi": round(psi(reference[feature], current[feature]), 4)}
        for feature in FEATURES
    ]
    drift = pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
    drift["status"] = drift["psi"].map(drift_status)
    return drift


# --- Decisions ----------------------------------------------------------------


def simulate_new_month(applicants: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """A new month of applicants after opening an acquisition channel that skips income.

    Applicants without reported income are three times as likely to be drawn. Income drifts
    directly, and so does debt_ratio, which holds dollars instead of a ratio without income.
    """
    weights = np.where(applicants["monthly_income"].isna(), 3.0, 1.0)
    return applicants.sample(n=size, weights=weights, random_state=seed)


def decide_all(applicants: pd.DataFrame, risk_model: RiskModel) -> pd.DataFrame:
    """Run every applicant through `decide`, the same path the API uses."""
    records = applicants[FEATURES].astype(object).where(applicants[FEATURES].notna(), None)
    results = [decide(Application(**record), risk_model) for record in records.to_dict("records")]
    return pd.DataFrame([result.model_dump() for result in results])


def decision_metrics(decisions: pd.DataFrame) -> dict[str, float]:
    scored = decisions["probability_of_default"].notna()
    return {
        "applicants": len(decisions),
        "approval_rate": round(float((decisions["decision"] == "approve").mean()), 4),
        "declined_by_policy": round(float((~scored).mean()), 4),
        "mean_probability_of_default": round(float(decisions["probability_of_default"].mean()), 4),
    }


# --- Summary ------------------------------------------------------------------


def business_summary(metrics: dict, client: anthropic.Anthropic) -> str:
    """Plain-language monitoring note written by Claude from the metrics only."""
    response = client.beta.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # on a policy refusal, the API retries on a fallback model
        system=SUMMARY_INSTRUCTIONS,
        messages=[{"role": "user", "content": json.dumps(metrics, indent=2)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The summary request was refused")
    return "".join(block.text for block in response.content if block.type == "text")


def summary_or_none(metrics: dict, api_key: str | None) -> str | None:
    """The business summary, or None when Claude is unavailable: monitoring must not fail on it."""
    try:
        return business_summary(metrics, anthropic.Anthropic(api_key=api_key or None))
    # The SDK raises TypeError, not AnthropicError, when it finds no credentials at all.
    except (anthropic.AnthropicError, RuntimeError, TypeError) as error:
        logger.warning("Summary skipped: %s", error)
        return None


# --- Report -------------------------------------------------------------------


def render_report(metrics: dict, drift: pd.DataFrame, summary: str | None) -> str:
    baseline, current = metrics["baseline"], metrics["new_month"]
    lines = [
        "# Monitoring report",
        "",
        f"Model `{metrics['model_version']}`, policy `{metrics['policy_version']}`. "
        "New month: simulated acquisition channel that oversamples applicants without income.",
        "",
        "## Summary",
        "",
        summary or "_Skipped: Claude was unavailable (set ANTHROPIC_API_KEY to enable it)._",
        "",
        "## Decisions",
        "",
        "| Metric | Baseline | New month |",
        "|---|---|---|",
        *(f"| {key} | {baseline[key]} | {current[key]} |" for key in baseline),
        "",
        f"## Feature drift (PSI; moderate >= {PSI_MODERATE}, major >= {PSI_MAJOR})",
        "",
        "| Feature | PSI | Status |",
        "|---|---|---|",
        *(f"| {row.feature} | {row.psi:.4f} | {row.status} |" for row in drift.itertuples()),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    settings = get_settings()
    risk_model = RiskModel.load(settings.artifacts_dir)
    train, _, test = split(load_applicants())

    baseline = test.sample(n=SAMPLE_SIZE, random_state=SEED)
    new_month = simulate_new_month(test, SAMPLE_SIZE, SEED)
    drift = feature_drift(train, new_month)
    metrics = {
        "model_version": risk_model.model_version,
        "policy_version": policy.POLICY_VERSION,
        "baseline": decision_metrics(decide_all(baseline, risk_model)),
        "new_month": decision_metrics(decide_all(new_month, risk_model)),
        "feature_drift": drift.to_dict("records"),
    }

    summary = summary_or_none(metrics, settings.anthropic_api_key)
    report_path = Path(settings.reports_dir) / "monitoring.md"
    report_path.write_text(render_report(metrics, drift, summary))
    logger.info("Metrics: %s", {k: v for k, v in metrics.items() if k != "feature_drift"})
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
