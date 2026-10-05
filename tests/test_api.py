import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from credit import policy
from credit.api import app, get_risk_model
from credit.db import DecisionLog, create_decisions_table, get_session
from credit.schemas import EXAMPLE_APPLICATION


@pytest.fixture
def engine():
    """In-memory SQLite, one shared connection so every session sees the same data."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    create_decisions_table(engine)
    return engine


@pytest.fixture
def client(engine, risk_model):
    def session_override():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_risk_model] = lambda: risk_model
    # Not used as a context manager, so the lifespan (real model, Postgres) does not run.
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_reports_versions(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "test-model",
        "policy_version": policy.POLICY_VERSION,
    }


def test_decision_rejects_impossible_values(client):
    response = client.post("/decision", json=EXAMPLE_APPLICATION | {"age": -1})

    assert response.status_code == 422


def test_decision_rejects_unknown_fields(client):
    response = client.post("/decision", json=EXAMPLE_APPLICATION | {"zip_code": "10001"})

    assert response.status_code == 422


def logged_decisions(engine) -> list[DecisionLog]:
    with Session(engine) as session:
        return list(session.scalars(select(DecisionLog)))


def test_decision_returns_and_logs_the_decision(client, engine):
    response = client.post("/decision", json=EXAMPLE_APPLICATION)

    assert response.status_code == 200
    assert response.json()["decision"] == "approve"
    [logged] = logged_decisions(engine)
    assert logged.decision == "approve"
    assert logged.application == EXAMPLE_APPLICATION
    assert logged.model_version == "test-model"


def test_policy_decline_is_returned_and_logged_without_a_score(client, engine):
    response = client.post("/decision", json=EXAMPLE_APPLICATION | {"monthly_income": None})

    assert response.json()["decision"] == "decline"
    assert response.json()["reason_codes"] == ["no_income"]
    assert response.json()["probability_of_default"] is None
    [logged] = logged_decisions(engine)
    assert logged.reason_codes == ["no_income"]
