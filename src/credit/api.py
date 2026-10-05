"""HTTP API: POST /decision decides on one application and logs it; GET /health.

Run with: uvicorn credit.api:app
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from credit import policy
from credit.config import get_settings
from credit.db import create_decisions_table, get_engine, get_session, log_decision
from credit.decision import RiskModel, decide
from credit.schemas import Application, DecisionResult


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Fail fast: load the model and prepare the `decisions` table before serving."""
    app.state.risk_model = RiskModel.load(get_settings().artifacts_dir)
    create_decisions_table(get_engine())
    yield


app = FastAPI(title="Credit decision service", lifespan=lifespan)


def get_risk_model(request: Request) -> RiskModel:
    return request.app.state.risk_model


SessionDep = Annotated[Session, Depends(get_session)]
RiskModelDep = Annotated[RiskModel, Depends(get_risk_model)]


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
    policy_version: str


@app.get("/health")
def health(risk_model: RiskModelDep) -> Health:
    return Health(
        status="ok",
        model_version=risk_model.model_version,
        policy_version=policy.POLICY_VERSION,
    )


@app.post("/decision")
def post_decision(
    application: Application, session: SessionDep, risk_model: RiskModelDep
) -> DecisionResult:
    """Decide on one application, log it in the `decisions` table, and return the decision."""
    result = decide(application, risk_model)
    log_decision(session, application, result)
    return result
