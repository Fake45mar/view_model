"""FastAPI serving layer for the views-prediction model."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from . import __version__
from .config import DEFAULT_ARTIFACT_PATH
from .predict import predict_views
from .startup import Readiness, validate_artifact

logger = logging.getLogger("views_model.api")

ARTIFACT_PATH = os.environ.get("VIEWS_MODEL_ARTIFACT", DEFAULT_ARTIFACT_PATH)

# Largest batch a single /predict call will accept. Without a ceiling one
# request can pin the worker for an unbounded time.
MAX_BATCH_SIZE = int(os.environ.get("VIEWS_MODEL_MAX_BATCH", "500"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate the artifact at boot without taking the process down.

    A failed check leaves the app running and unready: /health stays 200 so an
    operator can reach it, /ready returns 503 with the reason, and no traffic
    is routed. Crashing here instead would give a restart loop and no way to
    ask the container what went wrong.
    """
    app.state.readiness = validate_artifact(ARTIFACT_PATH)
    if not app.state.readiness.ready:
        logger.error("Service started UNREADY: %s", app.state.readiness.reason)
    yield


app = FastAPI(title="Job-posting views predictor", version=__version__, lifespan=lifespan)


class Posting(BaseModel):
    """Minimal raw-posting schema the model needs. Optional fields can be null."""

    job_id: Optional[int] = None
    company_id: Optional[float] = None
    title: Optional[str] = ""
    description: Optional[str] = ""
    skills_desc: Optional[str] = None
    normalized_salary: Optional[float] = None
    remote_allowed: Optional[float] = None
    sponsored: Optional[float] = None
    formatted_work_type: Optional[str] = None
    formatted_experience_level: Optional[str] = None
    pay_period: Optional[str] = None
    application_type: Optional[str] = None
    original_listed_time: float = Field(..., description="Epoch ms when first listed.")
    listed_time: float = Field(..., description="Epoch ms when (re)listed.")
    expiry: float = Field(..., description="Epoch ms when the posting expires.")


class PredictRequest(BaseModel):
    postings: list[Posting]
    target: Literal["views", "views_per_day"] = "views"


class PredictResponse(BaseModel):
    target: str
    predictions: list[float]


def require_ready(request: Request) -> Readiness:
    """Gate for every endpoint that needs a working model.

    Returns 503 with the failing check rather than letting a request reach a
    model that did not pass startup validation. /health deliberately does not
    use this: liveness must stay answerable when the model is broken.
    """
    state: Readiness = request.app.state.readiness
    if not state.ready:
        raise HTTPException(status_code=503, detail=state.as_dict())
    return state


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: is the process up. Deliberately independent of the model."""
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready(state: Readiness = Depends(require_ready)) -> dict:
    """Readiness: can this instance serve a prediction right now."""
    return state.as_dict()


@app.get("/model-info")
def model_info(state: Readiness = Depends(require_ready)) -> dict:
    """Training metrics and artifact provenance.

    Named /model-info rather than /metrics so the conventional Prometheus
    scrape path stays free for real runtime metrics.
    """
    bundle = state.bundle
    return {
        "metrics": bundle.get("metrics", []),
        "schema_version": bundle["schema_version"],
        "sklearn_version": bundle.get("sklearn_version"),
        "snapshot_ms": bundle.get("snapshot_ms"),
        "split_time_ms": bundle.get("split_time_ms"),
        "feature_cols": bundle.get("feature_cols"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    state: Readiness = Depends(require_ready),
) -> PredictResponse:
    if not req.postings:
        raise HTTPException(status_code=400, detail="`postings` must be non-empty")
    if len(req.postings) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"batch of {len(req.postings)} exceeds limit of {MAX_BATCH_SIZE}",
        )

    raw = pd.DataFrame([p.model_dump() for p in req.postings])
    preds = predict_views(raw, state.bundle, target=req.target)
    return PredictResponse(target=req.target, predictions=[float(x) for x in preds])
