"""FastAPI serving layer for the views-prediction model."""
from __future__ import annotations

import os
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import DEFAULT_ARTIFACT_PATH
from .predict import load_bundle, predict_views

ARTIFACT_PATH = os.environ.get("VIEWS_MODEL_ARTIFACT", DEFAULT_ARTIFACT_PATH)

app = FastAPI(title="Job-posting views predictor", version=__version__)


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


@app.on_event("startup")
def _warm_cache() -> None:
    # Fail fast at boot if the artifact is missing or incompatible.
    load_bundle(ARTIFACT_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/metrics")
def metrics() -> dict:
    bundle = load_bundle(ARTIFACT_PATH)
    return {"metrics": bundle.get("metrics", []), "schema_version": bundle["schema_version"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not req.postings:
        raise HTTPException(status_code=400, detail="`postings` must be non-empty")
    bundle = load_bundle(ARTIFACT_PATH)
    raw = pd.DataFrame([p.model_dump() for p in req.postings])
    preds = predict_views(raw, bundle, target=req.target)
    return PredictResponse(target=req.target, predictions=[float(x) for x in preds])
