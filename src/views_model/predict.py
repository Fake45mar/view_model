"""Load a saved bundle and run end-to-end inference on raw postings."""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_ARTIFACT_PATH, SCHEMA_VERSION
from .features import build_features

Target = str  # "views" | "views_per_day"


@lru_cache(maxsize=4)
def load_bundle(path: str = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    """Load and validate an artifact bundle (cached per path)."""
    bundle = joblib.load(path)
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible schema_version: expected {SCHEMA_VERSION}, "
            f"got {bundle.get('schema_version')}"
        )
    return bundle


def predict_views(
    raw: pd.DataFrame,
    bundle: dict[str, Any],
    target: Target = "views",
    now_ms: float | None = None,
) -> np.ndarray:
    """Predict total views for one or more raw posting rows.

    target='views'         -> direct model
    target='views_per_day' -> rate model, converted back to total views

    ``now_ms`` is the reference time age features are measured against and
    defaults to the moment of the call. It is deliberately *not*
    ``bundle["snapshot_ms"]``: that value is frozen at training, so using it
    here gives any posting listed after the training run a negative age --
    an input the model has never seen. The bundle keeps ``snapshot_ms`` as
    training provenance only.
    """
    if target not in bundle["models"]:
        raise ValueError(f"target must be one of {list(bundle['models'])}")

    if now_ms is None:
        now_ms = time.time() * 1000

    feats = build_features(
        raw,
        company_industry=bundle["company_industry"],
        snapshot_ms=now_ms,
        ms_per_day=bundle["ms_per_day"],
    )
    X = feats[bundle["feature_cols"]]
    pred = np.clip(np.expm1(bundle["models"][target].predict(X)), 0, None)

    if target == "views_per_day":
        age = feats["age_days"].clip(lower=bundle["age_floor_days"]).to_numpy()
        pred = pred * age
    return pred
