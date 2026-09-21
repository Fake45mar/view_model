"""Train both regressors and save a self-contained inference bundle.

Usage::

    python -m views_model.train \
        --postings data/postings.csv \
        --companies data/companies/company_industries.csv \
        --out artifacts/views_baseline.joblib
"""
from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import (
    AGE_FLOOR_DAYS,
    CATEGORICAL_FEATURES,
    DEFAULT_ARTIFACT_PATH,
    FEATURE_COLS,
    MS_PER_DAY,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SCHEMA_VERSION,
    TEXT_FEATURE,
)
from .features import build_features, load_company_industry
from .pipeline import build_pipeline

logger = logging.getLogger("views_model.train")


def _evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    logger.info("%-30s MAE=%8.2f  RMSE=%8.2f  R2=%6.3f", name, mae, rmse, r2)
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}


def train(
    postings_path: str,
    companies_path: str,
    out_path: str = DEFAULT_ARTIFACT_PATH,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Train the two baseline models and write the artifact bundle."""
    logger.info("Loading data: %s, %s", postings_path, companies_path)
    postings = pd.read_csv(postings_path)
    company_industry = load_company_industry(companies_path)

    # Snapshot reference (max listed time across the training data).
    snapshot_ms = float(
        max(postings["listed_time"].max(), postings["original_listed_time"].max())
    )

    feats = build_features(postings, company_industry, snapshot_ms)
    feats = feats.dropna(subset=["views"]).copy()
    feats["views"] = feats["views"].astype(float)

    X = feats[FEATURE_COLS]
    y = feats["views"]
    logger.info("Modelling rows: %d  features: %d", len(feats), len(FEATURE_COLS))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    # --- Model 1: predict views directly ----------------------------------
    model_views = build_pipeline()
    model_views.fit(X_train, np.log1p(y_train))

    # --- Model 2: predict views per day -----------------------------------
    age_train = feats.loc[X_train.index, "age_days"].clip(lower=AGE_FLOOR_DAYS)
    age_test = feats.loc[X_test.index, "age_days"].clip(lower=AGE_FLOOR_DAYS)
    model_vpd = build_pipeline()
    model_vpd.fit(X_train, np.log1p(y_train / age_train))

    # --- Evaluation -------------------------------------------------------
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    pred_dummy = dummy.predict(X_test)
    pred_views = np.clip(np.expm1(model_views.predict(X_test)), 0, None)
    pred_vpd = np.clip(np.expm1(model_vpd.predict(X_test)), 0, None) * age_test.values

    metrics = [
        _evaluate("mean baseline", y_test, pred_dummy),
        _evaluate("HGB on views", y_test, pred_views),
        _evaluate("HGB on views/day -> views", y_test, pred_vpd),
    ]

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "models": {"views": model_views, "views_per_day": model_vpd},
        "feature_cols": FEATURE_COLS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "text_feature": TEXT_FEATURE,
        "company_industry": company_industry,
        "snapshot_ms": snapshot_ms,
        "ms_per_day": MS_PER_DAY,
        "age_floor_days": AGE_FLOOR_DAYS,
        "target_transform": "log1p",
        "metrics": metrics,
    }

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    logger.info("Saved artifact -> %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return bundle


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--postings", default="data/postings.csv")
    p.add_argument("--companies", default="data/companies/company_industries.csv")
    p.add_argument("--out", default=DEFAULT_ARTIFACT_PATH)
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    train(args.postings, args.companies, args.out, args.test_size)


if __name__ == "__main__":
    main()
