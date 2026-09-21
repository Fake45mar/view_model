"""Smoke test: train on a tiny slice and round-trip through predict_views."""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from views_model.predict import load_bundle, predict_views  # noqa: E402
from views_model.train import train  # noqa: E402


def test_train_and_predict(tmp_path):
    # Subsample for speed.
    raw = pd.read_csv(ROOT / "data" / "postings.csv").sample(2000, random_state=0)
    postings_path = tmp_path / "postings.csv"
    raw.to_csv(postings_path, index=False)

    out = tmp_path / "bundle.joblib"
    bundle = train(
        postings_path=str(postings_path),
        companies_path=str(ROOT / "data" / "companies" / "company_industries.csv"),
        out_path=str(out),
    )
    assert out.exists()
    assert set(bundle["models"]) == {"views", "views_per_day"}

    loaded = load_bundle(str(out))
    sample = raw.head(5)
    preds_v = predict_views(sample, loaded, target="views")
    preds_r = predict_views(sample, loaded, target="views_per_day")
    assert len(preds_v) == len(sample) == len(preds_r)
    assert (preds_v >= 0).all() and (preds_r >= 0).all()
