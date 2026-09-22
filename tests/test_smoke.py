"""End-to-end smoke tests: train on synthetic data, round-trip through predict.

Runs entirely on fixtures from conftest.py — no dependency on data/.
"""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from views_model.config import FEATURE_COLS, SCHEMA_VERSION
from views_model.predict import load_bundle, predict_views
from views_model.train import train

from conftest import make_postings


def test_training_writes_an_artifact(artifact_path):
    assert pathlib.Path(artifact_path).exists()


def test_bundle_contains_both_models(artifact_path):
    bundle = load_bundle(artifact_path)
    assert set(bundle["models"]) == {"views", "views_per_day"}


def test_bundle_records_provenance(artifact_path):
    """Startup validation (step 5) depends on these keys being present."""
    bundle = load_bundle(artifact_path)
    assert bundle["schema_version"] == SCHEMA_VERSION
    assert bundle.get("sklearn_version")
    assert bundle["feature_cols"] == FEATURE_COLS


@pytest.mark.parametrize("target", ["views", "views_per_day"])
def test_predict_round_trip(artifact_path, target):
    bundle = load_bundle(artifact_path)
    sample = make_postings(n_rows=5, seed=99)

    preds = predict_views(sample, bundle, target=target)

    assert len(preds) == len(sample)
    assert (preds >= 0).all()
    assert pd.notna(preds).all()


def test_unknown_target_is_rejected(artifact_path):
    bundle = load_bundle(artifact_path)
    with pytest.raises(ValueError):
        predict_views(make_postings(n_rows=2), bundle, target="not_a_model")


def test_load_bundle_rejects_wrong_schema_version(tmp_path, artifact_path):
    """A bundle from an incompatible trainer must fail loudly, not serve."""
    import joblib

    bundle = dict(load_bundle(artifact_path))
    bundle["schema_version"] = SCHEMA_VERSION + 1
    bad_path = tmp_path / "bad.joblib"
    joblib.dump(bundle, bad_path)

    with pytest.raises(ValueError, match="schema_version"):
        load_bundle(str(bad_path))


def test_holdout_is_the_most_recent_slice(dataset, pipeline_params, tmp_path):
    """Finding #2: the inherited random split leaked the future into training.

    The holdout must be the newest rows, and every training row must predate
    every test row -- otherwise offline metrics are measured on data the model
    could not have had at serving time.
    """
    postings_path, companies_path = dataset
    bundle = train(
        postings_path=postings_path,
        companies_path=companies_path,
        out_path=str(tmp_path / "split.joblib"),
        test_size=0.2,
        pipeline_params=pipeline_params,
    )

    listed = pd.read_csv(postings_path)["original_listed_time"]
    boundary = bundle["split_time_ms"]

    n_after = int((listed >= boundary).sum())
    assert n_after == round(len(listed) * 0.2), (
        f"holdout is {n_after} rows, expected {round(len(listed) * 0.2)}"
    )
    assert listed[listed < boundary].max() < boundary
