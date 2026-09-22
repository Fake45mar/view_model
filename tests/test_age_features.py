"""Regression tests for the training-serving skew in ``age_days``.

The inherited code measured age against ``bundle["snapshot_ms"]`` -- a value
frozen when the model was trained. Any posting listed after that moment got a
negative age, which the model never saw during training. Since new postings are
the entire point of the service, this silently corrupted the main use case.
"""
from __future__ import annotations

import numpy as np
import pytest

from views_model.config import MS_PER_DAY
from views_model.features import build_features
from views_model.predict import load_bundle, predict_views

from conftest import BASE_MS, make_postings


def test_posting_listed_after_training_never_gets_negative_age(artifact_path):
    """The headline regression: a fresh posting must not get a negative age."""
    bundle = load_bundle(artifact_path)
    snapshot_ms = bundle["snapshot_ms"]

    # Listed five days after the training snapshot -- the old code produced
    # age_days = -5.0 here.
    five_days_later = snapshot_ms + 5 * MS_PER_DAY
    fresh = make_postings(n_rows=3, seed=7, first_listed_ms=int(five_days_later), span_days=0)

    feats = build_features(
        fresh,
        company_industry=bundle["company_industry"],
        snapshot_ms=five_days_later,
        ms_per_day=bundle["ms_per_day"],
    )

    assert (feats["age_days"] >= 0).all()
    assert (feats["days_since_relisted"] >= 0).all()


def test_predict_uses_request_time_not_the_frozen_snapshot(artifact_path):
    """Age must move with wall-clock time, not stay pinned to training."""
    bundle = load_bundle(artifact_path)
    posting = make_postings(n_rows=1, seed=3, first_listed_ms=BASE_MS, span_days=0)

    def age_at(now_ms: float) -> float:
        feats = build_features(
            posting,
            company_industry=bundle["company_industry"],
            snapshot_ms=now_ms,
            ms_per_day=bundle["ms_per_day"],
        )
        return float(feats["age_days"].iloc[0])

    day_10 = age_at(BASE_MS + 10 * MS_PER_DAY)
    day_40 = age_at(BASE_MS + 40 * MS_PER_DAY)

    assert np.isclose(day_10, 10.0)
    assert np.isclose(day_40, 40.0)


def test_predict_defaults_to_current_time(artifact_path):
    """Calling predict without now_ms must not fall back to snapshot_ms."""
    bundle = load_bundle(artifact_path)

    # Listed well after training. Under the old behaviour this row's age was
    # negative; the prediction still returned a number, which is what made the
    # bug invisible.
    fresh = make_postings(
        n_rows=2,
        seed=11,
        first_listed_ms=int(bundle["snapshot_ms"] + 30 * MS_PER_DAY),
        span_days=0,
    )

    # The old code is reproduced by passing the frozen snapshot explicitly.
    # If the default were still snapshot_ms these two would be identical.
    default = predict_views(fresh, bundle, target="views")
    frozen = predict_views(fresh, bundle, target="views", now_ms=bundle["snapshot_ms"])

    assert (default >= 0).all()
    assert not np.allclose(default, frozen), (
        "predict_views without now_ms still behaves like the frozen snapshot"
    )


def test_snapshot_ms_is_still_recorded_as_provenance(artifact_path):
    """We stopped serving from it, but it stays in the bundle for auditing."""
    bundle = load_bundle(artifact_path)
    assert bundle["snapshot_ms"] > 0


def test_views_per_day_age_floor_still_applies(artifact_path):
    """A brand-new posting divides by the age floor, not by zero."""
    bundle = load_bundle(artifact_path)
    now = BASE_MS + 100 * MS_PER_DAY
    brand_new = make_postings(n_rows=1, seed=5, first_listed_ms=now, span_days=0)

    preds = predict_views(brand_new, bundle, target="views_per_day", now_ms=now)

    assert len(preds) == 1
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()


@pytest.mark.parametrize("target", ["views", "views_per_day"])
def test_age_still_influences_the_prediction(artifact_path, target):
    """The frozen snapshot turned age_days into a constant: every posting,
    whatever its real age, landed in the same bin and got the same number.
    Asserting age_days >= 0 would not have caught that -- this does.
    """
    bundle = load_bundle(artifact_path)
    now = BASE_MS + 400 * MS_PER_DAY

    def predict_at_age(days: int) -> float:
        row = make_postings(
            n_rows=1, seed=4,
            first_listed_ms=int(now - days * MS_PER_DAY), span_days=0,
        )
        return float(predict_views(row, bundle, target=target, now_ms=now)[0])

    assert predict_at_age(5) != predict_at_age(90), (
        f"{target}: age_days is not influencing the prediction"
    )
