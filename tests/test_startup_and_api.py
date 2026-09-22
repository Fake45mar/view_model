"""Startup validation and the readiness contract.

The operational claim being tested: a bad artifact makes the service unready,
not dead. It keeps answering /health, explains itself on /ready, and refuses
to predict -- rather than crash-looping with the reason only in the logs.
"""
from __future__ import annotations

import importlib

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from views_model.config import SCHEMA_VERSION
from views_model.predict import load_bundle
from views_model.startup import validate_artifact

from conftest import BASE_MS, make_postings


# --------------------------------------------------------------------------
# validate_artifact
# --------------------------------------------------------------------------

def test_valid_artifact_is_ready(artifact_path):
    state = validate_artifact(artifact_path)
    assert state.ready
    assert state.reason is None
    assert state.checks["smoke_prediction"] == "ok"


def test_missing_artifact_reports_instead_of_raising(tmp_path):
    state = validate_artifact(str(tmp_path / "does_not_exist.joblib"))
    assert not state.ready
    assert "could not be loaded" in state.reason
    assert state.checks["artifact_loads"] == "fail"


def test_wrong_schema_version_is_not_ready(tmp_path, artifact_path):
    bundle = dict(load_bundle(artifact_path))
    bundle["schema_version"] = SCHEMA_VERSION + 1
    bad = tmp_path / "bad_schema.joblib"
    joblib.dump(bundle, bad)

    state = validate_artifact(str(bad))
    assert not state.ready
    assert state.checks["artifact_loads"] == "fail"


def test_sklearn_minor_version_mismatch_is_not_ready(tmp_path, artifact_path):
    bundle = dict(load_bundle(artifact_path))
    bundle["sklearn_version"] = "0.1.0"
    bad = tmp_path / "bad_sklearn.joblib"
    joblib.dump(bundle, bad)

    state = validate_artifact(str(bad))
    assert not state.ready
    assert "sklearn version mismatch" in state.reason


def test_missing_sklearn_version_is_not_ready(tmp_path, artifact_path):
    """Unverifiable provenance is refused, not warned about: without it,
    model rollback and code/model separation stop being checkable."""
    bundle = dict(load_bundle(artifact_path))
    bundle.pop("sklearn_version", None)
    bad = tmp_path / "no_sklearn.joblib"
    joblib.dump(bundle, bad)

    state = validate_artifact(str(bad))
    assert not state.ready
    assert "does not record sklearn_version" in state.reason


def test_feature_contract_mismatch_is_not_ready(tmp_path, artifact_path):
    bundle = dict(load_bundle(artifact_path))
    bundle["feature_cols"] = ["not", "the", "real", "features"]
    bad = tmp_path / "bad_features.joblib"
    joblib.dump(bundle, bad)

    state = validate_artifact(str(bad))
    assert not state.ready
    assert state.checks["feature_contract"] == "fail"


# --------------------------------------------------------------------------
# HTTP contract
# --------------------------------------------------------------------------

def _client(monkeypatch, artifact: str) -> TestClient:
    """Build an app bound to a specific artifact path."""
    monkeypatch.setenv("VIEWS_MODEL_ARTIFACT", artifact)
    import views_model.api as api

    importlib.reload(api)
    return TestClient(api.app)


def _payload(n: int = 2) -> dict:
    rows = make_postings(n_rows=n, seed=42, first_listed_ms=BASE_MS, span_days=1)
    # NaN is not JSON-encodable; cast to object first or pandas puts it back.
    rows = rows.astype(object).where(pd.notna(rows), None)
    return {"postings": rows.to_dict("records"), "target": "views"}


@pytest.fixture
def healthy_client(monkeypatch, artifact_path):
    with _client(monkeypatch, artifact_path) as client:
        yield client


@pytest.fixture
def broken_client(monkeypatch, tmp_path):
    with _client(monkeypatch, str(tmp_path / "missing.joblib")) as client:
        yield client


def test_ready_returns_200_when_model_loaded(healthy_client):
    r = healthy_client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_predict_works_when_ready(healthy_client):
    r = healthy_client.post("/predict", json=_payload())
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_model_info_replaces_metrics(healthy_client):
    assert healthy_client.get("/metrics").status_code == 404
    r = healthy_client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["schema_version"] == SCHEMA_VERSION


def test_batch_size_is_capped(healthy_client):
    import views_model.api as api

    r = healthy_client.post("/predict", json=_payload(api.MAX_BATCH_SIZE + 1))
    assert r.status_code == 413


def test_broken_artifact_keeps_process_alive(broken_client):
    """The whole point: no crash, /health still answers."""
    assert broken_client.get("/health").status_code == 200


def test_broken_artifact_is_not_ready_and_explains_why(broken_client):
    r = broken_client.get("/ready")
    assert r.status_code == 503
    assert r.json()["detail"]["ready"] is False
    assert r.json()["detail"]["reason"]


def test_broken_artifact_refuses_to_predict(broken_client):
    r = broken_client.post("/predict", json=_payload())
    assert r.status_code == 503
