"""Artifact validation run at process start.

The service must not serve predictions from an artifact it cannot vouch for,
but it must also not crash on one: a crashing container gives an operator a
restart loop and no diagnostics. So every check here returns a result instead
of raising, the process stays up, and ``/ready`` reports what failed.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import sklearn

from .config import FEATURE_COLS, MS_PER_DAY
from .predict import load_bundle, predict_views

logger = logging.getLogger("views_model.startup")

# A single synthetic posting used to prove the model can actually score a
# request. Values are arbitrary but must satisfy every column build_features
# reads, so that a missing feature shows up here rather than on live traffic.
_SMOKE_POSTING_FIELDS: dict[str, Any] = {
    "company_id": 1.0,
    "title": "senior data engineer",
    "description": "lorem ipsum",
    "skills_desc": None,
    "normalized_salary": 100_000.0,
    "remote_allowed": 1.0,
    "sponsored": 0.0,
    "formatted_work_type": "FULL_TIME",
    "formatted_experience_level": "Mid-Senior level",
    "pay_period": "YEARLY",
    "application_type": "OffsiteApply",
}

# Age of the smoke posting, in days before "now".
_SMOKE_POSTING_AGE_DAYS: int = 7


def _smoke_posting(now_ms: float | None = None) -> dict[str, Any]:
    """A realistic, in-range posting to score at boot.

    Timestamps are computed at call time, never hardcoded. A frozen date here
    would drift out of the age range the model was trained on -- exactly the
    way ``bundle["snapshot_ms"]`` did at serving time. The boot check has to
    look like real traffic, and stay looking like it.
    """
    now = time.time() * 1000 if now_ms is None else now_ms
    listed = now - _SMOKE_POSTING_AGE_DAYS * MS_PER_DAY
    return {
        **_SMOKE_POSTING_FIELDS,
        "original_listed_time": listed,
        "listed_time": listed,
        "expiry": now + 23 * MS_PER_DAY,
    }


@dataclass
class Readiness:
    """Outcome of startup validation, served verbatim by ``/ready``."""

    ready: bool = False
    reason: str | None = None
    checks: dict[str, str] = field(default_factory=dict)
    bundle: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Public view. Excludes the bundle, which holds the fitted models."""
        return {"ready": self.ready, "reason": self.reason, "checks": self.checks}


def _minor(version: str) -> str:
    """'1.7.2' -> '1.7'. Patch releases do not break sklearn pickles."""
    return ".".join(version.split(".")[:2])


def validate_artifact(path: str) -> Readiness:
    """Load an artifact and check it is safe to serve from.

    Never raises. A failure is reported through the returned ``Readiness``.
    """
    state = Readiness()

    try:
        bundle = load_bundle(path)
    except Exception as exc:  # noqa: BLE001 - any load failure means not ready
        state.reason = f"artifact could not be loaded: {exc}"
        state.checks["artifact_loads"] = "fail"
        logger.error("Startup validation failed: %s", state.reason)
        return state

    state.checks["artifact_loads"] = "ok"

    # sklearn pickles are not guaranteed portable across minor versions. A
    # mismatch can change predictions without raising, so refuse to serve.
    trained_with = bundle.get("sklearn_version")
    running = sklearn.__version__
    if trained_with is None:
        # An artifact that cannot state its provenance is not safer than one
        # with the wrong provenance: both leave the serving code guessing.
        # Refusing keeps model rollback, backward compatibility and the
        # code/model separation verifiable rather than a matter of luck.
        state.reason = (
            f"artifact does not record sklearn_version (runtime has {running}); "
            "retrain with a version that stamps it"
        )
        state.checks["sklearn_version"] = "fail"
        logger.error("Startup validation failed: %s", state.reason)
        return state
    if _minor(trained_with) != _minor(running):
        state.reason = (
            f"sklearn version mismatch: artifact trained with {trained_with}, "
            f"runtime has {running}"
        )
        state.checks["sklearn_version"] = "fail"
        logger.error("Startup validation failed: %s", state.reason)
        return state
    state.checks["sklearn_version"] = f"ok ({trained_with})"

    # The serving code and the artifact must agree on the feature contract.
    if bundle.get("feature_cols") != FEATURE_COLS:
        state.reason = "artifact feature_cols do not match this build's FEATURE_COLS"
        state.checks["feature_contract"] = "fail"
        logger.error("Startup validation failed: %s", state.reason)
        return state
    state.checks["feature_contract"] = "ok"

    # Prove end to end that a request can actually be scored.
    try:
        preds = predict_views(pd.DataFrame([_smoke_posting()]), bundle, target="views")
        if len(preds) != 1 or not float(preds[0]) >= 0:
            raise ValueError(f"unusable smoke prediction: {preds!r}")
    except Exception as exc:  # noqa: BLE001
        state.reason = f"smoke prediction failed: {exc}"
        state.checks["smoke_prediction"] = "fail"
        logger.error("Startup validation failed: %s", state.reason)
        return state
    state.checks["smoke_prediction"] = "ok"

    state.ready = True
    state.bundle = bundle
    logger.info("Startup validation passed: %s", state.checks)
    return state
