"""Shared fixtures.

The synthetic data generators live in ``views_model.synthetic`` rather than
here, because CI also needs them to build a demo artifact outside the test
run. They are re-exported so test modules can import them from one place.
"""
from __future__ import annotations

import pytest

from views_model.synthetic import (  # noqa: F401 - re-exported for test modules
    BASE_MS,
    SMALL_PIPELINE_PARAMS,
    make_company_industries,
    make_postings,
    write_dataset,
)
from views_model.train import train


@pytest.fixture(scope="session")
def pipeline_params() -> dict:
    return dict(SMALL_PIPELINE_PARAMS)


@pytest.fixture(scope="session")
def dataset(tmp_path_factory) -> tuple[str, str]:
    """Paths to a synthetic postings CSV and company-industry CSV."""
    return write_dataset(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def artifact_path(tmp_path_factory, dataset, pipeline_params) -> str:
    """Train once per session and return the bundle path.

    Session-scoped because training is the slowest thing in the suite. Each
    bundle gets a unique path: ``load_bundle`` is ``lru_cache``d on the path,
    so reusing one would hand back a stale object.
    """
    postings_path, companies_path = dataset
    out_path = tmp_path_factory.mktemp("artifacts") / "bundle.joblib"
    train(
        postings_path=postings_path,
        companies_path=companies_path,
        out_path=str(out_path),
        pipeline_params=pipeline_params,
    )
    return str(out_path)
