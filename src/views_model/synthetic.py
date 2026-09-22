"""Deterministic synthetic postings, and a demo artifact built from them.

Two consumers:

* the test suite, so CI never downloads the 500 MB Kaggle dataset;
* ``python -m views_model.synthetic``, which trains a small artifact so the
  container can be built and smoke-tested without any real data.

The resulting model is meaningless -- it is trained on invented rows. It
exists to prove the pipeline and the serving path, not to predict anything.

Everything here is deterministic: a fixed seed and fixed epoch-ms constants,
never a wall-clock read, so CI gets identical input on every run.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import tempfile

import numpy as np
import pandas as pd

from .config import DEFAULT_ARTIFACT_PATH, MS_PER_DAY

logger = logging.getLogger("views_model.synthetic")

# Fixed reference point (2024-01-01T00:00:00Z) so ages never depend on "now".
BASE_MS: int = 1_704_067_200_000

# Vocabulary pieces. The cross product gives enough distinct TF-IDF terms to
# stay above `n_components`, while each word repeats often enough to survive
# `min_df`.
_SENIORITY = ["senior", "junior", "staff", "lead"]
_DOMAIN = ["data", "software", "platform", "machine learning"]
_ROLE = ["engineer", "scientist", "analyst"]

_WORK_TYPE = ["FULL_TIME", "PART_TIME", "CONTRACT"]
_EXPERIENCE = ["Entry level", "Mid-Senior level", "Director"]
_PAY_PERIOD = ["YEARLY", "MONTHLY", "HOURLY"]
_APPLICATION_TYPE = ["OffsiteApply", "ComplexOnsiteApply"]
_INDUSTRY = ["IT Services", "Financial Services", "Staffing", "Hospitals"]

# Pipeline hyperparameters sized for synthetic data. The production values in
# config.PIPELINE_PARAMS assume a full dataset and cannot fit anything this
# small -- TruncatedSVD alone needs more features than a tiny vocabulary has.
SMALL_PIPELINE_PARAMS: dict = {
    "min_df": 2,
    "n_components": 5,
    "min_frequency": 2,
    "max_iter": 20,
}


def make_postings(
    n_rows: int = 200,
    seed: int = 0,
    first_listed_ms: int = BASE_MS,
    span_days: int = 200,
    n_companies: int = 12,
) -> pd.DataFrame:
    """Build a synthetic postings frame with the columns the model reads.

    Rows are ordered by ``original_listed_time`` so a time-based split has
    something meaningful to split on.
    """
    rng = np.random.default_rng(seed)

    # Evenly spaced listing times, oldest first.
    offsets = np.linspace(0, span_days, n_rows) * MS_PER_DAY
    original_listed_time = first_listed_ms + offsets

    # ~20% of postings are relisted a few days after first appearing.
    relisted = rng.random(n_rows) < 0.2
    listed_time = original_listed_time + relisted * rng.integers(1, 10, n_rows) * MS_PER_DAY

    # Every posting expires after it was listed.
    expiry = original_listed_time + rng.integers(14, 60, n_rows) * MS_PER_DAY

    titles = [
        f"{rng.choice(_SENIORITY)} {rng.choice(_DOMAIN)} {rng.choice(_ROLE)}"
        for _ in range(n_rows)
    ]
    descriptions = ["lorem ipsum " * rng.integers(5, 60) for _ in range(n_rows)]

    salary = rng.normal(100_000, 25_000, n_rows).round(2)
    salary[rng.random(n_rows) < 0.15] = np.nan  # exercise the median imputer

    skills_desc = np.where(rng.random(n_rows) < 0.4, "python, sql", None)
    remote_allowed = rng.choice([0.0, 1.0, np.nan], n_rows, p=[0.5, 0.3, 0.2])
    sponsored = rng.choice([0.0, 1.0], n_rows, p=[0.7, 0.3])

    frame = pd.DataFrame(
        {
            "company_id": rng.integers(1, n_companies + 1, n_rows).astype(float),
            "title": titles,
            "description": descriptions,
            "skills_desc": skills_desc,
            "normalized_salary": salary,
            "remote_allowed": remote_allowed,
            "sponsored": sponsored,
            "formatted_work_type": rng.choice(_WORK_TYPE, n_rows),
            "formatted_experience_level": rng.choice(_EXPERIENCE, n_rows),
            "pay_period": rng.choice(_PAY_PERIOD, n_rows),
            "application_type": rng.choice(_APPLICATION_TYPE, n_rows),
            "original_listed_time": original_listed_time,
            "listed_time": listed_time,
            "expiry": expiry,
        }
    )

    # Target with mild signal so the fit is not pure noise. Older postings have
    # accumulated more views, which is what the real target looks like.
    age_days = (original_listed_time.max() - original_listed_time) / MS_PER_DAY
    frame["views"] = (
        10
        + 0.5 * age_days
        + 20 * frame["sponsored"]
        + rng.normal(0, 5, n_rows)
    ).clip(lower=0).round()

    return frame


def make_company_industries(n_companies: int = 12, seed: int = 0) -> pd.DataFrame:
    """Build the ``company_id -> industry`` lookup that features.py merges on."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "company_id": np.arange(1, n_companies + 1, dtype=float),
            "industry": rng.choice(_INDUSTRY, n_companies),
        }
    )


def write_dataset(directory: pathlib.Path, **kwargs) -> tuple[str, str]:
    """Write both CSVs to ``directory`` and return their paths.

    ``train()`` calls ``pd.read_csv``, so callers have to hit the disk rather
    than hand over a DataFrame.
    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    postings_path = directory / "postings.csv"
    companies_path = directory / "company_industries.csv"

    make_postings(**kwargs).to_csv(postings_path, index=False)
    make_company_industries().to_csv(companies_path, index=False)
    return str(postings_path), str(companies_path)


def build_demo_artifact(out_path: str = DEFAULT_ARTIFACT_PATH, n_rows: int = 500) -> str:
    """Train a throwaway model on synthetic rows and save it to ``out_path``."""
    from .train import train  # imported here to keep module import cheap

    with tempfile.TemporaryDirectory() as tmp:
        postings_path, companies_path = write_dataset(pathlib.Path(tmp), n_rows=n_rows)
        train(
            postings_path=postings_path,
            companies_path=companies_path,
            out_path=out_path,
            pipeline_params=SMALL_PIPELINE_PARAMS,
        )

    logger.warning(
        "Demo artifact at %s is trained on synthetic rows. Its predictions are "
        "meaningless; it exists to exercise the serving path.",
        out_path,
    )
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=build_demo_artifact.__doc__)
    p.add_argument("--out", default=DEFAULT_ARTIFACT_PATH)
    p.add_argument("--rows", type=int, default=500)
    args = p.parse_args()
    build_demo_artifact(args.out, args.rows)


if __name__ == "__main__":
    main()
