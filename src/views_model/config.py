"""Shared constants and feature configuration."""
from __future__ import annotations

MS_PER_DAY: int = 1000 * 60 * 60 * 24
AGE_FLOOR_DAYS: float = 1.0
RANDOM_STATE: int = 42

NUMERIC_FEATURES: list[str] = [
    "normalized_salary",
    "description_len",
    "title_len",
    "has_skills_desc",
    "has_remote",
    "is_sponsored",
    "planned_duration_days",
    "age_days",
    "days_since_relisted",
    "was_relisted",
]
CATEGORICAL_FEATURES: list[str] = [
    "formatted_work_type",
    "formatted_experience_level",
    "pay_period",
    "application_type",
    "industry",
]
TEXT_FEATURE: str = "title"
FEATURE_COLS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]

DEFAULT_ARTIFACT_PATH: str = "artifacts/views_baseline.joblib"
SCHEMA_VERSION: int = 1
