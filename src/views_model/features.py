"""Raw postings -> model-ready feature frame."""
from __future__ import annotations

import pandas as pd

from .config import MS_PER_DAY


def load_company_industry(path: str) -> pd.DataFrame:
    """Load and deduplicate the company -> industry lookup."""
    ci = pd.read_csv(path)
    return ci.drop_duplicates(subset="company_id", keep="first")


def build_features(
    raw: pd.DataFrame,
    company_industry: pd.DataFrame,
    snapshot_ms: float,
    ms_per_day: int = MS_PER_DAY,
) -> pd.DataFrame:
    """Apply training-time feature engineering to raw postings.

    Parameters
    ----------
    raw : DataFrame with the schema of ``postings.csv``.
    company_industry : 1:1 ``company_id -> industry`` lookup.
    snapshot_ms : reference "now" in epoch ms; controls ``age_days``.
        At training this is the newest listing in the data. At serving it must
        be the time of the request -- never the value frozen into the bundle,
        or every posting listed after training gets a negative age.
    """
    out = raw.merge(company_industry, on="company_id", how="left").copy()

    out["description_len"] = out.get("description", pd.Series("", index=out.index)).fillna("").str.len()
    out["title_len"] = out.get("title", pd.Series("", index=out.index)).fillna("").str.len()
    out["has_skills_desc"] = (
        out["skills_desc"].notna().astype(int) if "skills_desc" in out.columns else 0
    )
    out["has_remote"] = (
        out["remote_allowed"].fillna(0).astype(int) if "remote_allowed" in out.columns else 0
    )
    out["is_sponsored"] = (
        out["sponsored"].fillna(0).astype(int) if "sponsored" in out.columns else 0
    )
    out["title"] = out["title"].fillna("") if "title" in out.columns else ""

    out["planned_duration_days"] = (out["expiry"] - out["original_listed_time"]) / ms_per_day
    # Clipped at zero: at training time the reference is the newest listing so
    # ages are non-negative by construction, but at serving time a posting can
    # carry a timestamp ahead of the reference. The model never saw negative
    # ages, so feeding it one is out-of-distribution garbage.
    out["age_days"] = ((snapshot_ms - out["original_listed_time"]) / ms_per_day).clip(lower=0)
    out["days_since_relisted"] = ((snapshot_ms - out["listed_time"]) / ms_per_day).clip(lower=0)
    out["was_relisted"] = (out["listed_time"] != out["original_listed_time"]).astype(int)
    return out
