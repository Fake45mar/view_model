"""sklearn pipeline construction (preprocessing + model)."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TEXT_FEATURE


def build_pipeline() -> Pipeline:
    """Returns an unfitted end-to-end Pipeline.

    The transformers are pure stateless config; ``fit`` learns everything.
    """
    title_pipeline = make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=20,
            max_features=20_000,
            stop_words="english",
        ),
        TruncatedSVD(n_components=50, random_state=RANDOM_STATE),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "ohe",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=20,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            ("title_tfidf", title_pipeline, TEXT_FEATURE),
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.05,
                    max_depth=8,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
