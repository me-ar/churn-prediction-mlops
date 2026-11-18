"""Feature engineering as sklearn transformers, so the exact same logic runs in
training and in the API (no train/serve skew)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config as C


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Domain features derived from raw columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)
        X["SeniorCitizen"] = X["SeniorCitizen"].replace({0: "No", 1: "Yes"}).astype(str)
        X["avg_monthly_spend"] = np.where(X["tenure"] > 0, X["TotalCharges"] / X["tenure"].clip(lower=1),
                                          X["MonthlyCharges"])
        X["n_services"] = sum((X[c] == "Yes").astype(int) for c in C.SERVICE_COLS)
        X["tenure_band"] = pd.cut(X["tenure"], bins=[-1, 6, 12, 24, 48, 1000],
                                  labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"]).astype(str)
        return X


def build_preprocessor(scale_numeric: bool = True) -> Pipeline:
    numeric = StandardScaler() if scale_numeric else "passthrough"
    ct = ColumnTransformer(
        [
            ("num", numeric, C.NUMERIC_FEATURES),
            ("bin", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             C.BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             C.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", FeatureBuilder()), ("encode", ct)])
