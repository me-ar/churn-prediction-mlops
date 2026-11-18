"""Data download, validation and cleaning."""
from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd

from . import config as C

EXPECTED_COLUMNS = {
    C.ID_COL, "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", C.TARGET,
}


def download(force: bool = False) -> None:
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if C.RAW_PATH.exists() and not force:
        return
    print(f"Downloading dataset -> {C.RAW_PATH}")
    urllib.request.urlretrieve(C.DATA_URL, C.RAW_PATH)


def validate(df: pd.DataFrame) -> None:
    """Fail fast on schema drift — cheap insurance before training."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[C.ID_COL].duplicated().any():
        raise ValueError("Duplicate customer IDs found")
    if not set(df[C.TARGET].unique()) <= {"Yes", "No"}:
        raise ValueError("Unexpected target values")
    if (df["tenure"] < 0).any() or (df["MonthlyCharges"] < 0).any():
        raise ValueError("Negative tenure or charges")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges is blank for brand-new customers (tenure == 0): they have paid nothing yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip()
                                       .replace("", np.nan), errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])
    return df


def load(path=None) -> tuple[pd.DataFrame, pd.Series]:
    path = path or C.RAW_PATH
    df = pd.read_csv(path)
    validate(df)
    df = clean(df)
    y = (df.pop(C.TARGET) == "Yes").astype(int)
    df = df.drop(columns=[C.ID_COL])
    return df, y
