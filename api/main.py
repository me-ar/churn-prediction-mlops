"""FastAPI service for real-time churn scoring.

Run:  uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")
STATE: dict = {}

YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)",
                           "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 171.0}}}


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_band: Literal["low", "medium", "high"]
    recommend_retention_offer: bool


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["bundle"] = joblib.load(MODEL_PATH)
    yield
    STATE.clear()


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


def _score(customers: list[Customer]) -> list[Prediction]:
    bundle = STATE["bundle"]
    df = pd.DataFrame([c.model_dump() for c in customers])[bundle["features"]]
    proba = bundle["model"].predict_proba(df)[:, 1]
    thr = bundle["threshold"]
    out = []
    for p in proba:
        band = "high" if p >= 0.5 else "medium" if p >= thr else "low"
        out.append(Prediction(churn_probability=round(float(p), 4), will_churn=bool(p >= 0.5),
                              risk_band=band, recommend_retention_offer=bool(p >= thr)))
    return out


@app.get("/health")
def health():
    bundle = STATE.get("bundle")
    if bundle is None:
        raise HTTPException(503, "Model not loaded")
    return {"status": "ok", "model": bundle["model_name"], "trained_at": bundle["trained_at"],
            "threshold": bundle["threshold"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return _score([customer])[0]


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    t0 = time.perf_counter()
    preds = _score(req.customers)
    return {"predictions": preds, "n": len(preds),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
