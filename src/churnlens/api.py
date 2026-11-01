"""Minimal scoring API: GET /health, POST /score.

İşlətmək: PYTHONPATH=src uvicorn churnlens.api:app --reload
Əvvəl train.py işlədilməlidir (models/churn.joblib lazımdır).
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "churn.joblib"

YesNo = Literal["Yes", "No"]
Service = Literal["Yes", "No", "No internet service"]


class CustomerIn(BaseModel):
    tenure: int = Field(ge=0, le=120)
    MonthlyCharges: float = Field(ge=0)
    gender: Literal["Female", "Male"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    PaperlessBilling: YesNo
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Service
    OnlineBackup: Service
    DeviceProtection: Service
    TechSupport: Service
    StreamingTV: Service
    StreamingMovies: Service


class ScoreOut(BaseModel):
    churn_probability: float
    decile: int = Field(description="1..10, 10 = ən yüksək risk")


@asynccontextmanager
async def lifespan(app):
    path = Path(os.environ.get("CHURN_MODEL_PATH", DEFAULT_MODEL))
    app.state.bundle = joblib.load(path) if path.exists() else None
    yield


app = FastAPI(title="ChurnLens", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": getattr(app.state, "bundle", None) is not None}


@app.post("/score", response_model=ScoreOut)
def score(customer: CustomerIn):
    bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model yüklənməyib: əvvəl train.py işlədin")
    row = pd.DataFrame([customer.model_dump()])
    X = row[bundle["num_features"] + bundle["cat_features"]]
    p = float(bundle["pipeline"].predict_proba(X)[0, 1])
    d = int(1 + np.searchsorted(bundle["decile_edges"], p, side="right"))
    return ScoreOut(churn_probability=p, decile=d)
