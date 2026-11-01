"""Dataset yükləmə və təmizləmə."""
from pathlib import Path

import pandas as pd

# src/churnlens/data.py -> parents[2] = layihənin kökü
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

TELCO_NUM = ["tenure", "MonthlyCharges"]
TELCO_CAT = [
    "gender", "Contract", "PaymentMethod", "PaperlessBilling",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def load_telco(path=None):
    """Telco datasını yükləyir. TotalCharges boşluqları 0.0 ilə doldurulur."""
    df = pd.read_csv(path or DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].str.strip(), errors="coerce"
    ).fillna(0.0)
    y = (df["Churn"] == "Yes").astype(int)
    return df, y


def load_bank(path=None):
    """Bank Marketing datasını yükləyir. Hədəf 'y' feature-lardan ayrılır.

    'duration' sütunu burada çıxarılmır: leakage qərarı model səviyyəsində verilir.
    """
    df = pd.read_csv(path or DATA_DIR / "bank-full.csv", sep=";")
    y = (df["y"] == "yes").astype(int)
    return df.drop(columns=["y"]), y
