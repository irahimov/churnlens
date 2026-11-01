"""Bank Marketing propensity: out-of-fold ehtimallar, top-k və dəyər cədvəli.

Dəyər proxy-si: max(balance, 0). Mənfi balans (overdraft) 0 sayılır.
Bu illik gəlir deyil, anlıq balans vəziyyətidir.

İşlətmək: PYTHONPATH=src python -m churnlens.bank
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from churnlens.data import load_bank
from churnlens.metrics import top_k_metrics, value_weighted_lift
from churnlens.model import build_pipeline

BANK_NUM = ["age", "balance", "day", "campaign", "pdays_clean", "previous"]
BANK_CAT = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome", "previously_contacted",
]

# Telco ilə eyni fərziyyələr. Nəticəyə görə tune edilməyib.
# Propensity-də SAVE_RATE "zəngdən sonra əmanət açma" ehtimalının fərziyyəsidir.
# EV_AZN bu modulda balance proksi üzrə hesablanır və GƏLİR DEYİL:
# bankın real gəliri marjadır, qalığın özü deyil. Daxili yoxlama üçündür,
# README-də mütləq AZN kimi göstərilmir.
SAVE_RATE = 0.30
CALL_COST = 10.0


def prepare_bank(df):
    """Leakage və pdays kodlaşdırması. duration çıxarılır, pdays=-1 ayrıca işarələnir."""
    df = df.copy()
    df["previously_contacted"] = (df["pdays"] != -1).astype(int)
    df["pdays_clean"] = df["pdays"].replace(-1, 0)
    return df.drop(columns=["duration", "pdays"])


def bank_value_table(fractions=(0.05, 0.10, 0.20)):
    raw, y = load_bank()
    X = prepare_bank(raw)
    value = X["balance"].clip(lower=0).to_numpy(dtype=float)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(
        build_pipeline(BANK_NUM, BANK_CAT),
        X[BANK_NUM + BANK_CAT], y, cv=cv, method="predict_proba",
    )[:, 1]
    order = np.argsort(-oof, kind="stable")

    rows = []
    for frac in fractions:
        m = top_k_metrics(y, oof, fraction=frac)
        v = value_weighted_lift(y, oof, value, fraction=frac)
        idx = order[: m["k"]]
        ev = float(np.sum(oof[idx] * value[idx] * SAVE_RATE - CALL_COST))
        rows.append({
            "top": f"{frac:.0%}",
            "k": m["k"],
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "lift": round(m["lift"], 2),
            "val_lift": round(v["value_weighted_lift"], 2),
            "EV_AZN": round(ev),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(bank_value_table().to_string(index=False))
