"""Telco churn modelini öyrədir, saxlayır və OOF nəticələrini çap edir.

İşlətmək: PYTHONPATH=src python -m churnlens.train
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from churnlens.data import TELCO_CAT, TELCO_NUM, load_telco
from churnlens.metrics import top_k_metrics, value_weighted_lift
from churnlens.model import build_pipeline

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "churn.joblib"
DEFAULT_SCORED = ROOT / "data" / "processed" / "telco_scored.csv"

# Fərziyyələr. Nəticəyə görə tune edilməyib, README-də də belə yazılır.
SAVE_RATE = 0.30   # zəng edilən churn-çülərin neçə faizi qalır
CALL_COST = 10.0   # AZN, bir zəngin xərci


def decile_edges(oof_proba):
    """OOF ehtimallarından 9 sərhəd: decile 1..10 üçün."""
    return np.quantile(oof_proba, np.linspace(0.1, 0.9, 9))


def to_decile(proba, edges):
    """1..10. 10 = ən yüksək risk."""
    return 1 + np.searchsorted(edges, proba, side="right")


def train(out_path=DEFAULT_MODEL, scored_path=DEFAULT_SCORED):
    df, y = load_telco()
    X = df[TELCO_NUM + TELCO_CAT]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    res = cross_validate(build_pipeline(TELCO_NUM, TELCO_CAT), X, y, cv=cv,
                         scoring=["roc_auc", "average_precision"])
    oof = cross_val_predict(build_pipeline(TELCO_NUM, TELCO_CAT), X, y, cv=cv,
                            method="predict_proba")[:, 1]
    edges = decile_edges(oof)

    final = build_pipeline(TELCO_NUM, TELCO_CAT).fit(X, y)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "num_features": TELCO_NUM,
                 "cat_features": TELCO_CAT, "decile_edges": edges}, out_path)

    value = (df["MonthlyCharges"] * 12).to_numpy()
    scored_path = Path(scored_path)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "customerID": df["customerID"],
        "y": y.to_numpy(),
        "Contract": df["Contract"],
        "MonthlyCharges": df["MonthlyCharges"],
        "oof_proba": oof,
        "decile": to_decile(oof, edges),
    }).to_csv(scored_path, index=False)

    print(f"ROC-AUC : {res['test_roc_auc'].mean():.4f} ± {res['test_roc_auc'].std():.4f}")
    print(f"AP      : {res['test_average_precision'].mean():.4f} ± {res['test_average_precision'].std():.4f}")
    print(f"Model   : {out_path}")
    print(f"Scored  : {scored_path}\n")

    print(f"{'top':>5} {'k':>6} {'precision':>10} {'recall':>8} {'lift':>6} {'val_lift':>9} {'EV (AZN)':>12}")
    order = np.argsort(-oof, kind="stable")
    for frac in (0.05, 0.10, 0.20):
        m = top_k_metrics(y, oof, fraction=frac)
        v = value_weighted_lift(y, oof, value, fraction=frac)
        idx = order[: m["k"]]
        ev = float(np.sum(oof[idx] * value[idx] * SAVE_RATE - CALL_COST))
        print(f"{frac:>5.0%} {m['k']:>6} {m['precision']:>10.4f} {m['recall']:>8.4f} "
              f"{m['lift']:>6.2f} {v['value_weighted_lift']:>9.2f} {ev:>12,.0f}")


if __name__ == "__main__":
    train()
