"""ChurnLens dashboard: gains curve, müqavilə üzrə churn, top-k siyahısı.

İşlətmək: streamlit run app/dashboard.py  (əvvəl train.py işlədilməlidir)
"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SCORED = ROOT / "data" / "processed" / "telco_scored.csv"

st.set_page_config(page_title="ChurnLens", layout="wide")
st.title("ChurnLens: zəng siyahısı")

if not SCORED.exists():
    st.error("data/processed/telco_scored.csv yoxdur. Əvvəl train.py işlədin.")
    st.stop()

df = pd.read_csv(SCORED).sort_values("oof_proba", ascending=False).reset_index(drop=True)
total = int(df["y"].sum())
df["pct_called"] = (df.index + 1) / len(df) * 100
df["cum_recall"] = df["y"].cumsum() / total

pct = st.slider("Zəng edilən müştəri faizi", 5, 50, 10, step=5)
top = df.head(int(len(df) * pct / 100))

c1, c2, c3 = st.columns(3)
c1.metric("Zəng sayı", f"{len(top):,}")
c2.metric("Tutulan churn", f"{int(top['y'].sum())} / {total}")
c3.metric("Precision", f"{top['y'].mean():.1%}")

st.subheader("Cumulative gains")
st.line_chart(df.set_index("pct_called")["cum_recall"])

st.subheader("Müqavilə üzrə churn faizi (bütün data)")
st.bar_chart(df.groupby("Contract")["y"].mean())

st.subheader("Top siyahı (ən riskli 200)")
st.dataframe(top[["customerID", "Contract", "MonthlyCharges", "oof_proba", "y"]].head(200))
