"""sql/churn_queries.sql sorğularını DuckDB ilə işlədir.

İşlətmək: python sql/run_queries.py  (əvvəl train.py işlədilməlidir)
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SCORED = ROOT / "data" / "processed" / "telco_scored.csv"

con = duckdb.connect()
con.execute(f"CREATE VIEW telco AS SELECT * FROM read_csv_auto('{RAW}')")
con.execute(f"CREATE VIEW scored AS SELECT * FROM read_csv_auto('{SCORED}')")

text = (ROOT / "sql" / "churn_queries.sql").read_text(encoding="utf-8")
chunks = [c.strip() for c in text.split(";")]
queries = [c for c in chunks if "SELECT" in c.upper()]

for i, q in enumerate(queries, 1):
    print(f"\n=== Q{i} ===")
    con.sql(q).show()
