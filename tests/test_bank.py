import pandas as pd

from churnlens.bank import prepare_bank


def test_prepare_bank_removes_duration_and_encodes_pdays():
    df = pd.DataFrame({"duration": [100, 200], "pdays": [-1, 30], "balance": [-5, 50]})
    out = prepare_bank(df)
    assert "duration" not in out.columns
    assert "pdays" not in out.columns
    assert out["previously_contacted"].tolist() == [0, 1]
    assert out["pdays_clean"].tolist() == [0, 30]
