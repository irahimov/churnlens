"""ChurnLens metrikaları: top-k precision/recall/lift və PSI."""
import numpy as np


def top_k_metrics(y_true, y_proba, fraction=0.10):
    """Ehtimala görə azalan sıralanmış ilk `fraction` hissəsinin metrikaları."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    order = np.argsort(-y_proba, kind="stable")
    k = int(fraction * len(y_true))
    hits = int(y_true[order[:k]].sum())
    precision = hits / k
    return {
        "k": k,
        "hits": hits,
        "precision": precision,
        "recall": hits / y_true.sum(),
        "lift": precision / y_true.mean(),
    }


def psi(expected, actual, bins=10):
    """Population Stability Index. Bin sərhədləri yalnız `expected`-dən götürülür."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def value_weighted_lift(y_true, y_proba, value, fraction=0.10):
    """Top-k-nın tutduğu churn/conversion dəyərinin faizi, sadə lift ilə müqayisə.

    value: hər müştərinin dəyəri (Telco: MonthlyCharges*12, Bank: max(balance,0)).
    Qaytarır: top-k-da tutulan dəyərin payı, bütün churn/conversion dəyərinə nisbətən,
    və həmin pay ÷ top-k-nın dəyər payı (lift-in dəyər versiyası).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    value = np.asarray(value, dtype=float)

    order = np.argsort(-y_proba, kind="stable")
    k = int(fraction * len(y_true))
    top = order[:k]

    captured_value = value[top][y_true[top] == 1].sum()
    total_positive_value = value[y_true == 1].sum()
    value_share_captured = captured_value / total_positive_value

    # Təsadüfi seçimdə gözlənilən pay: top-k-nın dəyər payı
    random_share = fraction

    return {
        "k": k,
        "captured_value": float(captured_value),
        "value_share_captured": float(value_share_captured),
        "value_weighted_lift": float(value_share_captured / random_share),
    }
