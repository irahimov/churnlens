"""Pipeline qurulması (Faza 4-dəki pipeline_B ilə eyni)."""
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(num_features, cat_features):
    """Rəqəm sütunları StandardScaler, kateqorial sütunlar OneHotEncoder.

    Scaler və encoder yalnız fit zamanı train datasında öyrənir (sızmanın qarşısı).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_features),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
