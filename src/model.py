import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.data import CATEGORICAL_COLS, feature_frame

# Chosen on validation (see reports/model_selection.md); not tuned further.
HGB_PARAMS = dict(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=200,
    l2_regularization=1.0,
    random_state=0,
)


def model_features(df):
    """Model inputs: no label, month, days_since_request or customer_age."""
    return feature_frame(df).drop(columns=["customer_age"])


def make_hgb(X):
    cat = [c for c in CATEGORICAL_COLS if c in X.columns]
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cat)],
        remainder="passthrough",
    )
    clf = HistGradientBoostingClassifier(categorical_features=list(range(len(cat))), **HGB_PARAMS)
    return Pipeline([("pre", pre), ("clf", clf)])