import numpy as np
import pandas as pd

LABEL = "fraud_bool"
SPLIT_COL = "month"

# -1 is the "value unknown" code (confirmed in the BAF datasheet)
MISSING_CODE_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
# Any negative value means "unknown" in this column (datasheet)
NEGATIVE_MISSING_COLS = ["intended_balcon_amount"]

DROP_COLS = ["device_fraud_count"]  # constant column, carries no information
CATEGORICAL_COLS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn missing-value codes into NaN, keep a was-missing flag, drop constant columns."""
    df = df.copy()
    for c in MISSING_CODE_COLS:
        df[f"{c}_missing"] = (df[c] == -1).astype("int8")
        df[c] = df[c].replace(-1, np.nan)
    for c in NEGATIVE_MISSING_COLS:
        df[f"{c}_missing"] = (df[c] < 0).astype("int8")
        df.loc[df[c] < 0, c] = np.nan
    return df.drop(columns=DROP_COLS)


def split_by_month(df: pd.DataFrame):
    """Time-based split. Months 0-4 train, 5 validation, 6-7 test.
    The test set must be evaluated only once, at the very end."""
    train = df[df[SPLIT_COL] <= 4]
    valid = df[df[SPLIT_COL] == 5]
    test = df[df[SPLIT_COL] >= 6]
    return train, valid, test