"""Request schema for the scoring API.

Limits are the widest of (a) the ranges in the BAF datasheet and (b) the ranges observed in
months 0-5 (the model's training data). The test months were not consulted.
Unknown values are sent as null and mapped back to the dataset's own -1 code.
"""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.data import MISSING_CODE_COLS, NEGATIVE_MISSING_COLS

Flag = Literal[0, 1]
UNKNOWN = -1  # the dataset's code for "value unknown"


class Application(BaseModel):
    # customer_age, days_since_request, month and the label are not model inputs, so they are rejected
    model_config = ConfigDict(extra="forbid")

    income: float = Field(..., ge=0.1, le=0.9, description="Income decile")
    name_email_similarity: float = Field(..., ge=0, le=1)
    prev_address_months_count: int | None = Field(..., ge=0, le=380, description="null = unknown")
    current_address_months_count: int | None = Field(..., ge=0, le=429, description="null = unknown")
    intended_balcon_amount: float | None = Field(..., ge=0, le=114, description="null = unknown")
    payment_type: Literal["AA", "AB", "AC", "AD", "AE"]
    zip_count_4w: int = Field(..., ge=1, le=6830)
    velocity_6h: float = Field(..., ge=-175, le=16818)
    velocity_24h: float = Field(..., ge=1297, le=9586)
    velocity_4w: float = Field(..., ge=2825, le=7020)
    bank_branch_count_8w: int = Field(..., ge=0, le=2404)
    date_of_birth_distinct_emails_4w: int = Field(..., ge=0, le=39)
    employment_status: Literal["CA", "CB", "CC", "CD", "CE", "CF", "CG"]
    credit_risk_score: int = Field(..., ge=-191, le=389)
    email_is_free: Flag
    housing_status: Literal["BA", "BB", "BC", "BD", "BE", "BF", "BG"]
    phone_home_valid: Flag
    phone_mobile_valid: Flag
    bank_months_count: int | None = Field(..., ge=0, le=32, description="null = unknown")
    has_other_cards: Flag
    proposed_credit_limit: float = Field(..., ge=190, le=2100)
    foreign_request: Flag
    source: Literal["INTERNET", "TELEAPP"]
    session_length_in_minutes: float | None = Field(..., ge=0, le=107, description="null = unknown")
    device_os: Literal["linux", "macintosh", "other", "windows", "x11"]
    keep_alive_session: Flag
    device_distinct_emails_8w: int | None = Field(..., ge=0, le=2, description="null = unknown")


def to_frame(apps: list[Application]) -> pd.DataFrame:
    """Applications -> DataFrame in the dataset's raw format (null turned back into the -1 code),
    so the same clean() used in training handles them."""
    rows = []
    for app in apps:
        row = app.model_dump()
        for c in MISSING_CODE_COLS + NEGATIVE_MISSING_COLS:
            if row[c] is None:
                row[c] = UNKNOWN
        rows.append(row)
    return pd.DataFrame(rows)