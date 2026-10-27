"""Check that the running API (for example the container) reproduces the offline model's scores.
Usage: python check_container.py   (start the API first)"""
import httpx
import numpy as np

from src.data import clean, load_raw
from src.model import model_features
from src.serve import load_model
from tests.test_schema import to_record

raw = load_raw("data/Base.csv")
sample = raw[raw["month"] <= 5].sample(50, random_state=21)  # never the test months
model, _ = load_model("v1")
offline = model.predict_proba(model_features(clean(sample)))[:, 1]

served = []
for rec in sample.to_dict("records"):
    r = httpx.post("http://127.0.0.1:8000/score", json=to_record(rec), timeout=60)
    r.raise_for_status()
    served.append(r.json()["risk_score"])

diff = np.abs(np.array(served) - offline).max()
print(f"max absolute difference over {len(sample)} applications: {diff:.7f}")
print("PASS" if diff < 1e-5 else "FAIL: the service does not reproduce the offline model")