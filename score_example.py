"""Send a real application (months 0-5 only) to the running API and print the response.
Usage: python score_example.py high|medium|low
Start the server first, in another terminal:  uvicorn src.api:app --reload"""
import time
import json
import sys

import httpx

from src.data import clean, load_raw
from src.model import model_features
from src.serve import load_model
from tests.test_schema import to_record

want = sys.argv[1] if len(sys.argv) > 1 else "high"
raw = load_raw("data/Base.csv")
raw = raw[raw["month"] <= 5].sample(20000, random_state=11).copy()  # never the test months
model, _ = load_model("v1")
raw["score"] = model.predict_proba(model_features(clean(raw)))[:, 1]
cut = json.load(open("models/bands_v1.json"))["cutoffs"]

if want == "high":
    pool = raw[raw["score"] >= cut["high"]]
elif want == "medium":
    pool = raw[(raw["score"] >= cut["medium"]) & (raw["score"] < cut["high"])]
else:
    pool = raw[raw["score"] < cut["medium"]]

idx = pool.index[0]
print("actual label in the data (fraud_bool):", int(pool.loc[idx, "fraud_bool"]))
record = to_record(pool.drop(columns="score").loc[idx].to_dict())


t0 = time.perf_counter()
r = httpx.post("http://127.0.0.1:8000/score", json=record, timeout=60)
print(f"HTTP {r.status_code} in {time.perf_counter() - t0:.3f} s")
print(json.dumps(r.json(), indent=2))