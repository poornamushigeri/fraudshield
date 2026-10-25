import json
import sys

from src.data import clean, load_raw
from src.model import model_features
from src.serve import load_model
from tests.test_schema import to_record

want = sys.argv[1] if len(sys.argv) > 1 else "any"
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
    pool = raw

row = pool.drop(columns="score").to_dict("records")[0]
print(json.dumps(to_record(row), indent=2))