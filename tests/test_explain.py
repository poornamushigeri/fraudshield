from pathlib import Path

import numpy as np
import pytest

from src.data import clean, load_raw
from src.explain import Explainer
from src.model import model_features
from src.serve import load_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Base.csv"
MODEL = ROOT / "models" / "hgb_v1.joblib"
BACKGROUND = ROOT / "models" / "background_v1.npy"
pytestmark = pytest.mark.skipif(not (DATA.exists() and MODEL.exists() and BACKGROUND.exists()),
                                reason="data, model or background sample not available")


@pytest.fixture(scope="module")
def setup():
    model, _ = load_model("v1")
    df = clean(load_raw(DATA))
    X = model_features(df[df["month"] <= 5].sample(40, random_state=4))  # never the test months
    return Explainer.load(model, "v1"), X


def test_contributions_add_up_to_the_model_score(setup):
    explainer, X = setup
    contrib, base, raw_score = explainer.contributions(X)
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy() + base, raw_score, atol=1e-3)


def test_same_application_gets_the_same_reasons(setup):
    explainer, X = setup
    one = X.head(1)
    assert explainer.top_reasons(one) == explainer.top_reasons(one)