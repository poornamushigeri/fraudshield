import numpy as np
import pandas as pd

from src.drift import feature_psi


def test_identical_distributions_have_near_zero_psi():
    x = pd.Series(np.random.default_rng(0).normal(size=20000))
    assert feature_psi(x, x.sample(frac=1, random_state=1)) < 0.001


def test_a_clearly_shifted_feature_has_large_psi():
    rng = np.random.default_rng(1)
    assert feature_psi(pd.Series(rng.normal(0, 1, 20000)), pd.Series(rng.normal(2, 1, 20000))) > 0.25


def test_missing_values_count_as_their_own_bin():
    rng = np.random.default_rng(2)
    ref = pd.Series(rng.normal(size=20000))
    cur = ref.copy()
    cur[:10000] = np.nan                     # half the values suddenly missing
    assert feature_psi(ref, cur) > 0.25


def test_new_category_in_a_discrete_feature_is_detected():
    ref = pd.Series(["A", "B"] * 5000)
    cur = pd.Series(["A", "B", "C"] * 3000)
    assert feature_psi(ref, cur) > 0.1