"""Save the background sample used for reason codes (months 0-5 only).   Usage: python -m src.background"""
from pathlib import Path

import numpy as np

from src.data import SPLIT_COL, clean, load_raw
from src.model import model_features
from src.serve import load_model

ROOT = Path(__file__).resolve().parents[1]


def main(version: str = "v1", n: int = 100) -> None:
    model, _ = load_model(version)
    df = clean(load_raw(ROOT / "data" / "Base.csv"))
    train = df[df[SPLIT_COL] <= 5]
    Z = model["pre"].transform(model_features(train.sample(n, random_state=0)))
    np.save(ROOT / "models" / f"background_{version}.npy", np.asarray(Z, dtype=float))
    print(f"saved models/background_{version}.npy  shape={np.asarray(Z).shape}")


if __name__ == "__main__":
    main()