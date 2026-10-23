from pathlib import Path

from src.data import load_raw, clean, split_by_month

RAW = Path("data/Base.csv")
OUT = Path("data/clean.pkl")


def main():
    df = clean(load_raw(RAW))
    train, valid, test = split_by_month(df)
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        print(f"{name}: {part.shape}, fraud rate {part['fraud_bool'].mean():.4%}")
    df.to_pickle(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()