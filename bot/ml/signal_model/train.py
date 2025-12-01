import argparse
from pathlib import Path

import xgboost as xgb

from bot.ml.signal_model.dataset_builder import DatasetBuilder


def train_model(symbol: str = "BTCUSDT", horizon: int = 1, min_rows: int = 1000):
    root = Path(__file__).resolve().parents[3]
    model_dir = root / "storage" / "models"
    dataset_dir = root / "storage" / "datasets"
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Building dataset for {symbol}, horizon={horizon} ...")
    builder = DatasetBuilder(symbol=symbol, horizon=horizon)
    X, y, df = builder.build()

    if X.empty or len(X) < min_rows:
        print(f"[ERROR] Not enough training data ({len(X)} rows). Need at least {min_rows}.")
        return

    if y.nunique() < 2:
        print("[ERROR] Target contains a single class. Need both up/down examples to train.")
        return

    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
    }

    model = xgb.XGBClassifier(**params)
    print("[INFO] Training XGBoost model ...")
    model.fit(X, y)

    model_path = model_dir / f"signal_xgb_{symbol}_h{horizon}.json"
    model.save_model(model_path)
    print(f"[OK] Model saved to {model_path}")

    dataset_path = dataset_dir / f"{symbol}_h{horizon}.parquet"
    try:
        df.to_parquet(dataset_path, index=False)
        print(f"[OK] Dataset saved to {dataset_path}")
    except Exception as exc:
        fallback_path = dataset_dir / f"{symbol}_h{horizon}.csv"
        df.to_csv(fallback_path, index=False)
        print(f"[WARN] Could not save parquet ({exc}). Saved CSV instead: {fallback_path}")
    print("[DONE] Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train signal model offline.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to train on (e.g., BTCUSDT)")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in ticks")
    parser.add_argument("--min-rows", type=int, default=1000, help="Minimum rows required to train")
    return parser.parse_args()


def main():
    args = parse_args()
    train_model(symbol=args.symbol, horizon=args.horizon, min_rows=args.min_rows)


if __name__ == "__main__":
    main()
