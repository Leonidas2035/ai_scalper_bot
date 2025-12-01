import os
from pathlib import Path

import xgboost as xgb

from .dataset_builder import DatasetBuilder, FEATURE_COLS, TARGET_COL


def train_model(symbol: str = "BTCUSDT", horizon: int = 1):
    print("[INFO] Building dataset...")

    builder = DatasetBuilder(symbol=symbol, horizon=horizon)
    X, y, df = builder.build_dataset()

    n = len(X)
    print(f"[INFO] Dataset size: {n} rows")

    if n < 500:
        print("[ERROR] Not enough data to train (need >= 500 rows).")
        return

    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    print("[INFO] Training model...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=150,
    )

    models_dir = Path("storage") / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"signal_xgb_{symbol}_h{horizon}.json"
    model.save_model(model_path)

    print(f"[OK] Saved model: {model_path}")

    # Save full dataset for debug
    out_ds = Path("storage") / "datasets"
    out_ds.mkdir(parents=True, exist_ok=True)
    builder.save_parquet(df, out_ds / f"{symbol}_h{horizon}.parquet")

    print("[DONE] Training complete.")


if __name__ == "__main__":
    train_model(symbol="BTCUSDT", horizon=1)
