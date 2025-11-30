import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

from .dataset_builder import build_dataset, load_dataset


MODELS_DIR = os.path.join("storage", "models")


def train_model(symbol: str, horizon: int = 3, test_size: float = 0.2) -> str:
    ds = build_dataset(symbol=symbol, horizon=horizon, min_rows=50)
    X, y = ds.X, ds.y

    if len(X) < 50:
        print(f"[TRAIN] Very small dataset: {len(X)} rows. Training anyway (demo mode).")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    try:
        auc = roc_auc_score(y_val, y_proba)
    except ValueError:
        auc = float("nan")

    print(f"[TRAIN] Validation accuracy: {acc:.4f}, AUC: {auc:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"signal_xgb_{symbol.upper()}_h{horizon}.json")
    clf.save_model(model_path)
    print(f"[TRAIN] Model saved to: {model_path}")

    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train signal model (XGBoost)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()

    train_model(symbol=args.symbol, horizon=args.horizon)


if __name__ == "__main__":
    main()
