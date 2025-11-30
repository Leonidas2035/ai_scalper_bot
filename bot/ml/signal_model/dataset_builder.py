import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


OFFLINE_DIR = os.path.join("data", "offline")
DATASET_DIR = os.path.join("data", "datasets")


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    columns: list


def _load_ticks(symbol: str) -> pd.DataFrame:
    symbol = symbol.upper()
    fname = f"{symbol}_ticks.csv"
    path = os.path.join(OFFLINE_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Offline ticks file not found: {path}")

    df = pd.read_csv(path)
    required = ["timestamp", "price"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column '{col}'")

    # заповнюємо відсутні колонки дефолтами
    if "qty" not in df.columns:
        df["qty"] = 0.0

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_dataset(symbol: str, horizon: int = 3, min_rows: int = 50) -> Dataset:
    """
    Створює supervised dataset для класифікації напрямку руху ціни.
    horizon = скільки тиков вперед дивимось.
    """
    df = _load_ticks(symbol)
    symbol = symbol.upper()

    # базові ряди
    df["mid"] = df["price"].astype(float)
    df["ret_1"] = df["mid"].pct_change()
    df["ret_log_1"] = np.log(df["mid"] / df["mid"].shift(1))

    # rolling features
    for w in (3, 5, 10):
        df[f"ret_mean_{w}"] = df["ret_1"].rolling(w).mean()
        df[f"ret_std_{w}"] = df["ret_1"].rolling(w).std()
        df[f"vol_sum_{w}"] = df["qty"].rolling(w).sum()

    # ціль: рух ціни через horizon тиков
    df["future_mid"] = df["mid"].shift(-horizon)
    df["future_ret"] = (df["future_mid"] - df["mid"]) / df["mid"]
    df["y"] = (df["future_ret"] > 0).astype(int)

    # чистка NaN
    df = df.dropna().reset_index(drop=True)

    if len(df) < min_rows:
        print(f"[DATASET] Warning: only {len(df)} rows, less than min_rows={min_rows}")

    feature_cols = [
        "ret_1",
        "ret_log_1",
        "ret_mean_3",
        "ret_std_3",
        "ret_mean_5",
        "ret_std_5",
        "ret_mean_10",
        "ret_std_10",
        "vol_sum_3",
        "vol_sum_5",
        "vol_sum_10",
    ]

    X = df[feature_cols].values.astype(float)
    y = df["y"].values.astype(int)

    os.makedirs(DATASET_DIR, exist_ok=True)
    out_path = os.path.join(DATASET_DIR, f"{symbol}_h{horizon}.parquet")
    df_out = df[feature_cols + ["y"]].copy()
    df_out.to_parquet(out_path, index=False)
    print(f"[DATASET] Saved dataset to {out_path}, shape={df_out.shape}")

    return Dataset(X=X, y=y, columns=feature_cols)


def load_dataset(symbol: str, horizon: int = 3) -> Dataset:
    """
    Завантажити вже збережений датасет, якщо він є.
    """
    symbol = symbol.upper()
    path = os.path.join(DATASET_DIR, f"{symbol}_h{horizon}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c != "y"]
    X = df[feature_cols].values.astype(float)
    y = df["y"].values.astype(int)
    return Dataset(X=X, y=y, columns=feature_cols)
