import os
import pandas as pd
import numpy as np
from pathlib import Path

FEATURE_COLS = [
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

TARGET_COL = "target"


class DatasetBuilder:
    """
    Створює навчальний датасет з CSV тиков у форматі:
    timestamp,price,qty,side
    """

    def __init__(self, ticks_path="data/ticks", symbol="BTCUSDT", horizon=1):
        self.ticks_path = Path(ticks_path)
        self.symbol = symbol
        self.horizon = horizon

    def load_ticks(self):
        dfs = []
        for f in self.ticks_path.glob(f"{self.symbol}_*.csv"):
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except:
                pass

        if not dfs:
            raise FileNotFoundError(f"No tick files found for {self.symbol}")

        df = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
        return df

    def build_features(self, df: pd.DataFrame):
        df["mid"] = df["price"].astype(float)

        df["ret_1"] = df["mid"].pct_change()
        df["ret_log_1"] = np.log(df["mid"] / df["mid"].shift(1))

        for w in (3, 5, 10):
            r = df["ret_1"].rolling(w)
            df[f"ret_mean_{w}"] = r.mean()
            df[f"ret_std_{w}"] = r.std()
            df[f"vol_sum_{w}"] = df["qty"].rolling(w).sum()

        return df

    def build_target(self, df: pd.DataFrame):
        """
        TARGET:
        Якщо ціна через 'horizon' тиков вище → 1 (LONG)
        Якщо нижче → 0 (SHORT)
        """
        df["future"] = df["mid"].shift(-self.horizon)
        df[TARGET_COL] = (df["future"] > df["mid"]).astype(int)
        return df

    def build_dataset(self):
        df = self.load_ticks()
        df = self.build_features(df)
        df = self.build_target(df)

        df = df.dropna().reset_index(drop=True)

        # Вибираємо тільки потрібні фічі
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        return X, y, df

    def save_parquet(self, df, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"[OK] Saved dataset: {out_path}")
