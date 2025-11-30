import pandas as pd
import numpy as np
from typing import Optional, List


class OnlineFeatureBuilder:
    def __init__(self, window: int = 100):
        self.window = window
        self.df = pd.DataFrame(columns=["timestamp", "price", "qty"])

    def update(self, ts: int, price: float, qty: float) -> Optional[np.ndarray]:
        row = {"timestamp": ts, "price": price, "qty": qty}
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

        # тримаємо тільки останні N
        if len(self.df) > self.window:
            self.df = self.df.iloc[-self.window :].reset_index(drop=True)

        if len(self.df) < 20:
            # мало даних для rolling
            return None

        df = self.df.copy()
        df["mid"] = df["price"].astype(float)
        df["ret_1"] = df["mid"].pct_change()
        df["ret_log_1"] = np.log(df["mid"] / df["mid"].shift(1))

        for w in (3, 5, 10):
            df[f"ret_mean_{w}"] = df["ret_1"].rolling(w).mean()
            df[f"ret_std_{w}"] = df["ret_1"].rolling(w).std()
            df[f"vol_sum_{w}"] = df["qty"].rolling(w).sum()

        last = df.iloc[-1]

        if last[["ret_1", "ret_log_1"]].isna().any():
            return None

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

        if any(col not in last for col in feature_cols):
            return None

        feat = last[feature_cols].to_numpy(dtype=float)
        return feat
class OnlineFeatureBuilder:
    def __init__(self, window: int = 100):
        self.window = window
        self.df = pd.DataFrame(columns=["timestamp", "price", "qty"])
        self.feature_cols = [
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

