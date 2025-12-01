import pandas as pd
import numpy as np
from typing import Optional

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


class OnlineFeatureBuilder:
    """
    Онлайн-обчислювач фіч, які відповідають тренувальному датасету:
    ret_1, ret_log_1, ret_mean/std_{3,5,10}, vol_sum_{3,5,10}
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.df = pd.DataFrame(columns=["timestamp", "price", "qty"])

    def update(self, ts: int, price: float, qty: float) -> Optional[np.ndarray]:
        row = {"timestamp": ts, "price": float(price), "qty": float(qty)}
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

        # тримаємо тільки останні N тиков
        if len(self.df) > self.window:
            self.df = self.df.iloc[-self.window :].reset_index(drop=True)

        # замало даних для rolling — ще нічого не рахуємо
        if len(self.df) < 20:
            return None

        df = self.df.copy()
        df["mid"] = df["price"].astype(float)

        df["ret_1"] = df["mid"].pct_change()
        df["ret_log_1"] = np.log(df["mid"] / df["mid"].shift(1))

        for w in (3, 5, 10):
            r = df["ret_1"].rolling(w)
            df[f"ret_mean_{w}"] = r.mean()
            df[f"ret_std_{w}"] = r.std()
            df[f"vol_sum_{w}"] = df["qty"].rolling(w).sum()

        last = df.iloc[-1]

        # якщо базові фічі ще NaN — чекаємо далі
        if last[["ret_1", "ret_log_1"]].isna().any():
            return None

        feats = []
        for col in FEATURE_COLS:
            val = last.get(col)
            if pd.isna(val):
                return None
            feats.append(float(val))

        return np.asarray(feats, dtype=float)
