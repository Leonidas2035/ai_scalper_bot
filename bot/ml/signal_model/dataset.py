import json
import numpy as np
import pandas as pd
from pathlib import Path

from bot.indicators.feature_builder import FeatureBuilder


class SignalDataset:

    def __init__(self, data_path="./data", symbol="BTCUSDT"):
        self.data_path = Path(data_path)
        self.symbol = symbol
        self.fb = FeatureBuilder(data_path)

    def create_dataset(self, limit=5000):
        rows = []

        for _ in range(limit):
            f = self.fb.build(self.symbol)
            if f:
                rows.append(f)

        df = pd.DataFrame(rows)
        df = df.dropna()

        # Create LABEL = direction change
        df["future_close"] = df["vwap"].shift(-1)
        df["label_raw"] = df["future_close"] - df["vwap"]

        df["label"] = df["label_raw"].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )

        df = df.dropna()

        X = df.drop(columns=["future_close", "label_raw", "label"])
        y = df["label"]

        return X, y
