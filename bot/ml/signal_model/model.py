import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import xgboost as xgb

from .dataset_builder import load_dataset

MODELS_DIR = os.path.join("storage", "models")


@dataclass
class SignalOutput:
    p_up: float
    p_down: float
    edge: float
    direction: int  # +1 long, -1 short, 0 no-trade


class SignalModel:
    def __init__(self, symbol: str, horizon: int = 3, edge_threshold: float = 0.0):
        self.symbol = symbol.upper()
        self.horizon = horizon
        self.edge_threshold = edge_threshold

        model_path = os.path.join(
            MODELS_DIR, f"signal_xgb_{self.symbol}_h{self.horizon}.json"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # завантажуємо модель
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        # завантажуємо датасет, щоб знати order/кількість фіч
        ds = load_dataset(self.symbol, self.horizon)
        self.n_features = ds.X.shape[1]
        self.feature_cols = ds.columns  # список назв фіч (на всякий випадок)

    def _prepare_features(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {features.shape[1]}"
            )
        return features

    def predict_proba(self, features: np.ndarray) -> SignalOutput:
        X = self._prepare_features(features)
        proba = self.model.predict_proba(X)[0]

        # індекс 1 — p_up, індекс 0 — p_down, але приводимо до точної суми 1.0
        p_up = float(proba[1])
        p_down = 1.0 - p_up

        edge = p_up - 0.5

        if abs(edge) >= self.edge_threshold:
            direction = 1 if edge > 0 else -1
        else:
            direction = 0

        return SignalOutput(
            p_up=p_up,
            p_down=p_down,
            edge=edge,
            direction=direction,
        )
