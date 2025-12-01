import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import xgboost as xgb

MODELS_DIR = os.path.join("storage", "models")


@dataclass
class SignalOutput:
    p_up: float
    p_down: float
    edge: float
    direction: int  # +1 long, -1 short, 0 flat


class SignalModel:
    def __init__(self, symbol: str, horizon: int = 1):
        self.symbol = symbol.upper()
        self.horizon = horizon

        model_path = os.path.join(
            MODELS_DIR, f"signal_xgb_{self.symbol}_h{self.horizon}.json"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                f"Train model first or put trained XGBoost model here."
            )

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

    def predict_proba(self, features: np.ndarray) -> SignalOutput:
        if features.ndim == 1:
            features = features.reshape(1, -1)

        proba = self.model.predict_proba(features)[0]  # [p_down, p_up]
        p_down = float(proba[0])
        p_up = float(proba[1])
        edge = p_up - 0.5

        if p_up > 0.5:
            direction = 1
        elif p_up < 0.5:
            direction = -1
        else:
            direction = 0

        return SignalOutput(
            p_up=p_up,
            p_down=p_down,
            edge=edge,
            direction=direction,
        )
