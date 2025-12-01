from collections import deque
from typing import Optional

import numpy as np

from bot.ml.signal_model.dataset_builder import FEATURE_COLS


class OnlineFeatureBuilder:
    """
    Incrementally builds the same feature vector as DatasetBuilder for live/offline streaming ticks.
    """

    def __init__(self, max_window: int = 10):
        self.max_window = max_window
        self.prices = deque(maxlen=max_window + 1)
        self.qty = deque(maxlen=max_window)

    def add_tick(self, timestamp: int, price: float, qty: float) -> Optional[np.ndarray]:
        self.prices.append(float(price))
        self.qty.append(float(qty))

        if len(self.prices) < self.max_window + 1:
            return None

        price_arr = np.array(self.prices, dtype=float)
        qty_arr = np.array(self.qty, dtype=float)

        ret_1 = price_arr[1:] / price_arr[:-1] - 1.0
        ret_log_1 = np.diff(np.log(price_arr))

        def _mean(arr, window):
            return float(np.mean(arr[-window:]))

        def _std(arr, window):
            return float(np.std(arr[-window:]))

        features = [
            float(ret_1[-1]),
            float(ret_log_1[-1]),
            _mean(ret_1, 3),
            _std(ret_1, 3),
            _mean(ret_1, 5),
            _std(ret_1, 5),
            _mean(ret_1, 10),
            _std(ret_1, 10),
            float(np.sum(qty_arr[-3:])),
            float(np.sum(qty_arr[-5:])),
            float(np.sum(qty_arr[-10:])),
        ]

        return np.array(features, dtype=float)
