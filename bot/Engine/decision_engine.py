from dataclasses import dataclass
from typing import Optional

from bot.ml.signal_model.model import SignalOutput


@dataclass
class Decision:
    action: str
    size: float
    order_type: str
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None


class DecisionEngine:
    def __init__(
        self,
        balance_usdt: float,
        max_risk_per_trade: float = 0.005,
        edge_min: float = 0.02,
        leverage: float = 5.0,
    ):
        self.balance_usdt = balance_usdt
        self.max_risk_per_trade = max_risk_per_trade
        self.edge_min = edge_min
        self.leverage = leverage

    def decide(self, signal: SignalOutput, price: float, current_position: float) -> Decision:
        if abs(signal.edge) < self.edge_min:
            return Decision(action="hold", size=0, order_type="market")

        if current_position != 0:
            if (current_position > 0 and signal.direction < 0) or \
               (current_position < 0 and signal.direction > 0):
                return Decision(action="close", size=abs(current_position), order_type="market")
            return Decision(action="hold", size=0, order_type="market")

        risk_usdt = self.balance_usdt * self.max_risk_per_trade
        sl_pct = 0.005
        sl_distance = price * sl_pct

        size = (risk_usdt * self.leverage) / sl_distance
        size = round(size, 3)

        if signal.direction > 0:
            return Decision(
                action="open_long",
                size=size,
                order_type="market",
                sl_price=price - sl_distance,
                tp_price=price + sl_distance * 1.5,
            )

        if signal.direction < 0:
            return Decision(
                action="open_short",
                size=size,
                order_type="market",
                sl_price=price + sl_distance,
                tp_price=price - sl_distance * 1.5,
            )

        return Decision(action="hold", size=0, order_type="market")
