from dataclasses import dataclass
from typing import Optional

from bot.ml.signal_model.model import SignalOutput


@dataclass
class Decision:
    action: str          # "open_long", "open_short", "close", "hold"
    size: float          # у контрактах
    order_type: str      # "market" / "limit"
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None


class DecisionEngine:
    def __init__(
        self,
        balance_usdt: float,
        max_risk_per_trade: float = 0.005,  # 0.5%
        edge_min: float = 0.02,
        leverage: float = 5.0,
    ):
        self.balance_usdt = balance_usdt
        self.max_risk_per_trade = max_risk_per_trade
        self.edge_min = edge_min
        self.leverage = leverage

    def decide(
        self,
        signal: SignalOutput,
        price: float,
        current_position: float,
    ) -> Decision:
        # фільтр за edge
        if abs(signal.edge) < self.edge_min:
            return Decision(action="hold", size=0.0, order_type="market")

        # якщо вже є позиція — простий логіка:
        if current_position != 0:
            # наприклад: якщо модель розвернулась проти нас — закриваємо
            if (current_position > 0 and signal.direction < 0) or (
                current_position < 0 and signal.direction > 0
            ):
                return Decision(action="close", size=abs(current_position), order_type="market")
            else:
                return Decision(action="hold", size=0.0, order_type="market")

        # нова позиція
        # обчислюємо ризик у доларах
        risk_usdt = self.balance_usdt * self.max_risk_per_trade

        # умовний SL на 0.5%
        sl_pct = 0.005
        sl_distance = price * sl_pct

        # приблизний розмір у контрактах:
        size = (risk_usdt * self.leverage) / sl_distance
        size = round(size, 3)

        if signal.direction > 0:
            action = "open_long"
        elif signal.direction < 0:
            action = "open_short"
        else:
            return Decision(action="hold", size=0.0, order_type="market")

        # SL / TP у цінах
        if action == "open_long":
            sl_price = price - sl_distance
            tp_price = price + sl_distance * 1.5
        else:
            sl_price = price + sl_distance
            tp_price = price - sl_distance * 1.5

        return Decision(
            action=action,
            size=size,
            order_type="market",
            sl_price=sl_price,
            tp_price=tp_price,
        )
