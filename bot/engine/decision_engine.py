from dataclasses import dataclass
from typing import Optional

from bot.ml.signal_model.model import SignalOutput


@dataclass
class RiskParams:
    max_risk_per_trade: float = 0.01  # fraction of equity per trade
    leverage: float = 1.0


@dataclass
class Decision:
    action: str  # "buy", "sell", "close", "hold"
    size: float = 0.0
    order_type: str = "market"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class DecisionEngine:
    """Threshold-based decision engine using SignalOutput probabilities."""

    def __init__(self, min_confidence: float = 0.55, min_edge: float = 0.0, risk_params: Optional[RiskParams] = None):
        self.min_confidence = min_confidence
        self.min_edge = min_edge
        self.risk = risk_params or RiskParams()

    def decide(self, signal: SignalOutput, price: float, position: int, approved: bool = True) -> Decision:
        size = self.risk.max_risk_per_trade * self.risk.leverage

        if not approved or signal.edge < self.min_edge:
            return Decision(action="hold")

        if position == 0:
            if signal.p_up >= self.min_confidence:
                return Decision(action="buy", size=size)
            if signal.p_down >= self.min_confidence:
                return Decision(action="sell", size=size)
            return Decision(action="hold")

        if position > 0:
            if signal.p_down >= self.min_confidence:
                return Decision(action="close")
            return Decision(action="hold")

        if position < 0:
            if signal.p_up >= self.min_confidence:
                return Decision(action="close")
            return Decision(action="hold")

        return Decision(action="hold")
