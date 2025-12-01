import asyncio
import random
from dataclasses import dataclass, field
from typing import List, Optional

from bot.engine.decision_engine import Decision


@dataclass
class PaperTrade:
    timestamp: int
    action: str
    price: float
    size: float
    fee: float
    pnl: float


class PaperTrader:
    """
    Minimal paper trading executor: applies decisions, simulates small latency and fees,
    and tracks positions/PnL.
    """

    def __init__(self, fee_bps: float = 2.0, latency_ms_range=(2, 5)):
        self.fee_rate = fee_bps / 10_000  # bps to fraction
        self.latency_ms_range = latency_ms_range
        self.position: float = 0.0
        self.entry_price: Optional[float] = None
        self.realized_pnl: float = 0.0
        self.trades: List[PaperTrade] = []

    async def _latency(self):
        delay = random.uniform(*self.latency_ms_range) / 1000.0
        if delay > 0:
            await asyncio.sleep(delay)

    def _fee(self, notional: float) -> float:
        return abs(notional) * self.fee_rate

    async def process(self, decision: Decision, price: float, timestamp: int):
        await self._latency()

        if decision.action == "hold":
            return

        size = decision.size if decision.size else 1.0
        fee = self._fee(price * size)
        pnl = 0.0

        # Close existing position if opposite signal arrives
        if decision.action in ("buy", "close") and self.position < 0:
            pnl = (self.entry_price - price) * abs(self.position) - fee
            self.realized_pnl += pnl
            self.trades.append(PaperTrade(timestamp, "close_short", price, self.position, fee, pnl))
            self.position = 0.0
            self.entry_price = None

        if decision.action in ("sell", "close") and self.position > 0:
            pnl = (price - self.entry_price) * abs(self.position) - fee
            self.realized_pnl += pnl
            self.trades.append(PaperTrade(timestamp, "close_long", price, self.position, fee, pnl))
            self.position = 0.0
            self.entry_price = None

        # Open new position if flat and actionable
        if decision.action == "buy" and self.position == 0:
            self.position = size
            self.entry_price = price
            self.trades.append(PaperTrade(timestamp, "open_long", price, size, fee, pnl))
        elif decision.action == "sell" and self.position == 0:
            self.position = -size
            self.entry_price = price
            self.trades.append(PaperTrade(timestamp, "open_short", price, size, fee, pnl))

    def summary(self):
        open_pnl = 0.0
        if self.position and self.entry_price:
            # mark-to-market with last known trade price from order log
            last_price = self.trades[-1].price if self.trades else self.entry_price
            open_pnl = (last_price - self.entry_price) * self.position
        return {
            "position": self.position,
            "entry_price": self.entry_price,
            "realized_pnl": self.realized_pnl,
            "open_pnl": open_pnl,
            "trades": len(self.trades),
        }

    def process_sync(self, decision: Decision, price: float, timestamp: int):
        """
        Convenience wrapper for non-async contexts.
        """
        import asyncio
        asyncio.run(self.process(decision, price, timestamp))
