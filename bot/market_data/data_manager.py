import json
from datetime import datetime
from pathlib import Path

from bot.core.config_loader import config


class DataManager:
    """
    Persists incoming trades/orderbooks and appends ticks to CSV with schema:
    timestamp,price,qty,side
    """

    def __init__(self):
        self.base = Path(config.get("app.data_path"))
        self.base.mkdir(exist_ok=True)
        (self.base / "ticks").mkdir(parents=True, exist_ok=True)

    def _save_json(self, folder: str, data: dict):
        path = self.base / folder
        path.mkdir(parents=True, exist_ok=True)

        ts = int(data.get("T") or data.get("E") or datetime.utcnow().timestamp() * 1000)
        symbol = data.get("s", "UNKNOWN")

        file = path / f"{symbol}_{ts}.json"
        with open(file, "w") as f:
            json.dump(data, f)

    async def save_trade(self, data: dict):
        if config.get("storage.save_trades"):
            self._save_json("trades", data)
        try:
            ts = int(data.get("T") or data.get("E") or datetime.utcnow().timestamp() * 1000)
            price = float(data.get("p"))
            qty = float(data.get("q"))
            side = "sell" if data.get("m") else "buy"
            self._append_tick(data.get("s", "UNKNOWN"), ts, price, qty, side)
        except Exception:
            # best-effort persistence; ignore malformed payloads
            pass

    async def save_orderbook(self, data: dict):
        if config.get("storage.save_orderbook"):
            self._save_json("orderbooks", data)

    def _append_tick(self, symbol: str, ts: int, price: float, qty: float, side: str):
        path = self.base / "ticks" / f"{symbol}_stream.csv"
        header_needed = not path.exists()
        with open(path, "a") as f:
            if header_needed:
                f.write("timestamp,price,qty,side\n")
            f.write(f"{ts},{price},{qty},{side}\n")
