import json
from pathlib import Path
from datetime import datetime
from bot.core.config_loader import config


class DataManager:
    def __init__(self):
        self.base = Path(config.get("app.data_path"))
        self.base.mkdir(exist_ok=True)

    def _save_json(self, folder: str, data: dict):
        path = self.base / folder
        path.mkdir(exist_ok=True)

        ts = int(data.get("T") or data.get("E") or datetime.utcnow().timestamp() * 1000)
        symbol = data.get("s", "UNKNOWN")

        file = path / f"{symbol}_{ts}.json"
        with open(file, "w") as f:
            json.dump(data, f)

    async def save_trade(self, data: dict):
        if config.get("storage.save_trades"):
            self._save_json("trades", data)

    async def save_orderbook(self, data: dict):
        if config.get("storage.save_orderbook"):
            self._save_json("orderbooks", data)
