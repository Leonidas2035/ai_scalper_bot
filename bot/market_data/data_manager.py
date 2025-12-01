import csv
import json
from pathlib import Path
from datetime import datetime
from bot.core.config_loader import config


class DataManager:
    """
    Зберігає тики у форматі:
    data/ticks/BTCUSDT_2025-01-15.csv

    та стакан у форматі:
    data/orderbooks/BTCUSDT_2025-01-15.jsonl
    """

    def __init__(self):
        base = Path(config.get("app.data_path"))
        self.ticks_path = base / "ticks"
        self.ob_path = base / "orderbooks"

        self.ticks_path.mkdir(parents=True, exist_ok=True)
        self.ob_path.mkdir(parents=True, exist_ok=True)

        self.open_files = {}

    def _get_tick_writer(self, symbol):
        date = datetime.utcnow().strftime("%Y-%m-%d")
        file_path = self.ticks_path / f"{symbol}_{date}.csv"

        if file_path not in self.open_files:
            f = open(file_path, "a", newline="")
            writer = csv.writer(f)
            # якщо файл новий → пишемо хедер
            if f.tell() == 0:
                writer.writerow(["timestamp", "price", "qty", "side"])
            self.open_files[file_path] = (f, writer)

        return self.open_files[file_path]

    def _get_ob_file(self, symbol):
        date = datetime.utcnow().strftime("%Y-%m-%d")
        file_path = self.ob_path / f"{symbol}_{date}.jsonl"
        return open(file_path, "a")

    async def save_trade(self, data):
        if not config.get("storage.save_trades"):
            return

        ts = int(data["T"])
        price = float(data["p"])
        qty = float(data["q"])
        side = "sell" if data["m"] else "buy"

        f, writer = self._get_tick_writer(data["s"])
        writer.writerow([ts, price, qty, side])

    async def save_orderbook(self, data):
        if not config.get("storage.save_orderbook"):
            return

        symbol = data.get("s", "UNKNOWN")
        ts = int(data.get("E"))

        record = {
            "timestamp": ts,
            "bids": data.get("b", []),
            "asks": data.get("a", []),
        }

        with self._get_ob_file(symbol) as f:
            f.write(json.dumps(record) + "\n")
