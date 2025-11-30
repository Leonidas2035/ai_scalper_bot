import asyncio
import json
import websockets
import traceback

from bot.core.config_loader import config
from bot.market_data.data_manager import DataManager


class WSManager:
    def __init__(self):
        self.base_ws = config.get("binance.ws_spot_base")  # wss://stream.binance.com:9443
        self.symbols = config.get("binance.symbols")
        self.data_manager = DataManager()

    def _build_stream_url(self):
        streams = []
        for sym in self.symbols:
            s = sym.lower()
            streams.append(f"{s}@trade")
            streams.append(f"{s}@depth20@100ms")

        stream_str = "/".join(streams)
        return f"{self.base_ws}/stream?streams={stream_str}"

    async def connect(self):
        url = self._build_stream_url()
        print(f"[WS] Connecting to: {url}")

        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    print("[WS] Connected!")
                    async for msg in ws:
                        await self.process_message(msg)

            except Exception as e:
                print("[WS] Error:", e)
                print("[WS] Reconnecting in 2s...")
                await asyncio.sleep(2)

    async def process_message(self, msg):
        try:
            data = json.loads(msg)

            stream = data.get("stream")
            payload = data.get("data")

            if not payload:
                return

            if "trade" in stream:
                await self.data_manager.save_trade(payload)

            elif "depth" in stream:
                await self.data_manager.save_orderbook(payload)

        except Exception:
            traceback.print_exc()


async def main():
    ws = WSManager()
    await ws.connect()


if __name__ == "__main__":
    asyncio.run(main())
