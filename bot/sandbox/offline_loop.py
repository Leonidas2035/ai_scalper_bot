import csv
import os
import numpy as np

from bot.ml.signal_model.online_features import OnlineFeatureBuilder
from bot.ml.signal_model.model import SignalModel
from bot.engine.decision_engine import DecisionEngine


OFFLINE_TICKS_PATH = os.path.join("data", "ticks", "BTCUSDT_synthetic.csv")


def load_ticks(filepath: str):
    if not os.path.exists(filepath):
        print(f"[ERROR] Ticks file not found: {filepath}")
        return []

    ticks = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = int(row["timestamp"])
                price = float(row["price"])
                qty = float(row.get("qty") or 0.0)
                ticks.append({"timestamp": ts, "price": price, "qty": qty})
            except Exception:
                continue
    return ticks


def offline_backtest(
    symbol: str = "BTCUSDT",
    file_path: str = OFFLINE_TICKS_PATH,
    horizon: int = 1,
):
    print(f"[INFO] Loading ticks from {file_path} ...")
    ticks = load_ticks(file_path)
    print(f"[INFO] Loaded {len(ticks)} ticks")

    if not ticks:
        print("[ERROR] No ticks loaded. Check file path or generate data.")
        return

    try:
        model = SignalModel(symbol=symbol, horizon=horizon)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[HINT] Run training first: python -m bot.ml.signal_model.train")
        return

    fb = OnlineFeatureBuilder(window=200)
    risk = DecisionEngine(balance_usdt=1000, max_risk_per_trade=0.005)

    position = 0.0
    entry_price = 0.0
    pnl = 0.0
    trades = 0
    last_price = None

    for t in ticks:
        ts = t["timestamp"]
        price = t["price"]
        qty = t["qty"]
        last_price = price

        feat = fb.update(ts, price, qty)
        if feat is None:
            continue

        signal = model.predict_proba(feat)
        decision = risk.decide(signal, price, position)

        if decision.action == "open_long":
            position = decision.size
            entry_price = price
            trades += 1
            print(f"[OPEN LONG] size={position} entry={price}")

        elif decision.action == "open_short":
            position = -decision.size
            entry_price = price
            trades += 1
            print(f"[OPEN SHORT] size={position} entry={price}")

        elif decision.action == "close" and position != 0:
            if position > 0:
                trade_pnl = (price - entry_price) * abs(position)
            else:
                trade_pnl = (entry_price - price) * abs(position)

            pnl += trade_pnl
            print(f"[CLOSE] trade_pnl={trade_pnl:.3f} total={pnl:.3f} exit={price}")
            position = 0.0

    if position != 0 and last_price is not None:
        if position > 0:
            trade_pnl = (last_price - entry_price) * abs(position)
        else:
            trade_pnl = (entry_price - last_price) * abs(position)
        pnl += trade_pnl
        print(f"[FORCE CLOSE] trade_pnl={trade_pnl:.3f} total={pnl:.3f} exit={last_price}")

    print("\n========= RESULT =========")
    print(f"Symbol: {symbol}  Horizon: {horizon}")
    print(f"Ticks: {len(ticks)}")
    print(f"Trades: {trades}")
    print(f"Final PnL: {pnl:.3f} USDT")
    print("==========================")


if __name__ == "__main__":
    offline_backtest()
