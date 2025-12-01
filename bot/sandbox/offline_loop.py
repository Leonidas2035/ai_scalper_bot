import argparse
from pathlib import Path

import pandas as pd

from bot.engine.decision_engine import DecisionEngine
from bot.ml.ensemble import EnsembleSignalModel
from bot.ml.signal_model.model import SignalOutput
from bot.ml.signal_model.online_features import OnlineFeatureBuilder
from bot.trading.paper_trader import PaperTrader

DEFAULT_TICKS = Path("data") / "ticks" / "BTCUSDT_synthetic.csv"


def load_ticks(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] Tick file not found: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        required = {"timestamp", "price", "qty", "side"}
        missing = required - set(df.columns)
        if missing:
            print(f"[ERROR] Tick file missing columns: {missing}")
            return pd.DataFrame()
        return df
    except Exception as exc:
        print(f"[ERROR] Failed to load ticks: {exc}")
        return pd.DataFrame()


def run_backtest(ticks_path: Path, symbol: str = "BTCUSDT"):
    df = load_ticks(ticks_path)
    if df.empty:
        print("[ERROR] No ticks to run offline loop.")
        return

    print(f"[INFO] Loaded {len(df)} ticks from {ticks_path}")

    ensemble = EnsembleSignalModel(symbol=symbol, horizons=[1, 3, 10])
    if not ensemble.models:
        print("[ERROR] No models available for ensemble. Train models first.")
        return

    feature_builder = OnlineFeatureBuilder()
    engine = DecisionEngine(min_confidence=0.55, min_edge=0.0)
    trader = PaperTrader()

    ticks_used = 0

    for _, row in df.iterrows():
        ts = int(row["timestamp"])
        price = float(row["price"])
        qty = float(row["qty"])

        features = feature_builder.add_tick(ts, price, qty)
        if features is None:
            continue

        block, reason = EnsembleSignalModel.filter_blocks(features)
        if block:
            continue

        ticks_used += 1
        ens_out = ensemble.predict(features)
        if not ens_out.components:
            continue

        p_up = 0.5 + ens_out.meta_edge
        p_down = 0.5 - ens_out.meta_edge
        pseudo_signal = SignalOutput(p_up=p_up, p_down=p_down, edge=ens_out.meta_edge, direction=ens_out.direction)

        decision = engine.decide(pseudo_signal, price, position=int(trader.position))
        trader.process_sync(decision, price, ts)

    summary = trader.summary()
    print("----- Offline Backtest Summary -----")
    print(f"Ticks processed: {ticks_used}")
    print(f"Trades: {summary['trades']}")
    print(f"Final PnL: {summary['realized_pnl'] + summary['open_pnl']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline backtest loop.")
    parser.add_argument("--ticks-path", type=Path, default=DEFAULT_TICKS, help="Path to tick CSV file")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    return parser.parse_args()


def main():
    args = parse_args()
    run_backtest(ticks_path=args.ticks_path, symbol=args.symbol)


if __name__ == "__main__":
    main()
