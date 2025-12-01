import asyncio
import time

from bot.ai.risk_moderator import LLMRiskModerator
from bot.core.config_loader import config
from bot.engine.decision_engine import DecisionEngine
from bot.market_data.mock_ws_manager import MockWSManager
from bot.ml.ensemble import EnsembleSignalModel, EnsembleOutput
from bot.ml.signal_model.model import SignalOutput
from bot.ml.signal_model.online_features import OnlineFeatureBuilder
from bot.trading.paper_trader import PaperTrader
from bot.market_data.data_manager import DataManager


async def _event_stream():
    websocket_type = config.get("app.websocket", "mock")
    symbols = config.get("binance.symbols", ["BTCUSDT"])

    if websocket_type != "mock":
        print("[WARN] Binance websocket disabled in this environment. Falling back to mock.")

    mock = MockWSManager(symbols)
    async for event in mock.stream():
        yield event


def _build_signal_from_meta(meta: EnsembleOutput) -> SignalOutput:
    p_up = 0.5 + meta.meta_edge
    p_down = 0.5 - meta.meta_edge
    return SignalOutput(p_up=p_up, p_down=p_down, edge=meta.meta_edge, direction=meta.direction)


async def main():
    symbols = config.get("binance.symbols", ["BTCUSDT"])
    symbol = symbols[0]
    ensemble = EnsembleSignalModel(symbol=symbol, horizons=[1, 3, 10])
    if not ensemble.models:
        print("[ERROR] Ensemble has no loaded models. Train models first with python -m bot.ml.signal_model.train.")
        return

    feature_builder = OnlineFeatureBuilder()
    app_risk = config.get("app.risk", {}) or {}
    min_edge = app_risk.get("llm_require_edge", 0.0)
    engine = DecisionEngine(min_confidence=0.55, min_edge=min_edge)
    trader = PaperTrader()
    risk_mod = LLMRiskModerator()
    data_manager = DataManager()

    llm_enabled = bool(config.get("app.llm_enabled", True))

    last_report = time.time()
    report_interval = 5.0

    async for event in _event_stream():
        try:
            ts = int(event.get("E") or event.get("T") or time.time() * 1000)
            price = float(event["p"])
            qty = float(event["q"])
        except Exception:
            continue

        await data_manager.save_trade(event)

        features = feature_builder.add_tick(ts, price, qty)
        if features is None:
            continue

        block, reason = EnsembleSignalModel.filter_blocks(features)
        if block:
            continue

        meta = ensemble.predict(features)
        if not meta.components:
            continue

        pseudo_signal = _build_signal_from_meta(meta)

        approved = True
        if llm_enabled:
            shock = abs(float(features[0]))
            market_context = {
                "drawdown": 0.0,  # placeholder for real equity curve tracking
                "exposure": abs(trader.position),
                "shock": shock,
            }
            verdict = await risk_mod.evaluate(features, pseudo_signal, market_context)
            approved = verdict.get("approve", True)

        decision = engine.decide(pseudo_signal, price, position=int(trader.position), approved=approved)
        await trader.process(decision, price, ts)

        now = time.time()
        if now - last_report >= report_interval:
            summary = trader.summary()
            print(
                f"[STATS] pos={summary['position']:.2f} trades={summary['trades']} "
                f"pnl={summary['realized_pnl'] + summary['open_pnl']:.4f} meta_edge={meta.meta_edge:.4f}"
            )
            last_report = now


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down.")
