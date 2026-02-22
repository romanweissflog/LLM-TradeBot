import asyncio
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.symbol_selector_agent import SymbolSelectorAgent


def _build_klines(prices: List[float], volumes: List[float]) -> List[Dict]:
    rows = []
    for i, (price, volume) in enumerate(zip(prices, volumes)):
        rows.append(
            {
                "timestamp": i * 60_000,
                "open": float(price),
                "high": float(price * 1.003),
                "low": float(price * 0.997),
                "close": float(price),
                "volume": float(volume),
            }
        )
    return rows


class DummyBinanceClient:
    def __init__(self, *args, **kwargs):
        self._series = {
            "AAAUSDT": {
                "1m": _build_klines(
                    [1.0 + 0.0022 * i for i in range(120)],
                    [120.0] * 90 + [280.0] * 30,
                ),
                "15m": _build_klines(
                    [1.0 + 0.01 * i for i in range(80)],
                    [1000.0 + i * 4 for i in range(80)],
                ),
                "1h": _build_klines(
                    [1.0 + 0.03 * i for i in range(80)],
                    [1800.0 + i * 5 for i in range(80)],
                ),
            },
            "BBBUSDT": {
                "1m": _build_klines(
                    [1.4 - 0.0023 * i for i in range(120)],
                    [110.0] * 90 + [270.0] * 30,
                ),
                "15m": _build_klines(
                    [1.8 - 0.012 * i for i in range(80)],
                    [1050.0 + i * 4 for i in range(80)],
                ),
                "1h": _build_klines(
                    [2.6 - 0.035 * i for i in range(80)],
                    [1750.0 + i * 5 for i in range(80)],
                ),
            },
            "CCCUSDT": {
                "1m": _build_klines(
                    [1.2 + (0.0001 * (-1) ** i) for i in range(120)],
                    [140.0] * 120,
                ),
                "15m": _build_klines(
                    [1.2 + 0.0002 * i for i in range(80)],
                    [900.0] * 80,
                ),
                "1h": _build_klines(
                    [1.2 + 0.0005 * i for i in range(80)],
                    [1200.0] * 80,
                ),
            },
        }

    def get_all_tickers(self):
        return [
            {
                "symbol": "AAAUSDT",
                "quoteVolume": "25000000",
                "lastPrice": "1.26",
                "priceChangePercent": "9.5",
            },
            {
                "symbol": "BBBUSDT",
                "quoteVolume": "22000000",
                "lastPrice": "1.12",
                "priceChangePercent": "-8.7",
            },
            {
                "symbol": "CCCUSDT",
                "quoteVolume": "18000000",
                "lastPrice": "1.20",
                "priceChangePercent": "0.1",
            },
        ]

    def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time: int = None):
        data = self._series[symbol][interval]
        return data[-limit:]


def test_timeframe_alignment_bias():
    selector = SymbolSelectorAgent()
    up_closes = [1.0 + 0.01 * i for i in range(90)]
    down_closes = [2.0 - 0.012 * i for i in range(90)]

    assert selector._compute_timeframe_alignment(up_closes, direction=1) > 0.3
    assert selector._compute_timeframe_alignment(up_closes, direction=-1) < -0.3
    assert selector._compute_timeframe_alignment(down_closes, direction=-1) > 0.3


def test_directional_score_rewards_trend_quality():
    selector = SymbolSelectorAgent()
    strong = selector._build_directional_score(
        change_pct=2.1,
        volume_ratio=2.5,
        adx_15m=32.0,
        adx_1h=36.0,
        consistency=0.9,
        alignment_score=0.8,
        day_change_pct=7.0,
    )
    weak = selector._build_directional_score(
        change_pct=1.0,
        volume_ratio=1.0,
        adx_15m=14.0,
        adx_1h=12.0,
        consistency=0.45,
        alignment_score=-0.2,
        day_change_pct=-2.0,
    )
    assert strong > weak


def test_auto1_prefers_clear_up_and_down_candidates(monkeypatch):
    import src.api.binance_client as binance_client_module

    monkeypatch.setattr(binance_client_module, "BinanceClient", DummyBinanceClient)
    selector = SymbolSelectorAgent()

    selected = asyncio.run(
        selector.select_auto1_recent_momentum(
            candidates=["AAAUSDT", "BBBUSDT", "CCCUSDT"],
            window_minutes=30,
            interval="1m",
            threshold_pct=0.6,
            volume_ratio_threshold=1.1,
            min_adx=16,
            min_directional_score=1.0,
            min_alignment_score=-0.1,
            relax_factor=0.8,
        )
    )

    assert "AAAUSDT" in selected
    assert "BBBUSDT" in selected
    assert "CCCUSDT" not in selected
