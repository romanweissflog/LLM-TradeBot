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


class DummyVolumeBiasClient:
    def __init__(self, *args, **kwargs):
        base_1m = _build_klines(
            [1.0 + 0.001 * i for i in range(120)],
            [100.0] * 90 + [100.0] * 30,
        )
        base_15m = _build_klines(
            [1.0 + 0.008 * i for i in range(80)],
            [800.0 + i * 3 for i in range(80)],
        )
        base_1h = _build_klines(
            [1.0 + 0.02 * i for i in range(80)],
            [1500.0 + i * 4 for i in range(80)],
        )
        self._series = {
            "AAAUSDT": {"1m": base_1m, "15m": base_15m, "1h": base_1h},
        }

    def get_all_tickers(self):
        return [
            {
                "symbol": "AAAUSDT",
                "quoteVolume": "25000000",
                "lastPrice": "1.12",
                "priceChangePercent": "5.0",
            }
        ]

    def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time: int = None):
        data = self._series[symbol][interval]
        return data[-limit:]


class DummySmoothVsChoppyClient:
    def __init__(self, *args, **kwargs):
        smooth_1m = _build_klines(
            [1.0 + 0.0015 * i for i in range(120)],
            [120.0] * 90 + [180.0] * 30,
        )
        choppy_prices = []
        for i in range(120):
            trend = 1.0 + 0.0015 * i
            noise = 0.015 if i % 2 else -0.015
            choppy_prices.append(trend + noise)
        choppy_1m = _build_klines(choppy_prices, [120.0] * 90 + [180.0] * 30)

        smooth_15m = _build_klines(
            [1.0 + 0.01 * i for i in range(80)],
            [900.0 + i * 3 for i in range(80)],
        )
        smooth_1h = _build_klines(
            [1.0 + 0.025 * i for i in range(80)],
            [1500.0 + i * 4 for i in range(80)],
        )
        self._series = {
            "SMOOTHUSDT": {"1m": smooth_1m, "15m": smooth_15m, "1h": smooth_1h},
            "CHOPPYUSDT": {"1m": choppy_1m, "15m": smooth_15m, "1h": smooth_1h},
        }

    def get_all_tickers(self):
        return [
            {
                "symbol": "SMOOTHUSDT",
                "quoteVolume": "28000000",
                "lastPrice": "1.18",
                "priceChangePercent": "6.1",
            },
            {
                "symbol": "CHOPPYUSDT",
                "quoteVolume": "28000000",
                "lastPrice": "1.18",
                "priceChangePercent": "6.1",
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


def test_auto1_volume_ratio_uses_per_bar_average(monkeypatch):
    import src.api.binance_client as binance_client_module

    monkeypatch.setattr(binance_client_module, "BinanceClient", DummyVolumeBiasClient)
    selector = SymbolSelectorAgent()

    selected = asyncio.run(
        selector.select_auto1_recent_momentum(
            candidates=["AAAUSDT"],
            window_minutes=30,
            interval="1m",
            threshold_pct=0.2,
            volume_ratio_threshold=0.9,
            min_adx=10,
            min_directional_score=0.1,
            min_alignment_score=-0.5,
            relax_factor=0.8,
        )
    )

    assert selected == ["AAAUSDT"]
    result = selector.last_auto1["results"]["AAAUSDT"]
    assert result["volume_ratio"] == 1.0


def test_auto1_prefers_smoother_impulse_when_move_size_similar(monkeypatch):
    import src.api.binance_client as binance_client_module

    monkeypatch.setattr(binance_client_module, "BinanceClient", DummySmoothVsChoppyClient)
    selector = SymbolSelectorAgent()

    selected = asyncio.run(
        selector.select_auto1_recent_momentum(
            candidates=["SMOOTHUSDT", "CHOPPYUSDT"],
            window_minutes=30,
            interval="1m",
            threshold_pct=0.3,
            volume_ratio_threshold=1.0,
            min_adx=10,
            min_directional_score=0.1,
            min_alignment_score=-0.5,
            relax_factor=0.8,
        )
    )

    assert selected == ["SMOOTHUSDT"]
    smooth_res = selector.last_auto1["results"]["SMOOTHUSDT"]
    choppy_res = selector.last_auto1["results"]["CHOPPYUSDT"]
    assert smooth_res["impulse_ratio"] > choppy_res["impulse_ratio"]
