import os
import time
import asyncio

from typing import List, TYPE_CHECKING
from datetime import datetime

from src.api.binance_client import BinanceClient
from src.config import Config
from src.agents.agent_config import AgentConfig

from src.utils.logger import log
from src.server.state import global_state

if TYPE_CHECKING:
    from src.agents.agent_provider import AgentProvider

class SymbolManager:
    def __init__(
        self,
        config: Config,
        agent_config: AgentConfig,
        client: BinanceClient,
        agent_provider: "AgentProvider",
        test_mode: bool = False
    ):
        self.agent_provider = agent_provider
        self.test_mode = test_mode
        self.client = client
        self._symbols = []
        self.agent_config = agent_config

        # Symbol selector cadence (AUTO1/AUTO3)
        self.selector_interval_sec = 10 * 60
        self.selector_last_run = 0.0
        self.selector_startup_done = False

        # 多币种支持: 优先级顺序
        # 1. 环境变量 TRADING_SYMBOLS (来自 .env，Dashboard 设置会更新这个)
        # 2. config.yaml 中的 trading.symbols (list)
        # 3. config.yaml 中的 trading.symbol (str/csv, 向后兼容)
        env_symbols = os.environ.get('TRADING_SYMBOLS', '').strip()

        if env_symbols:
            # Dashboard 设置的币种 (逗号分隔)
            self._symbols = [s.strip() for s in env_symbols.split(',') if s.strip()]
        else:
            # 从 config.yaml 读取
            symbols_config = config.get('trading.symbols', None)

            if symbols_config and isinstance(symbols_config, list):
                self._symbols = symbols_config
            else:
                # 向后兼容: 使用旧版 trading.symbol 配置 (支持 CSV 字符串 "BTCUSDT,ETHUSDT")
                symbol_str = config.get('trading.symbol', 'AI500_TOP5')  # ✅ 默认 AI500 Top 5
                if ',' in symbol_str:
                    self._symbols = [s.strip() for s in symbol_str.split(',') if s.strip()]
                else:
                    self._symbols = [symbol_str]

        self.use_auto3 = 'AUTO3' in self._symbols

        # Normalize legacy AUTO2 -> AUTO1
        if 'AUTO2' in self._symbols:
            self._symbols = ['AUTO1' if s == 'AUTO2' else s for s in self._symbols]

        if self.use_auto3:
            self._symbols = [s for s in self._symbols if s not in ('AUTO3', 'AUTO1')]
            # If AUTO3 was the only symbol, add temporary placeholder (will be replaced at startup)
            if not self._symbols:
                self._symbols = ['FETUSDT']  # Temporary, replaced by AUTO3 selection in main()
            log.info("🔝 AUTO3 mode enabled - Startup backtest will run")

        self.use_auto1 = (not self.use_auto3) and ('AUTO1' in self._symbols)
        if self.use_auto1:
            self._symbols = [s for s in self._symbols if s != 'AUTO1']
            if not self._symbols:
                self._symbols = ['BTCUSDT']  # Temporary placeholder before selector runs
            log.info("🎯 AUTO1 mode enabled - Symbol selector will run at startup")

        use_ai500 = 'AI500_TOP5' in self._symbols and not self.use_auto3
        if use_ai500:
            self._symbols.remove('AI500_TOP5')

            # Merge and deduplicate
            ai_top5 = self._resolve_ai500_symbols()
            self._symbols = list(set(self._symbols + ai_top5))

            # Sort to keep stable order
            self._symbols.sort()

        # 🔧 Primary symbol must be in the symbols list
        configured_primary = config.get('trading.primary_symbol', 'BTCUSDT')
        if configured_primary in self._symbols:
            self._primary_symbol = configured_primary
        else:
            # Use first symbol if configured primary not in list
            self._primary_symbol = self._symbols[0]
            log.info(f"Primary symbol {configured_primary} not in symbols list, using {self._primary_symbol}")

        self.current_symbol = self._primary_symbol  # 当前处理的交易对
        global_state.symbols = self._symbols  # 🆕 Sync symbols to global state for API

    @property
    def symbols(self):
        return self._symbols

    @symbols.setter
    def symbols(self, value):
        self._symbols = value

    @property
    def current_symbol(self):
        return self._current_symbol

    @current_symbol.setter
    def current_symbol(self, value):
        self._current_symbol = value
        global_state.current_symbol = value

    @property
    def primary_symbol(self):
        return self._primary_symbol

    @primary_symbol.setter
    def primary_symbol(self, value):
        self._primary_symbol = value

    def reload_symbols(self, config) -> bool:
        env_symbols = os.environ.get('TRADING_SYMBOLS', '').strip()

        old_symbols = self._symbols.copy()

        if env_symbols:
            self._symbols = [s.strip() for s in env_symbols.split(',') if s.strip()]
        else:
            symbols_config = config.get('trading.symbols', None)
            if symbols_config and isinstance(symbols_config, list):
                self._symbols = symbols_config
            else:
                symbol_str = config.get('trading.symbol', 'AI500_TOP5')
                if ',' in symbol_str:
                    self._symbols = [s.strip() for s in symbol_str.split(',') if s.strip()]
                else:
                    self._symbols = [symbol_str]

        # Normalize legacy AUTO2 -> AUTO1
        if 'AUTO2' in self._symbols:
            self._symbols = ['AUTO1' if s == 'AUTO2' else s for s in self._symbols]

        # 🔝 AUTO3 Dynamic Resolution (takes priority)
        self.use_auto3 = 'AUTO3' in self._symbols
        if self.use_auto3:
            self._symbols = [s for s in self._symbols if s not in ('AUTO3', 'AUTO1')]
            if not self._symbols:
                self._symbols = ['FETUSDT']
            log.info("🔝 AUTO3 mode enabled - Startup backtest will run")

        # AUTO1 Dynamic Selection
        self.use_auto1 = (not self.use_auto3) and ('AUTO1' in self._symbols)
        if self.use_auto1:
            self._symbols = [s for s in self._symbols if s != 'AUTO1']
            if not self._symbols:
                self._symbols = ['BTCUSDT']
            log.info("🎯 AUTO1 mode enabled - Symbol selector will run at startup")

        # 🤖 AI500 Dynamic Resolution
        if 'AI500_TOP5' in self._symbols:
            self._symbols.remove('AI500_TOP5')
            ai_top5 = self._resolve_ai500_symbols()
            self._symbols = list(set(self._symbols + ai_top5))
            self._symbols.sort()

        if set(self._symbols) != set(old_symbols):
            log.info(f"🔄 Trading symbols reloaded: {', '.join(self._symbols)}")
            global_state.add_log(f"[🔄 CONFIG] Symbols reloaded: {', '.join(self._symbols)}")
            # Update global state
            global_state.symbols = self._symbols
            # Initialize PredictAgent for any new symbols
            self._update_predict_agents()
            return True
        
        return False

    def update_ai500(self):
        new_top5 = self._resolve_ai500_symbols()

        # Update symbols list
        old_symbols = set(self._symbols)
        # Remove old AI coins and add new ones
        # Keep non-AI coins unchanged
        major_coins = {'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT'}
        non_ai_symbols = [s for s in self._symbols if s in major_coins]

        # Merge with new AI top5
        self._symbols = list(set(non_ai_symbols + new_top5))
        self._symbols.sort()

        # Update global state
        global_state.symbols = self._symbols

        # Log changes
        added = set(self._symbols) - old_symbols
        removed = old_symbols - set(self._symbols)
        if added or removed:
            log.info(f"📊 AI500 Updated - Added: {added}, Removed: {removed}")
            log.info(f"📋 Current symbols: {', '.join(self._symbols)}")
            for symbol in added:
                self.predict_add_callback(symbol, horizon='30m')
        else:
            log.info("✅ AI500 Updated - No changes in Top5")

    def run_symbol_selector(
        self,
        reason: str = "scheduled",
        check_for_startup_done: bool = False
    ) -> None:
        if not self.agent_config.symbol_selector_agent:
            return
        
        if check_for_startup_done and self.selector_startup_done:
            return
        
        now_ts = time.time()
        active_symbols = self.get_active_position_symbols()
        has_positions = bool(active_symbols)
        
        # Run symbol selector on schedule (every 10 minutes, skip if holding positions)
        # Note: symbol_manager.run_symbol_selector also has internal position check for safety
        if ((now_ts - self.selector_last_run) < self.selector_interval_sec
            or global_state.execution_mode != "Running"
            or has_positions):
            return
        
        if reason != "startup" and self.selector_last_run > 0 and (now_ts - self.selector_last_run) < self.selector_interval_sec:
            return

        if active_symbols:
            locked = [s for s in self._symbols if s in active_symbols]
            if not locked:
                locked = sorted(set(active_symbols))
            log.info(f"🔒 SymbolSelectorAgent skipped (active positions: {', '.join(locked)})")
            global_state.add_log(f"[🔒 SELECTOR] Skipped: active positions ({', '.join(locked)})")
            self.selector_last_run = now_ts
            if reason == "startup":
                self.selector_startup_done = True
            return

        selector_started = now_ts
        try:
            log.info(f"🎰 SymbolSelectorAgent ({reason}) running before analysis...")
            global_state.add_log(f"[🎰 SELECTOR] Symbol selection started ({reason})")
            account_equity = self.client.get_account_equity_estimate()
            if hasattr(self.agent_provider.symbol_selector_agent, 'account_equity') and account_equity:
                self.agent_provider.symbol_selector_agent.account_equity = account_equity
            if self.use_auto3:
                top_symbols = asyncio.run(
                    self.agent_provider.symbol_selector_agent.select_top3(force_refresh=False, account_equity=account_equity))
            else:
                top_symbols = asyncio.run(
                    self.agent_provider.symbol_selector_agent.select_auto1_recent_momentum(account_equity=account_equity)
                ) or []

            if top_symbols:
                self._symbols = top_symbols
                self.current_symbol = top_symbols[0]
                global_state.symbols = top_symbols
                selector_payload = {
                    "mode": "AUTO3" if self.use_auto3 else "AUTO1",
                    "symbols": list(top_symbols),
                    "symbol": self._current_symbol,
                    "direction": None,
                    "change_pct": None,
                    "volume_ratio": None,
                    "score": None,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                if not self.use_auto3:
                    auto1 = getattr(self.agent_provider.symbol_selector_agent, "last_auto1", {}) or {}
                    metrics = auto1.get("results", {}).get(self._current_symbol, {})
                    change_pct = metrics.get("change_pct")
                    if isinstance(change_pct, (int, float)):
                        if change_pct > 0:
                            selector_payload["direction"] = "UP"
                        elif change_pct < 0:
                            selector_payload["direction"] = "DOWN"
                        else:
                            selector_payload["direction"] = "FLAT"
                        selector_payload["change_pct"] = change_pct
                    volume_ratio = metrics.get("volume_ratio")
                    if isinstance(volume_ratio, (int, float)):
                        selector_payload["volume_ratio"] = volume_ratio
                    score = metrics.get("score")
                    if isinstance(score, (int, float)):
                        selector_payload["score"] = score
                    selector_payload["window_minutes"] = auto1.get("window_minutes")
                global_state.symbol_selector = selector_payload
                self._update_primary()

                self._update_predict_agents()

                if self.use_auto3:
                    self.agent_provider.symbol_selector_agent.start_auto_refresh()
                log.info(f"✅ SymbolSelectorAgent ready: {', '.join(top_symbols)}")
                global_state.add_log(f"[🎰 SELECTOR] Selected: {', '.join(top_symbols)}")
                global_state.add_agent_message(
                    "symbol_selector",
                    f"Mode: {selector_payload.get('mode', 'AUTO')} | Symbols: {', '.join(top_symbols)}",
                    level="info"
                )
            else:
                log.warning("⚠️ SymbolSelectorAgent returned empty selection")
                global_state.add_log("[🎰 SELECTOR] Empty selection (fallback to configured symbols)")
        except Exception as e:
            log.error(f"❌ SymbolSelectorAgent failed: {e}")
            global_state.add_log(f"[🎰 SELECTOR] Failed: {e}")
        finally:
            self.selector_last_run = selector_started
            if reason == "startup":
                self.selector_startup_done = True

    def get_active_position_symbols(self) -> List[str]:
        """Return symbols with active positions (test + live)."""
        if self.test_mode:
            active = []
            for symbol, pos in (global_state.virtual_positions or {}).items():
                try:
                    qty = float(pos.get('quantity', 0) or 0)
                except (TypeError, ValueError):
                    qty = 0
                if abs(qty) > 0:
                    active.append(symbol)
            return active

        try:
            account = self.client.get_futures_account()
            positions = account.get('positions', []) or []
            active = []
            for pos in positions:
                amt = pos.get('positionAmt')
                if amt is None:
                    amt = pos.get('position_amt', 0)
                try:
                    if abs(float(amt)) > 0:
                        symbol = pos.get('symbol')
                        if symbol:
                            active.append(symbol)
                except (TypeError, ValueError):
                    continue
            return active
        except Exception as e:
            log.warning(f"Failed to fetch active positions: {e}")
            return []

    def _resolve_ai500_symbols(self):
        """Dynamic resolution of AI500_TOP5 tag"""
        # AI Candidates List (30+ Major AI/Data/Compute Coins)
        AI_CANDIDATES = [
            "FETUSDT", "RENDERUSDT", "TAOUSDT", "NEARUSDT", "GRTUSDT", 
            "WLDUSDT", "ARKMUSDT", "LPTUSDT", "THETAUSDT", "ROSEUSDT",
            # Removed merged/renamed: AGIX, OCEAN, RNDR (now FET/RENDER)
            "PHBUSDT", "CTXCUSDT", "NMRUSDT", "RLCUSDT", "GLMUSDT",
            "IQUSDT", "MDTUSDT", "AIUSDT", "NFPUSDT", "XAIUSDT",
            "JASMYUSDT", "ICPUSDT", "FILUSDT", "VETUSDT", "LINKUSDT",
            "ACTUSDT", "GOATUSDT", "TURBOUSDT", "PNUTUSDT" 
        ]

        try:
            print("🤖 AI500 Dynamic Selection: Fetching 24h Volume Data...")
            # Use temporary client to fetch tickers
            temp_client = BinanceClient()
            tickers = temp_client.get_all_tickers()

            # Filter and Sort
            ai_stats = []
            for t in tickers:
                if t['symbol'] in AI_CANDIDATES:
                    try:
                        quote_vol = float(t['quoteVolume'])
                        ai_stats.append((t['symbol'], quote_vol))
                    except (KeyError, ValueError, TypeError) as e:
                        log.debug(f"Skipped {t.get('symbol', 'unknown')}: {e}")

            # Sort by Volume desc
            ai_stats.sort(key=lambda x: x[1], reverse=True)

            # Take Top 5
            top_5 = [x[0] for x in ai_stats[:5]]

            print(f"✅ AI500 Top 5 Selected (by Vol): {', '.join(top_5)}")
            return top_5
        except Exception as e:
            log.error(f"Failed to resolve AI500 symbols: {e}")
            # Fallback to defaults (Top 5)
            return ["FETUSDT", "RENDERUSDT", "TAOUSDT", "NEARUSDT", "GRTUSDT"]

    def _update_predict_agents(self, horizon='30m'):
        for symbol in self._symbols:
            self.agent_provider.predict_agents_provider.add_agent_for_symbol(symbol, horizon=horizon)

    def _update_primary(self):
        if self._primary_symbol not in self._symbols:
            self._primary_symbol = self._current_symbol
            log.info(f"🔄 Primary symbol updated to {self._primary_symbol} (selector)")
