import os
import time
import asyncio
import threading

from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Optional, List, Any, Tuple

from dataclasses import asdict

from src.config import Config

from src.strategy.llm_engine import StrategyEngine
from src.api.binance_client import BinanceClient
from src.execution.engine import ExecutionEngine
from src.risk.manager import RiskManager
from src.utils.data_saver import DataSaver
from src.data.processor import MarketDataProcessor  # ✅ Corrected Import
from src.exchanges import AccountManager, ExchangeAccount, ExchangeType  # ✅ Multi-Account Support
from src.agents.contracts import SuggestedTrade
from src.agents.predict_result import PredictResult  # ✅ PredictResult Import
from src.utils.semantic_converter import SemanticConverter  # ✅ Global Import
from src.agents.symbol_selector_agent import get_selector  # 🔝 AUTO3 Support
from src.agents.runtime_events import emit_global_runtime_event, emit_cycle_pipeline_end
from src.utils.task_util import run_task_with_timeout
from src.agents.regime_detector_agent import RegimeDetector  # ✅ Market Regime Detection

from src.utils.logger import log
from src.utils.trade_logger import trade_logger
from src.server.state import global_state

from .stage_result import StageResult
from .cycle_context import CycleContext
from .symbol_manager import SymbolManager
from .ai500_updater import Ai500Updater  # ✅ AI500 Dynamic Updater
from .predict_agents_provider import PredictAgentsProvider

from src.runner.parallel_analysis_runner import ParallelAnalysesRunner

from src.utils.action_protocol import (
    normalize_action,
    is_open_action,
    is_close_action,
    is_passive_action,
)

from src.agents import (
    DataSyncAgent,
    DecisionCoreAgent,
    RiskAuditAgent,
    PositionInfo,
    MultiPeriodParserAgent,
    AgentRegistry
)

from src.agents.reflection import (
    ReflectionAgentLLM,
    ReflectionAgentNoLLM
)

class MultiAgentTradingBot:
    """
    多Agent交易机器人 (重构版)
    
    工作流程:
    1. DataSyncAgent: 异步采集5m/15m/1h数据
    2. QuantAnalystAgent: 生成量化信号（趋势+震荡）
    3. DecisionCoreAgent: 加权投票决策
    4. RiskAuditAgent: 风控审计拦截
    5. ExecutionEngine: 执行交易
    """
    
    def __init__(
        self,
        max_position_size: float = 100.0,
        leverage: int = 1,
        stop_loss_pct: float = 1.0,
        take_profit_pct: float = 2.0,
        test_mode: bool = False,
        kline_limit: int = 300
    ):
        """
        初始化多Agent交易机器人
        
        Args:
            max_position_size: 最大单笔金额 (USDT)
            leverage: 杠杆倍数
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
            test_mode: 测试模式（不执行真实交易）
        """
        print("\n" + "="*80)
        print(f"🤖 AI Trader - DeepSeek LLM Decision Mode")
        print("="*80)
        
        self.config = Config()
        self.client = BinanceClient(test_mode)

        self.test_mode = test_mode
        global_state.is_test_mode = test_mode  # Set test mode in global state

        # Cycle logging (DB)
        self._cycle_logger = None
        self._last_cycle_realized_pnl = 0.0
        
        # 交易参数
        self.max_position_size = max_position_size
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.kline_limit = int(kline_limit) if kline_limit and kline_limit > 0 else 300
        
        # 初始化客户端
        self.risk_manager = RiskManager()
        self.execution_engine = ExecutionEngine(self.client, self.risk_manager)
        self.saver = DataSaver() # ✅ 初始化 Multi-Agent 数据保存器
        
        # 🧹 启动时清除历史实盘数据，只保留当前周期
        self.saver.clear_live_data()

        # 💰 Persistent Virtual Account (Test Mode)
        if self.test_mode:
            saved_va = self.saver.load_virtual_account()
            if saved_va:
                log.info("💰 Found persistent virtual account. Resetting to initial balance for new session.")
            global_state.virtual_balance = global_state.virtual_initial_balance
            global_state.virtual_positions = {}
            self._save_virtual_state()
        global_state.saver = self.saver # ✅ 将 saver 共享到全局状态，供各 Agent 使用
        
        
        # ✅ 初始化多账户管理器
        self.account_manager = AccountManager()
        self._init_accounts()
        # Initialize mtime for .env tracking (skip if not exists, e.g. Railway)
        self._env_mtime = 0
        self._env_path = os.path.join(os.path.dirname(__file__), '.env')
        self._env_exists = os.path.exists(self._env_path)  # 🔧 Railway fix
        
        # 初始化共享 Agent (与币种无关)
        print("\n🚀 Initializing agents...")
        
        # 🆕 Load Agent Configuration
        from src.agents.agent_config import AgentConfig
        agents_config = self.config.get('agents', {})
        self.agent_config = AgentConfig.from_dict({'agents': agents_config})
        print(f"  📋 Agent Config: {self.agent_config}")
        global_state.agent_config = self.agent_config.get_enabled_agents()
        self._last_agent_config = dict(global_state.agent_config)
        self.agent_registry = AgentRegistry(self.agent_config)
        self.agent_registry.register_class('regime_detector_agent', RegimeDetector)
        
        # Symbol manager and ai500 updater
        self.symbol_manager = SymbolManager(
            self.config,
            self.agent_config,
            self.client,
            self._predict_add_callback,
            test_mode)
        self.ai500_updater = Ai500Updater(self.symbol_manager)  # ✅ AI500 Updater
        
        # Core Agents (always enabled)
        self.data_sync_agent = DataSyncAgent(self.client)
        self.decision_core = DecisionCoreAgent()
        self.multi_period_agent = MultiPeriodParserAgent()
        self.risk_audit = RiskAuditAgent(
            max_leverage=10.0,
            max_position_pct=0.3,
            min_stop_loss_pct=0.005,
            max_stop_loss_pct=0.05
        )
        print("[DEBUG] Creating MarketDataProcessor...")
        self.processor = MarketDataProcessor()  # ✅ 初始化数据处理器
        print("[DEBUG] MarketDataProcessor created")
        print("[DEBUG] Creating TechnicalFeatureEngineer...")
        print("[DEBUG] TechnicalFeatureEngineer created")
        
        # 🆕 Optional Agent: RegimeDetectorAgent
        self.regime_detector = None
        if self.agent_config.regime_detector_agent:
            self.regime_detector = self.agent_registry.get('regime_detector_agent')
            if self.regime_detector is not None:
                print("  ✅ RegimeDetectorAgent ready")
            else:
                print("  ⚠️ RegimeDetectorAgent init failed")
        else:
            print("  ⏭️ RegimeDetectorAgent disabled")
        
        # 🆕 Optional Agent: PredictAgent (per symbol)
        self.predict_agents_provider = None
        if self.agent_config.predict_agent:
            print("[DEBUG] Creating PredictAgents...")
            self.predict_agents_provider = PredictAgentsProvider(self.client, self.symbol_manager, self.agent_config)
            print(f"  ✅ PredictAgent ready ({len(self.symbol_manager.symbols)} symbols)")
        else:
            print("  ⏭️ PredictAgent disabled")
        
        print("  ✅ DataSyncAgent ready")
        print("  ✅ QuantAnalystAgent ready")
        print("  ✅ RiskAuditAgent ready")
        
        # 🧠 DeepSeek 决策引擎
        print("[DEBUG] Creating StrategyEngine...")
        self.strategy_engine = StrategyEngine()
        print("[DEBUG] StrategyEngine created")
        if self.strategy_engine.is_ready:
            print("  ✅ DeepSeek StrategyEngine ready")
        else:
            print("  ⚠️ DeepSeek StrategyEngine not ready (Awaiting API Key)")
            
        # 🆕 Optional Agent: ReflectionAgent
        self.reflection_agent = None
        if self.agent_config.reflection_agent_llm or self.agent_config.reflection_agent_local:
            if self.agent_config.reflection_agent_llm:
                self.reflection_agent = ReflectionAgentLLM(self.agent_config)
            else:
                self.reflection_agent = ReflectionAgentNoLLM()
        else:
            print("  ⏭️ ReflectionAgent disabled")

        self.parallel_analyses_runner = ParallelAnalysesRunner(
            self.config,
            self.agent_config,
            self.symbol_manager,
            self.predict_agents_provider,
            self.reflection_agent,
            self.saver
        )
        
        print(f"\n⚙️  Trading Config:")
        print(f"  - Symbols: {', '.join(self.symbol_manager.symbols)}")
        print(f"  - Max Position: ${self.max_position_size:.2f} USDT")
        print(f"  - Leverage: {self.leverage}x")
        print(f"  - Stop Loss: {self.stop_loss_pct}%")
        print(f"  - Take Profit: {self.take_profit_pct}%")
        print(f"  - Kline Limit: {self.kline_limit}")
        print(f"  - Test Mode: {'✅ Yes' if self.test_mode else '❌ No'}")
        
        # ✅ Load initial trade history
        recent_trades = self.saver.get_recent_trades(limit=20)
        global_state.trade_history = recent_trades
        print(f"  📜 Loaded {len(recent_trades)} historical trades")
        
        # 🆕 Initialize Chatroom with a boot message
        global_state.add_agent_message(
            "decision_core", 
            "**System initialized.** All agents are online and ready for parallel execution. Standing by for market data...", 
            level="success"
        )
        
        self._sync_open_positions_to_trade_history()
        # [NEW] Initialize LLM metadata
        self._update_llm_metadata()

    def _update_llm_metadata(self):
        """Collect current LLM provider/model and agent system prompts for UI display"""
        try:
            from src.agents.trend_agent import TrendAgentLLM
            from src.agents.setup_agent import SetupAgentLLM
            from src.agents.trigger_agent import TriggerAgentLLM
            
            # 1. Collect LLM Engine info (Decision Core)
            llm_info = {
                "provider": getattr(self.strategy_engine, 'provider', 'None'),
                "model": getattr(self.strategy_engine, 'model', 'None')
            }
            global_state.llm_info = llm_info
            
            # 2. Collect System Prompts
            prompts = {}
            
            # Decision Core Prompt
            prompts["decision_core"] = self.strategy_engine._build_system_prompt()
            
            # Trend Agent
            try:
                trend_agent = TrendAgentLLM()
                prompts["trend_agent"] = trend_agent._get_system_prompt()
            except Exception: pass
            
            # Setup Agent
            try:
                setup_agent = SetupAgentLLM()
                prompts["setup_agent"] = setup_agent._get_system_prompt()
            except Exception: pass
            
            # Trigger Agent
            try:
                trigger_agent = TriggerAgentLLM()
                prompts["trigger_agent"] = trigger_agent._get_system_prompt()
            except Exception: pass
            
            # Reflection Agent
            if self.reflection_agent:
                prompt = self.reflection_agent.build_system_prompt()
                if prompt:
                    prompts["reflection_agent"] = prompt
            
            global_state.agent_prompts = prompts
            log.info(f"📊 LLM metadata updated: {llm_info['provider']} ({llm_info['model']}), {len(prompts)} prompts collected")
            
        except Exception as e:
            log.error(f"Failed to update LLM metadata: {e}")

    def _reload_symbols(self):
        """Reload trading symbols from environment/config without restart"""
        # Note: On Railway, os.environ is already updated by config_manager.
        # On local, load_dotenv refreshes from .env file.
        if self._env_exists:
            load_dotenv(override=True)
        # Reload full config to pick up updated LLM provider/keys and agents
        try:
            self.config._load_config()
        except Exception as e:
            log.warning(f"⚠️ Failed to reload config: {e}")
        # Reload LLM engine to pick up new provider/keys
        try:
            if hasattr(self, "strategy_engine"):
                self.strategy_engine.reload_config()
                self._update_llm_metadata()
        except Exception as e:
            log.warning(f"⚠️ Failed to reload LLM engine: {e}")

        need_reload = self.symbol_manager.reload_symbols(self.config)
            
        if need_reload:
            # Refresh LLM metadata in case config changed
            self._update_llm_metadata()
    
    def _predict_add_callback(self, symbol: str, horizon: str = '30m'):
        self.predict_agents_provider.add_agent_for_symbol(symbol, horizon=horizon)

    def _sync_open_positions_to_trade_history(self) -> None:
        """Ensure open positions appear in trade history for the UI."""
        def has_open_record(symbol: str) -> bool:
            for trade in global_state.trade_history:
                if trade.get('symbol') != symbol:
                    continue
                exit_price = trade.get('exit_price')
                if exit_price in (None, "", "N/A"):
                    return True
                try:
                    if float(exit_price) == 0:
                        return True
                except (TypeError, ValueError):
                    return True
            return False

        added = []
        if self.test_mode:
            for symbol, pos in (global_state.virtual_positions or {}).items():
                try:
                    qty = float(pos.get('quantity', 0) or 0)
                except (TypeError, ValueError):
                    qty = 0
                if abs(qty) == 0 or has_open_record(symbol):
                    continue
                side = (pos.get('side') or '').upper()
                action = f"OPEN_{side}" if side else "OPEN"
                entry_price = float(pos.get('entry_price', 0) or 0)
                trade_record = {
                    'open_cycle': 0,
                    'close_cycle': 0,
                    'timestamp': pos.get('entry_time') or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'action': action,
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'quantity': qty,
                    'cost': entry_price * qty,
                    'exit_price': 0,
                    'pnl': 0.0,
                    'confidence': 'N/A',
                    'status': 'OPEN (SYNC)',
                    'cycle': global_state.current_cycle_id or 'N/A'
                }
                global_state.trade_history.insert(0, trade_record)
                added.append(symbol)
        else:
            try:
                account = self.client.get_futures_account()
                positions = account.get('positions', []) or []
            except Exception as e:
                log.warning(f"Failed to sync live positions: {e}")
                positions = []
            for pos in positions:
                amt = pos.get('positionAmt')
                if amt is None:
                    amt = pos.get('position_amt', 0)
                try:
                    amt_val = float(amt)
                except (TypeError, ValueError):
                    continue
                if abs(amt_val) == 0:
                    continue
                symbol = pos.get('symbol')
                if not symbol or has_open_record(symbol):
                    continue
                side = "LONG" if amt_val > 0 else "SHORT"
                entry_price = pos.get('entryPrice') or pos.get('entry_price') or 0
                try:
                    entry_price = float(entry_price)
                except (TypeError, ValueError):
                    entry_price = 0.0
                qty = abs(amt_val)
                trade_record = {
                    'open_cycle': 0,
                    'close_cycle': 0,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'action': f"OPEN_{side}",
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'quantity': qty,
                    'cost': entry_price * qty,
                    'exit_price': 0,
                    'pnl': 0.0,
                    'confidence': 'N/A',
                    'status': 'OPEN (SYNC)',
                    'cycle': global_state.current_cycle_id or 'N/A'
                }
                global_state.trade_history.insert(0, trade_record)
                added.append(symbol)

        if added:
            if len(global_state.trade_history) > 50:
                global_state.trade_history = global_state.trade_history[:50]
            log.info(f"📜 Synced open positions into trade history: {', '.join(added)}")
            global_state.add_log(f"[📜 SYSTEM] Synced open positions: {', '.join(added)}")

    def _apply_agent_config(self, agents: Dict[str, bool]) -> None:
        """Apply runtime agent config and sync optional agent instances."""
        from src.agents.agent_config import AgentConfig

        self.agent_config = AgentConfig.from_dict({'agents': agents})
        self.agent_registry.config = self.agent_config
        normalized_agents = self.agent_config.get_enabled_agents()
        self._last_agent_config = dict(normalized_agents)
        global_state.agent_config = normalized_agents

        # Optional Agent: RegimeDetector
        if self.agent_config.regime_detector_agent:
            if self.regime_detector is None:
                self.regime_detector = self.agent_registry.get('regime_detector_agent')
                if self.regime_detector is not None:
                    log.info("✅ RegimeDetectorAgent enabled (runtime)")
                else:
                    log.warning("⚠️ RegimeDetectorAgent enable failed (runtime)")
        else:
            self.regime_detector = None

        # Optional Agent: PredictAgent (per symbol)
        if self.agent_config.predict_agent:
            self.predict_agents_provider.reload()
        else:
            self.predict_agents_provider = None

        # Optional Agent: ReflectionAgent
        if self.agent_config.reflection_agent_llm or self.agent_config.reflection_agent_local:
            if self.agent_config.reflection_agent_llm:
                if not isinstance(self.reflection_agent, ReflectionAgentLLM):
                    self.reflection_agent = ReflectionAgentLLM(self.agent_config)
            else:
                if not isinstance(self.reflection_agent, ReflectionAgentNoLLM):
                    self.reflection_agent = ReflectionAgentNoLLM()
        else:
            self.reflection_agent = None

    def _is_llm_enabled(self) -> bool:
        """Return True if LLM-driven agents are enabled by runtime config."""
        return bool(
            self.agent_config.trend_agent_llm
            or self.agent_config.setup_agent_llm
            or self.agent_config.trigger_agent_llm
            or self.agent_config.reflection_agent_llm
        )

    def _attach_agent_ui_fields(self, decision_dict: Dict) -> None:
        """Attach optional agent fields used by the dashboard."""
        four_layer = getattr(global_state, 'four_layer_result', {}) or {}
        ai_check = four_layer.get('ai_check', {}) if isinstance(four_layer, dict) else {}
        if ai_check:
            decision_dict['ai_filter_passed'] = not ai_check.get('ai_veto', False)
            decision_dict['ai_filter_reason'] = ai_check.get('reason')
            decision_dict['ai_filter_signal'] = ai_check.get('ai_signal')
            decision_dict['ai_filter_confidence'] = ai_check.get('ai_confidence')

        decision_dict['trigger_pattern'] = four_layer.get('trigger_pattern')
        decision_dict['trigger_rvol'] = four_layer.get('trigger_rvol')

        position = decision_dict.get('position')
        if isinstance(position, dict) and position.get('location'):
            decision_dict['position_zone'] = position.get('location')

        semantic_analyses = getattr(global_state, 'semantic_analyses', None)
        if semantic_analyses:
            decision_dict['semantic_analyses'] = semantic_analyses

        reflection_text = getattr(global_state, 'last_reflection_text', None)
        reflection_count = getattr(global_state, 'reflection_count', 0)
        trades = getattr(global_state, 'trade_history', []) or []
        pnl_values = []
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            pnl = trade.get('pnl', trade.get('realized_pnl'))
            if pnl is None:
                continue
            try:
                pnl_values.append(float(pnl))
            except (TypeError, ValueError):
                continue
        win_rate = None
        if pnl_values:
            wins = sum(1 for v in pnl_values if v > 0)
            win_rate = (wins / len(pnl_values)) * 100
        if reflection_text or reflection_count or pnl_values:
            decision_dict['reflection'] = {
                'count': reflection_count,
                'text': reflection_text,
                'trades': len(pnl_values),
                'win_rate': win_rate
            }

        indicator_snapshot = getattr(global_state, 'indicator_snapshot', None)
        if indicator_snapshot:
            snapshot = indicator_snapshot
            if isinstance(indicator_snapshot, dict) and 'ema_status' not in indicator_snapshot:
                symbol = decision_dict.get('symbol')
                snapshot = indicator_snapshot.get(symbol) if symbol else None
            if snapshot:
                decision_dict['indicator_snapshot'] = snapshot

    def _capture_indicator_snapshot(self, processed_dfs: Dict[str, "pd.DataFrame"], timeframe: str = '15m') -> Optional[Dict]:
        """Capture lightweight indicator snapshot for UI."""
        df = processed_dfs.get(timeframe)
        if df is None or df.empty:
            return None

        latest = df.iloc[-1]

        def _safe_float(value):
            try:
                val = float(value)
            except (TypeError, ValueError):
                return None
            if val != val or val in (float('inf'), float('-inf')):
                return None
            return val

        close = _safe_float(latest.get('close'))
        ema20 = _safe_float(latest.get('ema_20'))
        ema60 = _safe_float(latest.get('ema_60'))
        rsi = _safe_float(latest.get('rsi'))
        macd_diff = _safe_float(latest.get('macd_diff'))
        bb_upper = _safe_float(latest.get('bb_upper'))
        bb_lower = _safe_float(latest.get('bb_lower'))
        bb_middle = _safe_float(latest.get('bb_middle'))

        ema_status = None
        if close is not None and ema20 is not None and ema60 is not None:
            if close > ema20 > ema60:
                ema_status = 'bullish'
            elif close < ema20 < ema60:
                ema_status = 'bearish'
            elif ema20 > ema60:
                ema_status = 'bullish_bias'
            elif ema20 < ema60:
                ema_status = 'bearish_bias'
            else:
                ema_status = 'mixed'

        bb_position = None
        if close is not None and bb_upper is not None and bb_lower is not None and bb_middle is not None:
            if close > bb_upper:
                bb_position = 'upper'
            elif close < bb_lower:
                bb_position = 'lower'
            else:
                bb_position = 'middle'

        return {
            'timeframe': timeframe,
            'ema_status': ema_status,
            'rsi': rsi,
            'macd_diff': macd_diff,
            'bb_position': bb_position
        }

    def _detect_fast_trend_signal(
        self,
        df_5m: Optional["pd.DataFrame"],
        window_minutes: int = 30,
        threshold_pct: float = 0.8,
        volume_ratio_threshold: float = 1.2
    ) -> Optional[Dict]:
        """Detect short-term momentum over the last 30m and return a fast entry signal."""
        if df_5m is None or df_5m.empty:
            return None
        if 'close' not in df_5m.columns or 'volume' not in df_5m.columns:
            return None

        window_bars = max(2, int(window_minutes / 5))
        if len(df_5m) < window_bars * 2:
            return None

        recent = df_5m.iloc[-window_bars:]
        previous = df_5m.iloc[-window_bars * 2:-window_bars]
        first_close = float(recent['close'].iloc[0])
        last_close = float(recent['close'].iloc[-1])
        if first_close <= 0:
            return None

        change_pct = ((last_close - first_close) / first_close) * 100.0
        recent_volume = float(recent['volume'].sum())
        prev_volume = float(previous['volume'].sum()) if not previous.empty else 0.0
        volume_ratio = (recent_volume / prev_volume) if prev_volume > 0 else 1.0

        if change_pct >= threshold_pct and volume_ratio >= volume_ratio_threshold:
            action = 'open_long'
        elif change_pct <= -threshold_pct and volume_ratio >= volume_ratio_threshold:
            action = 'open_short'
        else:
            return None

        change_over = max(0.0, abs(change_pct) - threshold_pct)
        change_boost = min(15.0, change_over * 5.0)
        vol_over = max(0.0, volume_ratio - volume_ratio_threshold)
        vol_boost = min(10.0, vol_over * 10.0)
        confidence = min(92.0, 70.0 + change_boost + vol_boost)

        return {
            'action': action,
            'change_pct': change_pct,
            'volume_ratio': volume_ratio,
            'confidence': confidence,
            'window_minutes': window_minutes
        }

    def _get_agent_timeout(self, key: str, default_seconds: float) -> float:
        """
        Resolve per-agent timeout in seconds from config.
        Uses AgentConfig runtime policy and falls back to legacy config path.
        """
        cfg = getattr(self, 'agent_config', None)
        if cfg is not None and hasattr(cfg, 'get_timeout'):
            return cfg.get_timeout(key, default_seconds)
        raw = self.config.get(f'agents.timeouts.{key}', default_seconds)
        try:
            val = float(raw)
            return val if val > 0 else float(default_seconds)
        except (TypeError, ValueError):
            return float(default_seconds)

    async def _run_semantic_analysis(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        current_price: float,
        trend_1h: str,
        four_layer_result: Dict[str, Any],
        processed_dfs: Dict[str, "pd.DataFrame"]
    ) -> Dict[str, Any]:
        """
        Run optional trend/setup/trigger semantic agents and summarize their outputs.
        """
        use_trend_llm = self.agent_config.trend_agent_llm
        use_trend_local = self.agent_config.trend_agent_local
        use_setup_llm = self.agent_config.setup_agent_llm
        use_setup_local = self.agent_config.setup_agent_local
        use_trigger_llm = self.agent_config.trigger_agent_llm
        use_trigger_local = self.agent_config.trigger_agent_local
        use_trend = use_trend_llm or use_trend_local
        use_setup = use_setup_llm or use_setup_local
        use_trigger = use_trigger_llm or use_trigger_local

        if use_trend and use_trend_llm and use_trend_local:
            log.info("⚠️ Both TrendAgentLLM and TrendAgent enabled; using LLM version only")
        if use_setup and use_setup_llm and use_setup_local:
            log.info("⚠️ Both SetupAgentLLM and SetupAgent enabled; using LLM version only")
        if use_trigger and use_trigger_llm and use_trigger_local:
            log.info("⚠️ Both TriggerAgentLLM and TriggerAgent enabled; using LLM version only")

        if not (use_trend or use_setup or use_trigger):
            global_state.semantic_analyses = {}
            return {}

        if not (hasattr(self, '_headless_mode') and self._headless_mode):
            print("[Step 2.5/5] 🤖 Multi-Agent Semantic Analysis...")

        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="semantic_agents",
            phase="start",
            symbol=self.symbol_manager.current_symbol,
            cycle_id=cycle_id
        )

        try:
            trend_data = {
                'symbol': self.symbol_manager.current_symbol,
                'close_1h': four_layer_result.get('close_1h', current_price),
                'ema20_1h': four_layer_result.get('ema20_1h', current_price),
                'ema60_1h': four_layer_result.get('ema60_1h', current_price),
                'oi_change': four_layer_result.get('oi_change', 0),
                'adx': four_layer_result.get('adx', 20),
                'regime': four_layer_result.get('regime', 'unknown')
            }

            setup_data = {
                'symbol': self.symbol_manager.current_symbol,
                'close_15m': processed_dfs['15m']['close'].iloc[-1] if len(processed_dfs['15m']) > 0 else current_price,
                'kdj_j': four_layer_result.get('kdj_j', 50),
                'kdj_k': processed_dfs['15m']['kdj_k'].iloc[-1] if 'kdj_k' in processed_dfs['15m'].columns else 50,
                'bb_upper': processed_dfs['15m']['bb_upper'].iloc[-1] if 'bb_upper' in processed_dfs['15m'].columns else current_price * 1.02,
                'bb_middle': processed_dfs['15m']['bb_middle'].iloc[-1] if 'bb_middle' in processed_dfs['15m'].columns else current_price,
                'bb_lower': processed_dfs['15m']['bb_lower'].iloc[-1] if 'bb_lower' in processed_dfs['15m'].columns else current_price * 0.98,
                'trend_direction': trend_1h,
                'macd_diff': processed_dfs['15m']['macd_diff'].iloc[-1] if 'macd_diff' in processed_dfs['15m'].columns else 0
            }

            trigger_data = {
                'symbol': self.symbol_manager.current_symbol,
                'pattern': four_layer_result.get('trigger_pattern'),
                'rvol': four_layer_result.get('trigger_rvol', 1.0),
                'trend_direction': four_layer_result.get('final_action', 'neutral')
            }

            tasks = {}
            loop = asyncio.get_running_loop()
            if use_trend:
                if use_trend_llm:
                    from src.agents.trend_agent import TrendAgentLLM
                    if not hasattr(self, '_trend_agent_llm'):
                        self._trend_agent_llm = TrendAgentLLM()
                    trend_agent = self._trend_agent_llm
                else:
                    from src.agents.trend_agent import TrendAgent
                    if not hasattr(self, '_trend_agent_local'):
                        self._trend_agent_local = TrendAgent()
                    trend_agent = self._trend_agent_local
                tasks['trend'] = loop.run_in_executor(None, trend_agent.analyze, trend_data)

            if use_setup:
                if use_setup_llm:
                    from src.agents.setup_agent import SetupAgentLLM
                    if not hasattr(self, '_setup_agent_llm'):
                        self._setup_agent_llm = SetupAgentLLM()
                    setup_agent = self._setup_agent_llm
                else:
                    from src.agents.setup_agent import SetupAgent
                    if not hasattr(self, '_setup_agent_local'):
                        self._setup_agent_local = SetupAgent()
                    setup_agent = self._setup_agent_local
                tasks['setup'] = loop.run_in_executor(None, setup_agent.analyze, setup_data)

            if use_trigger:
                if use_trigger_llm:
                    from src.agents.trigger_agent import TriggerAgentLLM
                    if not hasattr(self, '_trigger_agent_llm'):
                        self._trigger_agent_llm = TriggerAgentLLM()
                    trigger_agent = self._trigger_agent_llm
                else:
                    from src.agents.trigger_agent import TriggerAgent
                    if not hasattr(self, '_trigger_agent_local'):
                        self._trigger_agent_local = TriggerAgent()
                    trigger_agent = self._trigger_agent_local
                tasks['trigger'] = loop.run_in_executor(None, trigger_agent.analyze, trigger_data)

            analyses: Dict[str, Any] = {}
            if tasks:
                semantic_timeout = self._get_agent_timeout('semantic_agent', 35.0)
                wrapped_tasks = {
                    key: run_task_with_timeout(
                        run_id=run_id,
                        cycle_id=cycle_id,
                        agent_name=f"{key}_agent",
                        timeout_seconds=semantic_timeout,
                        task_factory=(lambda fut=fut: fut),
                        symbol=self.symbol_manager.current_symbol,
                        fallback=None
                    )
                    for key, fut in tasks.items()
                }
                results = await asyncio.gather(*wrapped_tasks.values())
                analyses = {
                    key: val
                    for key, val in zip(wrapped_tasks.keys(), results)
                    if val is not None
                }

            global_state.semantic_analyses = analyses
            if analyses:
                trend_mark = '✓' if analyses.get('trend') else '○'
                setup_mark = '✓' if analyses.get('setup') else '○'
                trigger_mark = '✓' if analyses.get('trigger') else '○'
                global_state.add_log(f"[⚖️ CRITIC] 4-Layer Analysis: Trend={trend_mark} | Setup={setup_mark} | Trigger={trigger_mark}")

                trend_result = analyses.get('trend')
                if isinstance(trend_result, dict):
                    meta = trend_result.get('metadata', {}) or {}
                    summary = (
                        f"Stance: {trend_result.get('stance', 'UNKNOWN')} | "
                        f"Strength: {meta.get('strength', 'N/A')} | "
                        f"ADX: {meta.get('adx', 'N/A')} | "
                        f"OI Fuel: {meta.get('oi_fuel', 'N/A')}"
                    )
                    global_state.add_agent_message("trend_agent", summary, level="info")

                setup_result = analyses.get('setup')
                if isinstance(setup_result, dict):
                    meta = setup_result.get('metadata', {}) or {}
                    summary = (
                        f"Stance: {setup_result.get('stance', 'UNKNOWN')} | "
                        f"Zone: {meta.get('zone', 'N/A')} | "
                        f"KDJ: {meta.get('kdj_j', 'N/A')} | "
                        f"MACD: {meta.get('macd_signal', 'N/A')}"
                    )
                    global_state.add_agent_message("setup_agent", summary, level="info")

                trigger_result = analyses.get('trigger')
                if isinstance(trigger_result, dict):
                    meta = trigger_result.get('metadata', {}) or {}
                    summary = (
                        f"Stance: {trigger_result.get('stance', 'UNKNOWN')} | "
                        f"Pattern: {meta.get('pattern', 'NONE')} | "
                        f"RVOL: {meta.get('rvol', 'N/A')}x"
                    )
                    global_state.add_agent_message("trigger_agent", summary, level="info")

            emit_global_runtime_event(
                run_id=run_id,
                stream="lifecycle",
                agent="semantic_agents",
                phase="end",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"count": len(analyses)}
            )
            return analyses

        except Exception as e:
            log.error(f"❌ Multi-Agent analysis failed: {e}")
            fallback = {
                'trend': f"Trend analysis unavailable: {e}",
                'setup': f"Setup analysis unavailable: {e}",
                'trigger': f"Trigger analysis unavailable: {e}"
            }
            global_state.semantic_analyses = fallback
            emit_global_runtime_event(
                run_id=run_id,
                stream="error",
                agent="semantic_agents",
                phase="error",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"error": str(e)}
            )
            return fallback

    async def _run_decision_stage(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        processed_dfs: Dict[str, "pd.DataFrame"],
        quant_analysis: Dict[str, Any],
        predict_result: PredictResult,
        reflection_text: Optional[str],
        current_price: float,
        current_position_info: Optional[Dict[str, Any]],
        regime_result: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], str, Optional[Dict[str, Any]], Any, Dict[str, Any]]:
        """
        Build final decision payload (forced-exit/fast/LLM/rule) and convert to VoteResult.
        """
        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="decision_router",
            phase="start",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol
        )

        selected_agent_outputs = self._collect_selected_agent_outputs(
            predict_result=predict_result,
            reflection_text=reflection_text
        )
        if isinstance(quant_analysis, dict):
            quant_analysis['agent_outputs'] = selected_agent_outputs

        market_data = {
            'df_5m': processed_dfs['5m'],
            'df_15m': processed_dfs['15m'],
            'df_1h': processed_dfs['1h'],
            'current_price': current_price
        }
        regime_info = quant_analysis.get('regime', {})

        fast_signal = None
        decision_source = 'llm'
        forced_exit = self._check_forced_exit(current_position_info)
        llm_enabled = self._is_llm_enabled()
        if llm_enabled and getattr(self.strategy_engine, 'disable_llm', False):
            llm_enabled = bool(self.strategy_engine.reload_config())

        if forced_exit:
            decision_source = 'forced_exit'
            decision_payload = forced_exit
            global_state.add_log(f"[🧯 FORCED EXIT] {forced_exit.get('reasoning', 'Forced close')}")
            try:
                conf_val = float(decision_payload.get('confidence', 0) or 0)
            except (TypeError, ValueError):
                conf_val = 0.0
            global_state.add_agent_message(
                "decision_core",
                f"Action: {decision_payload.get('action', '').upper()} | Conf: {conf_val:.1f}% | Reason: {decision_payload.get('reasoning', '')} | Source: FORCED",
                level="warning"
            )
        else:
            fast_signal = self._detect_fast_trend_signal(processed_dfs.get('5m'))

        if fast_signal:
            decision_source = 'fast_trend'
            change_pct = fast_signal['change_pct']
            volume_ratio = fast_signal['volume_ratio']
            fast_action = fast_signal['action']
            fast_confidence = fast_signal['confidence']
            fast_reason = f"30m trend {change_pct:+.2f}% | RVOL {volume_ratio:.2f}x"

            if not (hasattr(self, '_headless_mode') and self._headless_mode):
                print("[Step 3/5] ⚡ Fast Trend Trigger - Immediate entry signal")

            global_state.add_log(f"[⚡ FAST] {fast_action.upper()} | {fast_reason}")

            if fast_action == 'open_long':
                bull_conf = fast_confidence
                bear_conf = max(0.0, 100.0 - fast_confidence)
                bull_stance = 'FAST_UP'
                bear_stance = 'HEDGE'
                bull_reason = fast_reason
                bear_reason = 'Short bias weak vs momentum'
            else:
                bull_conf = max(0.0, 100.0 - fast_confidence)
                bear_conf = fast_confidence
                bull_stance = 'HEDGE'
                bear_stance = 'FAST_DN'
                bull_reason = 'Long bias weak vs momentum'
                bear_reason = fast_reason

            decision_payload = {
                'action': fast_action,
                'confidence': fast_confidence,
                'position_size_pct': min(100.0, max(10.0, fast_confidence)),
                'reasoning': fast_reason,
                'bull_perspective': {
                    'bull_confidence': bull_conf,
                    'stance': bull_stance,
                    'bullish_reasons': bull_reason
                },
                'bear_perspective': {
                    'bear_confidence': bear_conf,
                    'stance': bear_stance,
                    'bearish_reasons': bear_reason
                }
            }
            try:
                conf_val = float(decision_payload.get('confidence', 0) or 0)
            except (TypeError, ValueError):
                conf_val = 0.0
            global_state.add_agent_message(
                "decision_core",
                f"Action: {decision_payload.get('action').upper()} | Conf: {conf_val:.1f}% | Reason: {decision_payload.get('reasoning')[:100]}... | Source: FAST",
                level="info"
            )
        elif not forced_exit:
            if llm_enabled:
                if not (hasattr(self, '_headless_mode') and self._headless_mode):
                    print("[Step 3/5] 🧠 DeepSeek LLM - Making decision...")

                global_state.add_agent_message("decision_core", "🧠 DeepSeek LLM is weighing options...", level="info")

                market_context_text = self._build_market_context(
                    quant_analysis=quant_analysis,
                    predict_result=predict_result,
                    market_data=market_data,
                    regime_info=regime_info,
                    position_info=current_position_info,
                    selected_agent_outputs=selected_agent_outputs
                )

                market_context_data = {
                    'symbol': self.symbol_manager.current_symbol,
                    'timestamp': datetime.now().isoformat(),
                    'current_price': current_price,
                    'position_side': (current_position_info or {}).get('side')
                }

                log.info("🐂🐻 Gathering Bull/Bear perspectives in PARALLEL...")
                llm_perspective_timeout = self._get_agent_timeout('llm_perspective', 45.0)
                loop = asyncio.get_running_loop()
                bull_p, bear_p = await asyncio.gather(
                    run_task_with_timeout(
                        run_id=run_id,
                        cycle_id=cycle_id,
                        agent_name="bull_agent",
                        timeout_seconds=llm_perspective_timeout,
                        task_factory=lambda: loop.run_in_executor(
                            None, self.strategy_engine.get_bull_perspective, market_context_text
                        ),
                        symbol=self.symbol_manager.current_symbol,
                        fallback={
                            "stance": "NEUTRAL",
                            "bullish_reasons": "Perspective timeout",
                            "bull_confidence": 50
                        }
                    ),
                    run_task_with_timeout(
                        run_id=run_id,
                        cycle_id=cycle_id,
                        agent_name="bear_agent",
                        timeout_seconds=llm_perspective_timeout,
                        task_factory=lambda: loop.run_in_executor(
                            None, self.strategy_engine.get_bear_perspective, market_context_text
                        ),
                        symbol=self.symbol_manager.current_symbol,
                        fallback={
                            "stance": "NEUTRAL",
                            "bearish_reasons": "Perspective timeout",
                            "bear_confidence": 50
                        }
                    )
                )

                bull_summary = bull_p.get('bullish_reasons', 'No reasons provided')
                bear_summary = bear_p.get('bearish_reasons', 'No reasons provided')
                global_state.add_agent_message("bull_agent", f"Stance: {bull_p.get('stance')} | Reason: {bull_summary}", level="success")
                global_state.add_agent_message("bear_agent", f"Stance: {bear_p.get('stance')} | Reason: {bear_summary}", level="warning")

                decision_payload = self.strategy_engine.make_decision(
                    market_context_text=market_context_text,
                    market_context_data=market_context_data,
                    reflection=reflection_text,
                    bull_perspective=bull_p,
                    bear_perspective=bear_p
                )

                try:
                    conf_val = float(decision_payload.get('confidence', 0) or 0)
                except (TypeError, ValueError):
                    conf_val = 0.0
                global_state.add_agent_message(
                    "decision_core",
                    f"Action: {decision_payload.get('action').upper()} | Conf: {conf_val:.1f}% | Reason: {decision_payload.get('reasoning')}",
                    level="info"
                )
            else:
                if not (hasattr(self, '_headless_mode') and self._headless_mode):
                    print("[Step 3/5] ⚖️ DecisionCore - Rule-based decision...")

                global_state.add_agent_message("decision_core", "⚖️ Running rule-based decision logic...", level="info")
                decision_source = 'decision_core'
                vote_core = await self.decision_core.make_decision(
                    quant_analysis=quant_analysis,
                    predict_result=predict_result,
                    market_data=market_data
                )
                size_pct = 0.0
                if vote_core.trade_params and self.max_position_size:
                    size_pct = min(
                        100.0,
                        max(5.0, (vote_core.trade_params.get('position_size', 0) / self.max_position_size) * 100)
                    )
                decision_payload = {
                    'action': vote_core.action,
                    'confidence': vote_core.confidence,
                    'position_size_pct': size_pct,
                    'reasoning': vote_core.reason,
                    'bull_perspective': {
                        'bull_confidence': 50,
                        'stance': 'NEUTRAL',
                        'bullish_reasons': 'LLM disabled'
                    },
                    'bear_perspective': {
                        'bear_confidence': 50,
                        'stance': 'NEUTRAL',
                        'bearish_reasons': 'LLM disabled'
                    }
                }
                try:
                    conf_val = float(decision_payload.get('confidence', 0) or 0)
                except (TypeError, ValueError):
                    conf_val = 0.0
                global_state.add_agent_message(
                    "decision_core",
                    f"Action: {decision_payload.get('action').upper()} | Conf: {conf_val:.1f}% | Reason: {decision_payload.get('reasoning')} | Source: RULE",
                    level="info"
                )

        if 'bull_perspective' not in decision_payload:
            decision_payload['bull_perspective'] = {
                'bull_confidence': 50,
                'stance': 'NEUTRAL',
                'bullish_reasons': 'N/A'
            }
        if 'bear_perspective' not in decision_payload:
            decision_payload['bear_perspective'] = {
                'bear_confidence': 50,
                'stance': 'NEUTRAL',
                'bearish_reasons': 'N/A'
            }
        decision_payload['action'] = normalize_action(
            decision_payload.get('action'),
            position_side=(current_position_info or {}).get('side')
        )

        from src.agents.decision_core_agent import VoteResult

        q_trend = quant_analysis.get('trend', {})
        q_osc = quant_analysis.get('oscillator', {})
        q_sent = quant_analysis.get('sentiment', {})
        q_comp = quant_analysis.get('comprehensive', {})

        vote_details = {
            'deepseek': decision_payload.get('confidence', 0),
            'strategist_total': q_comp.get('score', 0),
            'trend_1h': q_trend.get('trend_1h_score', 0),
            'trend_15m': q_trend.get('trend_15m_score', 0),
            'trend_5m': q_trend.get('trend_5m_score', 0),
            'oscillator_1h': q_osc.get('osc_1h_score', 0),
            'oscillator_15m': q_osc.get('osc_15m_score', 0),
            'oscillator_5m': q_osc.get('osc_5m_score', 0),
            'sentiment': q_sent.get('total_sentiment_score', 0),
            'prophet': predict_result.probability_up if predict_result else 0.5,
            'bull_confidence': decision_payload.get('bull_perspective', {}).get('bull_confidence', 50),
            'bear_confidence': decision_payload.get('bear_perspective', {}).get('bear_confidence', 50),
            'bull_stance': decision_payload.get('bull_perspective', {}).get('stance', 'UNKNOWN'),
            'bear_stance': decision_payload.get('bear_perspective', {}).get('stance', 'UNKNOWN'),
            'bull_reasons': decision_payload.get('bull_perspective', {}).get('bullish_reasons', ''),
            'bear_reasons': decision_payload.get('bear_perspective', {}).get('bearish_reasons', '')
        }

        if fast_signal:
            vote_details['fast_trend_change_pct'] = fast_signal.get('change_pct')
            vote_details['fast_trend_volume_ratio'] = fast_signal.get('volume_ratio')
            vote_details['fast_trend_window'] = fast_signal.get('window_minutes')

        trend_score_total = quant_analysis.get('trend', {}).get('total_trend_score', 0)
        regime_desc = SemanticConverter.get_trend_semantic(trend_score_total)

        pos_pct = decision_payload.get('position_size_pct', 0)
        if not pos_pct and decision_payload.get('position_size_usd') and self.max_position_size:
            pos_pct = (decision_payload.get('position_size_usd') / self.max_position_size) * 100
            pos_pct = min(pos_pct, 100)

        price_position_info = regime_result.get('position', {}) if regime_result else {}

        vote_result = VoteResult(
            action=decision_payload.get('action', 'wait'),
            confidence=decision_payload.get('confidence', 0),
            weighted_score=decision_payload.get('confidence', 0) - 50,
            vote_details=vote_details,
            multi_period_aligned=True,
            reason=decision_payload.get('reasoning', 'DeepSeek LLM decision'),
            regime={
                'regime': regime_desc,
                'confidence': decision_payload.get('confidence', 0)
            },
            position=price_position_info
        )

        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="decision_router",
            phase="end",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol,
            data={
                "action": vote_result.action,
                "confidence": vote_result.confidence,
                "source": decision_source
            }
        )

        return decision_payload, decision_source, fast_signal, vote_result, selected_agent_outputs

    def _build_risk_order_params(
        self,
        *,
        vote_result: Any,
        quant_analysis: Dict[str, Any],
        processed_dfs: Dict[str, "pd.DataFrame"],
        current_price: float,
        current_position_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build risk-audit input payload from decision + market context."""
        order_params = self._build_order_params(
            action=vote_result.action,
            current_price=current_price,
            confidence=vote_result.confidence,
            position_info=current_position_info
        )
        order_params['symbol'] = self.symbol_manager.current_symbol
        order_params['regime'] = vote_result.regime
        order_params['position'] = vote_result.position
        order_params['confidence'] = vote_result.confidence

        osc_data = quant_analysis.get('oscillator', {}) if isinstance(quant_analysis, dict) else {}
        order_params['oscillator_scores'] = {
            'osc_1h_score': osc_data.get('osc_1h_score', 0),
            'osc_15m_score': osc_data.get('osc_15m_score', 0),
            'osc_5m_score': osc_data.get('osc_5m_score', 0)
        }
        sentiment_data = quant_analysis.get('sentiment', {}) if isinstance(quant_analysis, dict) else {}
        order_params['sentiment_score'] = sentiment_data.get('total_sentiment_score', 0)
        order_params.update(self._get_symbol_trade_stats(self.symbol_manager.current_symbol))
        trend_data = quant_analysis.get('trend', {}) if isinstance(quant_analysis, dict) else {}
        order_params['trend_scores'] = {
            'trend_1h_score': trend_data.get('trend_1h_score', 0),
            'trend_15m_score': trend_data.get('trend_15m_score', 0),
            'trend_5m_score': trend_data.get('trend_5m_score', 0)
        }

        try:
            if self.agent_config.position_analyzer_agent:
                from src.agents.position_analyzer_agent import PositionAnalyzer
                df_1h = processed_dfs.get('1h')
                if df_1h is not None and len(df_1h) > 5:
                    analyzer = PositionAnalyzer()
                    order_params['position_1h'] = analyzer.analyze_position(
                        df_1h,
                        current_price,
                        timeframe='1h'
                    )
        except Exception:
            pass

        return order_params

    def _refresh_account_state_for_audit(self) -> float:
        """Fetch account state and sync dashboard fields before risk audit."""
        try:
            if self.test_mode:
                wallet_bal = global_state.virtual_balance
                avail_bal = global_state.virtual_balance
                unrealized_pnl = sum(
                    pos.get('unrealized_pnl', 0)
                    for pos in global_state.virtual_positions.values()
                )
                total_equity = wallet_bal + unrealized_pnl
                initial_balance = global_state.virtual_initial_balance
                total_pnl = total_equity - initial_balance

                global_state.update_account(
                    equity=total_equity,
                    available=avail_bal,
                    wallet=wallet_bal,
                    pnl=total_pnl
                )
                global_state.record_account_success()
                return float(avail_bal)

            acc_info = self.client.get_futures_account()
            wallet_bal = float(acc_info.get('total_wallet_balance', 0))
            unrealized_pnl = float(acc_info.get('total_unrealized_profit', 0))
            avail_bal = float(acc_info.get('available_balance', 0))
            total_equity = wallet_bal + unrealized_pnl

            global_state.update_account(
                equity=total_equity,
                available=avail_bal,
                wallet=wallet_bal,
                pnl=unrealized_pnl
            )
            global_state.record_account_success()
            return avail_bal
        except Exception as e:
            log.error(f"Failed to fetch account info: {e}")
            global_state.record_account_failure()
            global_state.add_log(f"❌ Account info fetch failed: {str(e)}")
            return 0.0

    async def _run_risk_audit_stage(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        vote_result: Any,
        quant_analysis: Dict[str, Any],
        processed_dfs: Dict[str, "pd.DataFrame"],
        current_price: float,
        current_position_info: Optional[Dict[str, Any]],
        regime_result: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Any, float, Optional[PositionInfo]]:
        """Run guardian audit stage and return order params + audit result."""
        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="risk_audit",
            phase="start",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol
        )

        regime_txt = vote_result.regime.get('regime', 'Unknown') if vote_result.regime else 'Unknown'
        global_state.add_log(
            f"⚖️ DecisionCoreAgent (The Critic): Context(Regime={regime_txt}) => "
            f"Vote: {vote_result.action.upper()} (Conf: {vote_result.confidence:.0f}%)"
        )
        global_state.guardian_status = "Auditing..."

        order_params = self._build_risk_order_params(
            vote_result=vote_result,
            quant_analysis=quant_analysis,
            processed_dfs=processed_dfs,
            current_price=current_price,
            current_position_info=current_position_info
        )

        print(f"  ✅ 信号方向: {vote_result.action}")
        print(f"  ✅ 综合信心: {vote_result.confidence:.1f}%")
        if vote_result.regime:
            print(f"  📊 市场状态: {vote_result.regime['regime']}")
        if vote_result.position:
            print(
                f"  📍 价格位置: {min(max(vote_result.position['position_pct'], 0), 100):.1f}% "
                f"({vote_result.position['location']})"
            )

        account_balance = self._refresh_account_state_for_audit()
        current_position = self._get_current_position()
        atr_pct = regime_result.get('atr_pct', None) if regime_result else None

        global_state.add_agent_message("risk_audit", "🛡️ Guardian is auditing risk and positions...", level="info")
        from src.agents.risk_audit_agent import RiskCheckResult, RiskLevel
        fallback_audit = RiskCheckResult(
            passed=False,
            risk_level=RiskLevel.FATAL,
            blocked_reason="risk_audit_unavailable",
            corrections=None,
            warnings=["Risk audit degraded, decision blocked by safety fallback"]
        )
        risk_timeout = self._get_agent_timeout('risk_audit', 20.0)
        try:
            audit_result = await asyncio.wait_for(
                self.risk_audit.audit_decision(
                    decision=order_params,
                    current_position=current_position,
                    account_balance=account_balance,
                    current_price=current_price,
                    atr_pct=atr_pct
                ),
                timeout=risk_timeout
            )
        except asyncio.TimeoutError:
            log.warning(f"⏱️ risk_audit timeout after {risk_timeout:.1f}s, blocking decision by fallback")
            global_state.add_agent_message(
                "risk_audit",
                f"TIMEOUT | audit>{risk_timeout:.1f}s, blocked by safety policy",
                level="warning"
            )
            emit_global_runtime_event(
                run_id=run_id,
                stream="error",
                agent="risk_audit",
                phase="timeout",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"timeout_seconds": risk_timeout}
            )
            audit_result = fallback_audit
        except Exception as e:
            emit_global_runtime_event(
                run_id=run_id,
                stream="error",
                agent="risk_audit",
                phase="error",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"error": str(e)}
            )
            log.error(f"❌ risk_audit failed, blocking decision by fallback: {e}")
            audit_result = fallback_audit

        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="risk_audit",
            phase="end",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol,
            data={
                "passed": audit_result.passed,
                "risk_level": audit_result.risk_level.value
            }
        )

        global_state.guardian_status = "PASSED" if audit_result.passed else "BLOCKED"
        if not audit_result.passed:
            global_state.add_log(f"[🛡️ GUARDIAN] ❌ BLOCKED ({audit_result.blocked_reason})")
            global_state.add_agent_message(
                "risk_audit",
                f"BLOCKED | {audit_result.blocked_reason}",
                level="warning"
            )
        else:
            global_state.add_log(f"[🛡️ GUARDIAN] ✅ PASSED (Risk: {audit_result.risk_level.value})")
            global_state.add_agent_message(
                "risk_audit",
                f"PASSED | Risk: {audit_result.risk_level.value}",
                level="success"
            )

        return order_params, audit_result, account_balance, current_position

    def _build_decision_snapshot(
        self,
        *,
        vote_result: Any,
        quant_analysis: Dict[str, Any],
        predict_result: Any,
        risk_level: str,
        guardian_passed: bool,
        guardian_reason: Optional[str] = None,
        action_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create dashboard-ready decision payload with shared enrichment fields."""
        decision_dict = asdict(vote_result)
        if action_override:
            decision_dict['action'] = action_override
        decision_dict['symbol'] = self.symbol_manager.current_symbol
        decision_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        decision_dict['cycle_number'] = global_state.cycle_counter
        decision_dict['cycle_id'] = global_state.current_cycle_id
        decision_dict['risk_level'] = risk_level
        decision_dict['guardian_passed'] = guardian_passed
        if guardian_reason is not None:
            decision_dict['guardian_reason'] = guardian_reason
        decision_dict['prophet_probability'] = predict_result.probability_up if predict_result else 0.5
        decision_dict['vote_analysis'] = SemanticConverter.convert_analysis_map(decision_dict.get('vote_details', {}))
        decision_dict['four_layer_status'] = global_state.four_layer_result
        self._attach_agent_ui_fields(decision_dict)

        if 'vote_details' not in decision_dict:
            decision_dict['vote_details'] = {}
        decision_dict['vote_details']['oi_fuel'] = quant_analysis.get('sentiment', {}).get('oi_fuel', {})

        kdj_zone = global_state.four_layer_result.get('kdj_zone')
        if not kdj_zone:
            bb_position = global_state.four_layer_result.get('bb_position', 'unknown')
            bb_to_zone_map = {
                'upper': 'overbought',
                'lower': 'oversold',
                'middle': 'neutral',
                'unknown': 'unknown'
            }
            kdj_zone = bb_to_zone_map.get(bb_position, 'unknown')
        decision_dict['vote_details']['kdj_zone'] = kdj_zone

        if 'regime' in decision_dict and decision_dict['regime']:
            decision_dict['regime']['adx'] = global_state.four_layer_result.get('adx', 20)

        if vote_result.regime:
            global_state.market_regime = vote_result.regime.get('regime', 'Unknown')
        if vote_result.position:
            pos_pct = min(max(vote_result.position.get('position_pct', 0), 0), 100)
            global_state.price_position = f"{pos_pct:.1f}% ({vote_result.position.get('location', 'Unknown')})"

        return decision_dict

    def _validate_market_snapshot(self, market_snapshot: Any) -> List[str]:
        """Validate mandatory market snapshot fields before analysis."""
        data_errors: List[str] = []

        if market_snapshot.stable_5m is None or (hasattr(market_snapshot.stable_5m, 'empty') and market_snapshot.stable_5m.empty):
            data_errors.append("5m data missing or empty")
        elif len(market_snapshot.stable_5m) < 50:
            data_errors.append(f"5m data incomplete ({len(market_snapshot.stable_5m)}/50 bars)")

        if market_snapshot.stable_15m is None or (hasattr(market_snapshot.stable_15m, 'empty') and market_snapshot.stable_15m.empty):
            data_errors.append("15m data missing or empty")
        elif len(market_snapshot.stable_15m) < 20:
            data_errors.append(f"15m data incomplete ({len(market_snapshot.stable_15m)}/20 bars)")

        if market_snapshot.stable_1h is None or (hasattr(market_snapshot.stable_1h, 'empty') and market_snapshot.stable_1h.empty):
            data_errors.append("1h data missing or empty")
        elif len(market_snapshot.stable_1h) < 10:
            data_errors.append(f"1h data incomplete ({len(market_snapshot.stable_1h)}/10 bars)")

        if not market_snapshot.live_5m or market_snapshot.live_5m.get('close', 0) <= 0:
            data_errors.append("Live price unavailable")

        return data_errors

    def _extract_position_info(self, market_snapshot: Any) -> Optional[Dict[str, Any]]:
        """Build unified position context for decision and risk stages."""
        current_position_info = None
        try:
            if self.test_mode:
                if self.symbol_manager.current_symbol in global_state.virtual_positions:
                    v_pos = global_state.virtual_positions[self.symbol_manager.current_symbol]
                    current_price_5m = market_snapshot.live_5m['close']
                    entry_price = v_pos['entry_price']
                    qty = v_pos['quantity']
                    side = v_pos['side']
                    leverage = v_pos.get('leverage', 1)

                    if side == 'LONG':
                        unrealized_pnl = (current_price_5m - entry_price) * qty
                    else:
                        unrealized_pnl = (entry_price - current_price_5m) * qty

                    margin = (entry_price * qty) / leverage if leverage > 0 else entry_price * qty
                    pnl_pct = (unrealized_pnl / margin) * 100 if margin > 0 else 0

                    current_position_info = {
                        'symbol': self.symbol_manager.current_symbol,
                        'side': side,
                        'quantity': qty,
                        'entry_price': entry_price,
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_pct': pnl_pct,
                        'leverage': leverage,
                        'is_test': True
                    }

                    v_pos['unrealized_pnl'] = unrealized_pnl
                    v_pos['pnl_pct'] = pnl_pct
                    v_pos['current_price'] = current_price_5m
                    log.info(f"💰 [Virtual Position] {side} {self.symbol_manager.current_symbol} PnL: ${unrealized_pnl:.2f} (ROE: {pnl_pct:+.2f}%)")
            else:
                try:
                    raw_pos = self.client.get_futures_position(self.symbol_manager.current_symbol)
                    if raw_pos and float(raw_pos.get('positionAmt', 0)) != 0:
                        amt = float(raw_pos.get('positionAmt', 0))
                        side = 'LONG' if amt > 0 else 'SHORT'
                        entry_price = float(raw_pos.get('entryPrice', 0))
                        unrealized_pnl = float(raw_pos.get('unRealizedProfit', 0))
                        qty = abs(amt)
                        leverage = int(raw_pos.get('leverage', 1))

                        margin = (entry_price * qty) / leverage if leverage > 0 else entry_price * qty
                        pnl_pct = (unrealized_pnl / margin) * 100 if margin > 0 else 0

                        current_position_info = {
                            'symbol': self.symbol_manager.current_symbol,
                            'side': side,
                            'quantity': qty,
                            'entry_price': entry_price,
                            'unrealized_pnl': unrealized_pnl,
                            'pnl_pct': pnl_pct,
                            'leverage': leverage,
                            'is_test': False
                        }
                        log.info(f"💰 [Real Position] {side} {self.symbol_manager.current_symbol} Amt:{amt} PnL:${unrealized_pnl:.2f} (ROE: {pnl_pct:+.2f}%)")
                except Exception as e:
                    log.error(f"Failed to fetch real position: {e}")
        except Exception as e:
            log.error(f"Error processing position info: {e}")

        return current_position_info

    def _process_market_snapshot(
        self,
        *,
        market_snapshot: Any,
        snapshot_id: str,
        cycle_id: Optional[str]
    ) -> Dict[str, "pd.DataFrame"]:
        """Save raw/indicators/features and return processed dataframes."""
        processed_dfs: Dict[str, "pd.DataFrame"] = {}
        for tf in ['5m', '15m', '1h']:
            raw_klines = getattr(market_snapshot, f'raw_{tf}')
            self.saver.save_market_data(raw_klines, self.symbol_manager.current_symbol, tf, cycle_id=cycle_id)

            stable_klines = self._get_closed_klines(raw_klines)
            df_with_indicators = self.processor.process_klines(
                stable_klines,
                self.symbol_manager.current_symbol,
                tf,
                save_raw=False
            )
            self.saver.save_indicators(df_with_indicators, self.symbol_manager.current_symbol, tf, snapshot_id, cycle_id=cycle_id)
            features_df = self.processor.extract_feature_snapshot(df_with_indicators)
            self.saver.save_features(features_df, self.symbol_manager.current_symbol, tf, snapshot_id, cycle_id=cycle_id)
            processed_dfs[tf] = df_with_indicators
        return processed_dfs

    def _build_warmup_wait_result(
        self,
        *,
        data_readiness: Dict[str, Any],
        snapshot_id: str,
        cycle_id: Optional[str]
    ) -> Dict[str, Any]:
        """Create wait result when indicators are still warming up."""
        reason = data_readiness.get('blocking_reason') or "data_warmup"
        log.warning(f"[{self.symbol_manager.current_symbol}] {reason}")
        global_state.add_log(f"[DATA] {reason}")
        global_state.oracle_status = "Warmup"
        global_state.guardian_status = "Warmup"
        global_state.four_layer_result = {
            'layer1_pass': False,
            'layer2_pass': False,
            'layer3_pass': False,
            'layer4_pass': False,
            'final_action': 'wait',
            'blocking_reason': reason,
            'data_ready': False,
            'data_validity': data_readiness['details']
        }

        decision_dict = {
            'action': 'wait',
            'confidence': 0,
            'reason': reason,
            'vote_details': {
                'data_validity': data_readiness['details']
            },
            'symbol': self.symbol_manager.current_symbol,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'cycle_number': global_state.cycle_counter,
            'cycle_id': global_state.current_cycle_id,
            'risk_level': 'safe',
            'guardian_passed': True
        }
        decision_dict['vote_analysis'] = SemanticConverter.convert_analysis_map(decision_dict.get('vote_details', {}))
        decision_dict['four_layer_status'] = global_state.four_layer_result
        self._attach_agent_ui_fields(decision_dict)

        global_state.update_decision(decision_dict)
        self.saver.save_decision(decision_dict, self.symbol_manager.current_symbol, snapshot_id, cycle_id=cycle_id)

        return {
            'status': 'wait',
            'action': 'wait',
            'details': {
                'reason': reason,
                'confidence': 0
            }
        }

    async def _run_oracle_stage(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        snapshot_id: str
    ) -> StageResult:
        """Run Step 1: fetch market data, validate, build position context, and process indicators."""
        if not (hasattr(self, '_headless_mode') and self._headless_mode):
            print("\n[Step 1/4] 🕵️ The Oracle (Data Agent) - Fetching data...")
        global_state.oracle_status = "Fetching Data..."
        global_state.add_agent_message("system", f"Fetching market data for {self.symbol_manager.current_symbol}...", level="info")
        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="oracle",
            phase="start",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol
        )

        data_sync_timeout = self._get_agent_timeout('data_sync', 200.0)
        try:
            market_snapshot = await asyncio.wait_for(
                self.data_sync_agent.fetch_all_timeframes(
                    self.symbol_manager.current_symbol,
                    limit=self.kline_limit
                ),
                timeout=data_sync_timeout
            )
        except asyncio.TimeoutError:
            error_msg = f"❌ DATA FETCH TIMEOUT: oracle>{data_sync_timeout:.1f}s"
            log.error(error_msg)
            global_state.add_log(f"[🚨 CRITICAL] {error_msg}")
            global_state.oracle_status = "DATA ERROR"
            emit_global_runtime_event(
                run_id=run_id,
                stream="error",
                agent="oracle",
                phase="timeout",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"error_type": "api_timeout", "timeout_seconds": data_sync_timeout}
            )
            return StageResult(early_result={
                'status': 'error',
                'action': 'blocked',
                'details': {
                    'reason': error_msg,
                    'error_type': 'api_timeout'
                }
            })
        except Exception as e:
            error_msg = f"❌ DATA FETCH FAILED: {str(e)}"
            log.error(error_msg)
            global_state.add_log(f"[🚨 CRITICAL] {error_msg}")
            global_state.oracle_status = "DATA ERROR"
            emit_global_runtime_event(
                run_id=run_id,
                stream="error",
                agent="oracle",
                phase="error",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"error_type": "api_failure", "error": str(e)}
            )
            return StageResult(early_result={
                'status': 'error',
                'action': 'blocked',
                'details': {
                    'reason': f'Data API call failed: {str(e)}',
                    'error_type': 'api_failure'
                }
            })

        data_errors = self._validate_market_snapshot(market_snapshot)
        if data_errors:
            error_msg = f"❌ DATA INCOMPLETE: {'; '.join(data_errors)}"
            log.error(error_msg)
            global_state.add_log(f"[🚨 CRITICAL] {error_msg}")
            global_state.oracle_status = "DATA INCOMPLETE"
            print(f"\n{'='*60}")
            print("🚨 TRADING BLOCKED - DATA ERROR")
            print(f"{'='*60}")
            for err in data_errors:
                print(f"   ❌ {err}")
            print(f"{'='*60}\n")
            emit_global_runtime_event(
                run_id=run_id,
                stream="error",
                agent="oracle",
                phase="error",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"error_type": "data_incomplete", "errors": data_errors}
            )
            return StageResult(early_result={
                'status': 'error',
                'action': 'blocked',
                'details': {
                    'reason': error_msg,
                    'error_type': 'data_incomplete',
                    'errors': data_errors
                }
            })

        global_state.oracle_status = "Data Ready"
        current_position_info = self._extract_position_info(market_snapshot)
        processed_dfs = self._process_market_snapshot(
            market_snapshot=market_snapshot,
            snapshot_id=snapshot_id,
            cycle_id=cycle_id
        )

        market_snapshot.stable_5m = processed_dfs['5m']
        market_snapshot.stable_15m = processed_dfs['15m']
        market_snapshot.stable_1h = processed_dfs['1h']

        current_price = market_snapshot.live_5m.get('close')
        print(f"  ✅ Data ready: ${current_price:,.2f} ({market_snapshot.timestamp.strftime('%H:%M:%S')})")
        global_state.add_log(f"[🕵️ ORACLE] Data ready: ${current_price:,.2f}")
        global_state.current_price[self.symbol_manager.current_symbol] = current_price

        data_readiness = self._assess_data_readiness(processed_dfs)
        if not data_readiness['is_ready']:
            emit_global_runtime_event(
                run_id=run_id,
                stream="lifecycle",
                agent="oracle",
                phase="end",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"status": "warmup"}
            )
            return StageResult(early_result=self._build_warmup_wait_result(
                data_readiness=data_readiness,
                snapshot_id=snapshot_id,
                cycle_id=cycle_id
            ))

        indicator_snapshot = self._capture_indicator_snapshot(processed_dfs, timeframe='15m')
        if indicator_snapshot:
            snapshots = getattr(global_state, 'indicator_snapshot', {})
            if not isinstance(snapshots, dict) or 'ema_status' in snapshots:
                snapshots = {}
            snapshots[self.symbol_manager.current_symbol] = indicator_snapshot
            global_state.indicator_snapshot = snapshots

        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="oracle",
            phase="end",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol,
            data={"status": "ok", "price": current_price}
        )
        return StageResult(payload={
            'market_snapshot': market_snapshot,
            'processed_dfs': processed_dfs,
            'current_price': current_price,
            'current_position_info': current_position_info
        })

    async def _run_agent_analysis_stage(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        snapshot_id: str,
        market_snapshot: Any,
        processed_dfs: Dict[str, "pd.DataFrame"]
    ) -> Tuple[Dict[str, Any], PredictResult, Any, Optional[str]]:
        """Run Step 2 parallel analysts and persist analysis context."""
        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="analysis_stage",
            phase="start",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol
        )
        try:
            if not (hasattr(self, '_headless_mode') and self._headless_mode):
                print("[Step 2/4] 👥 Multi-Agent Analysis (Parallel)...")
            global_state.add_log(f"[📊 SYSTEM] Parallel analysis started for {self.symbol_manager.current_symbol}")

            quant_analysis, predict_result, reflection_result, reflection_text = await self.parallel_analyses_runner.run(
                run_id=run_id,
                cycle_id=cycle_id,
                snapshot_id=snapshot_id,
                market_snapshot=market_snapshot,
                processed_dfs=processed_dfs
            )

            try:
                df_15m = processed_dfs['15m']
                if 'macd_diff' in df_15m.columns:
                    macd_val = float(df_15m['macd_diff'].iloc[-1])
                    if 'trend' not in quant_analysis:
                        quant_analysis['trend'] = {}
                    if 'details' not in quant_analysis['trend']:
                        quant_analysis['trend']['details'] = {}
                    quant_analysis['trend']['details']['15m_macd_diff'] = macd_val
            except Exception as e:
                log.warning(f"Failed to inject MACD data: {e}")

            self.saver.save_context(quant_analysis, self.symbol_manager.current_symbol, 'analytics', snapshot_id, cycle_id=cycle_id)

            trend_score = quant_analysis.get('trend', {}).get('total_trend_score', 0)
            osc_score = quant_analysis.get('oscillator', {}).get('total_osc_score', 0)
            sent_score = quant_analysis.get('sentiment', {}).get('total_sentiment_score', 0)
            global_state.add_log(f"[👨‍🔬 STRATEGIST] Trend={trend_score:+.0f} | Osc={osc_score:+.0f} | Sent={sent_score:+.0f}")

            emit_global_runtime_event(
                run_id=run_id,
                stream="lifecycle",
                agent="analysis_stage",
                phase="end",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol
            )
            return quant_analysis, predict_result, reflection_result, reflection_text
        except Exception as e:
            emit_global_runtime_event(
                run_id=run_id,
                stream="error",
                agent="analysis_stage",
                phase="error",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"error": str(e)}
            )
            raise

    def _run_four_layer_filter_stage(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        quant_analysis: Dict[str, Any],
        predict_result: PredictResult,
        processed_dfs: Dict[str, "pd.DataFrame"],
        current_price: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """Run four-layer strategy filtering and return regime/four-layer outputs."""
        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="four_layer_filter",
            phase="start",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol
        )
        if not (hasattr(self, '_headless_mode') and self._headless_mode):
            print("[Step 2.75/5] 🎯 Four-Layer Strategy Filter - 多层验证中...")

        sentiment = quant_analysis.get('sentiment', {})
        oi_fuel = sentiment.get('oi_fuel', {})

        funding_rate = sentiment.get('details', {}).get('funding_rate', 0)
        if funding_rate is None:
            funding_rate = 0

        df_1h = processed_dfs['1h']
        if self.regime_detector and len(df_1h) >= 20:
            regime_result = self.regime_detector.detect_regime(df_1h)
        else:
            regime_result = {'adx': 20, 'regime': 'unknown', 'confidence': 0}
        adx_value = regime_result.get('adx', 20)

        four_layer_result = {
            'layer1_pass': False,
            'layer2_pass': False,
            'layer3_pass': False,
            'layer4_pass': False,
            'final_action': 'wait',
            'blocking_reason': None,
            'confidence_boost': 0,
            'tp_multiplier': 1.0,
            'sl_multiplier': 1.0,
            'adx': adx_value,
            'funding_rate': funding_rate,
            'regime': regime_result.get('regime', 'unknown')
        }

        df_1h = processed_dfs['1h']
        if len(df_1h) >= 20:
            close_1h = df_1h['close'].iloc[-1]
            ema20_1h = df_1h['ema_20'].iloc[-1] if 'ema_20' in df_1h.columns else close_1h
            ema60_1h = df_1h['ema_60'].iloc[-1] if 'ema_60' in df_1h.columns else close_1h
            four_layer_result['close_1h'] = close_1h
            four_layer_result['ema20_1h'] = ema20_1h
            four_layer_result['ema60_1h'] = ema60_1h
        else:
            close_1h = current_price
            ema20_1h = current_price
            ema60_1h = current_price
            four_layer_result['close_1h'] = close_1h
            four_layer_result['ema20_1h'] = ema20_1h
            four_layer_result['ema60_1h'] = ema60_1h

        oi_change = oi_fuel.get('oi_change_24h', 0)
        if oi_change is None:
            oi_change = 0
        four_layer_result['oi_change'] = oi_change

        data_anomalies = []
        if abs(oi_change) > 200:
            data_anomalies.append(f"OI_ANOMALY: {oi_change:.1f}% (>200% likely data error)")
            log.warning(f"⚠️ DATA ANOMALY: OI Change {oi_change:.1f}% is abnormally high")
            oi_change = max(min(oi_change, 100), -100)
            four_layer_result['oi_change'] = oi_change
            four_layer_result['oi_change_raw'] = oi_fuel.get('oi_change_24h', 0)

        if adx_value < 5:
            data_anomalies.append(f"ADX_ANOMALY: {adx_value:.0f} (<5 may be unreliable)")
            log.warning(f"⚠️ DATA ANOMALY: ADX {adx_value:.0f} is abnormally low")

        if abs(funding_rate) > 1.0:
            data_anomalies.append(f"FUNDING_ANOMALY: {funding_rate:.3f}% (extreme)")
            log.warning(f"⚠️ DATA ANOMALY: Funding Rate {funding_rate:.3f}% is extreme")

        regime = regime_result.get('regime', 'unknown')
        if adx_value < 15 and regime in ['trending_up', 'trending_down']:
            data_anomalies.append(f"LOGIC_PARADOX: ADX={adx_value:.0f} (no trend) + Regime={regime} (trending)")
            log.warning(f"⚠️ LOGIC PARADOX: ADX={adx_value:.0f} indicates NO trend, but Regime={regime}. Forcing to choppy.")
            four_layer_result['regime'] = 'choppy'
            four_layer_result['regime_override'] = True

        four_layer_result['data_anomalies'] = data_anomalies if data_anomalies else None

        if len(df_1h) < 60:
            log.warning(f"⚠️ Insufficient 1h data: {len(df_1h)} bars (need 60+)")
            four_layer_result['blocking_reason'] = 'Insufficient 1h data'
            trend_1h = 'neutral'
        else:
            if ema20_1h > ema60_1h:
                trend_1h = 'long'
                if close_1h > ema20_1h:
                    four_layer_result['trend_confirmation'] = 'strong'
                else:
                    four_layer_result['trend_confirmation'] = 'moderate'
            elif ema20_1h < ema60_1h:
                trend_1h = 'short'
                if close_1h < ema20_1h:
                    four_layer_result['trend_confirmation'] = 'strong'
                else:
                    four_layer_result['trend_confirmation'] = 'moderate'
            else:
                trend_1h = 'neutral'

            log.info(f"📊 1h EMA: Close=${close_1h:.2f}, EMA20=${ema20_1h:.2f}, EMA60=${ema60_1h:.2f} => {trend_1h.upper()}")

        oi_divergence_warning = None
        oi_divergence_warn = 15.0
        oi_divergence_block = 60.0

        if trend_1h == 'neutral':
            four_layer_result['blocking_reason'] = 'No clear 1h trend (EMA 20/60)'
            log.info("❌ Layer 1 FAIL: No clear trend")
        elif adx_value < 15:
            four_layer_result['blocking_reason'] = f"Weak Trend Strength (ADX {adx_value:.0f} < 15)"
            log.info(f"❌ Layer 1 FAIL: ADX={adx_value:.0f} < 15, trend not strong enough")
        elif trend_1h == 'long' and oi_change < -oi_divergence_block:
            four_layer_result['blocking_reason'] = f"OI Divergence: Trend UP but OI {oi_change:.1f}%"
            log.warning(f"🚨 Layer 1 FAIL: OI Divergence - Price up but OI {oi_change:.1f}%")
        elif trend_1h == 'short' and oi_change > oi_divergence_block:
            four_layer_result['blocking_reason'] = f"OI Divergence: Trend DOWN but OI +{oi_change:.1f}%"
            log.warning(f"🚨 Layer 1 FAIL: OI Divergence - Price down but OI +{oi_change:.1f}%")
        elif trend_1h == 'long' and oi_change < -oi_divergence_warn:
            oi_divergence_warning = f"OI Divergence: Trend UP but OI {oi_change:.1f}%"
            log.warning(f"⚠️ Layer 1 WARNING: OI Divergence - Price up but OI {oi_change:.1f}%")
        elif trend_1h == 'short' and oi_change > oi_divergence_warn:
            oi_divergence_warning = f"OI Divergence: Trend DOWN but OI +{oi_change:.1f}%"
            log.warning(f"⚠️ Layer 1 WARNING: OI Divergence - Price down but OI +{oi_change:.1f}%")
        elif trend_1h == 'long' and oi_fuel.get('whale_trap_risk', False):
            four_layer_result['blocking_reason'] = f"Whale trap detected (OI {oi_change:.1f}%)"
            log.warning("🐋 Layer 1 FAIL: Whale exit trap")

        if not four_layer_result.get('blocking_reason'):
            four_layer_result['layer1_pass'] = True

            if abs(oi_change) < 1.0:
                four_layer_result['fuel_warning'] = f"Weak Fuel (OI {oi_change:.1f}%)"
                four_layer_result['confidence_penalty'] = -10
                log.warning(f"⚠️ Layer 1 WARNING: Weak fuel - OI {oi_change:.1f}% (proceed with caution)")
                fuel_strength = 'Weak'
            else:
                fuel_strength = 'Strong' if abs(oi_change) > 3.0 else 'Moderate'
            if oi_divergence_warning:
                existing_warning = four_layer_result.get('fuel_warning')
                if existing_warning:
                    four_layer_result['fuel_warning'] = f"{existing_warning} | {oi_divergence_warning}"
                else:
                    four_layer_result['fuel_warning'] = oi_divergence_warning
                current_penalty = four_layer_result.get('confidence_penalty', 0)
                four_layer_result['confidence_penalty'] = min(current_penalty, -10) if current_penalty else -10
            log.info(f"✅ Layer 1 PASS: {trend_1h.upper()} trend + {fuel_strength} Fuel (OI {oi_change:+.1f}%)")

            if self.agent_config.ai_prediction_filter_agent and self.agent_config.predict_agent:
                from src.agents.ai_prediction_filter_agent import AIPredictionFilter
                ai_filter = AIPredictionFilter()
                ai_check = ai_filter.check_divergence(trend_1h, predict_result)

                four_layer_result['ai_check'] = ai_check
                if adx_value < 5:
                    ai_check['ai_invalidated'] = True
                    ai_check['original_signal'] = ai_check.get('ai_signal', 'unknown')
                    ai_check['ai_signal'] = 'INVALID (ADX<5)'
                    four_layer_result['ai_prediction_note'] = (
                        f"AI prediction invalidated: ADX={adx_value:.0f} (<5), "
                        "directional signals are statistically meaningless"
                    )
                    log.warning(f"⚠️ AI prediction invalidated: ADX={adx_value:.0f} is too low for any directional signal to be reliable")

                if ai_check['ai_veto']:
                    four_layer_result['blocking_reason'] = ai_check['reason']
                    log.warning(f"🚫 Layer 2 VETO: {ai_check['reason']}")
                else:
                    four_layer_result['layer2_pass'] = True
                    four_layer_result['confidence_boost'] = ai_check['confidence_boost']
                    log.info(f"✅ Layer 2 PASS: AI {ai_check['ai_signal']} (boost: {ai_check['confidence_boost']:+d}%)")
            else:
                four_layer_result['layer2_pass'] = True
                four_layer_result['ai_check'] = {
                    'allow_trade': True,
                    'reason': 'AIPredictionFilter disabled',
                    'confidence_boost': 0,
                    'ai_veto': False,
                    'ai_signal': 'disabled',
                    'ai_confidence': 0
                }
                log.info("⏭️ Layer 2 SKIP: AIPredictionFilterAgent disabled")

            if four_layer_result['layer2_pass']:
                df_15m = processed_dfs['15m']
                if len(df_15m) < 20:
                    log.warning(f"⚠️ Insufficient 15m data: {len(df_15m)} bars")
                    four_layer_result['blocking_reason'] = 'Insufficient 15m data'
                    setup_ready = False
                else:
                    close_15m = df_15m['close'].iloc[-1]
                    bb_middle = df_15m['bb_middle'].iloc[-1]
                    bb_upper = df_15m['bb_upper'].iloc[-1]
                    bb_lower = df_15m['bb_lower'].iloc[-1]
                    kdj_j = df_15m['kdj_j'].iloc[-1]
                    kdj_k = df_15m['kdj_k'].iloc[-1]

                    log.info(f"📊 15m Setup: Close=${close_15m:.2f}, BB[{bb_lower:.2f}/{bb_middle:.2f}/{bb_upper:.2f}], KDJ_J={kdj_j:.1f}")
                    four_layer_result['setup_note'] = f"KDJ_J={kdj_j:.0f}, Close={'>' if close_15m > bb_middle else '<'}BB_mid"
                    four_layer_result['kdj_j'] = kdj_j
                    four_layer_result['bb_position'] = 'upper' if close_15m > bb_upper else 'lower' if close_15m < bb_lower else 'middle'

                    if kdj_j > 80 or close_15m > bb_upper:
                        four_layer_result['kdj_zone'] = 'overbought'
                    elif kdj_j < 20 or close_15m < bb_lower:
                        four_layer_result['kdj_zone'] = 'oversold'
                    else:
                        four_layer_result['kdj_zone'] = 'neutral'

                    if trend_1h == 'long':
                        if close_15m > bb_upper or kdj_j > 80:
                            setup_ready = False
                            four_layer_result['blocking_reason'] = f"15m overbought (J={kdj_j:.0f}) - wait for pullback"
                            log.info("⏳ Layer 3 WAIT: Overbought - waiting for pullback")
                        elif close_15m < bb_middle or kdj_j < 50:
                            setup_ready = True
                            four_layer_result['setup_quality'] = 'IDEAL'
                            log.info(f"✅ Layer 3 READY: IDEAL PULLBACK - J={kdj_j:.0f} < 50 or Close < BB_middle")
                        else:
                            setup_ready = True
                            four_layer_result['setup_quality'] = 'ACCEPTABLE'
                            log.info(f"✅ Layer 3 READY: Acceptable mid-range entry (J={kdj_j:.0f})")
                    elif trend_1h == 'short':
                        if close_15m < bb_lower or kdj_j < 20:
                            setup_ready = False
                            four_layer_result['blocking_reason'] = f"15m oversold (J={kdj_j:.0f}) - wait for rally"
                            log.info("⏳ Layer 3 WAIT: Oversold - waiting for rally")
                        elif close_15m > bb_middle or kdj_j > 50:
                            setup_ready = True
                            four_layer_result['setup_quality'] = 'IDEAL'
                            log.info(f"✅ Layer 3 READY: IDEAL RALLY - J={kdj_j:.0f} > 60 or Close > BB_middle")
                        else:
                            setup_ready = True
                            four_layer_result['setup_quality'] = 'ACCEPTABLE'
                            log.info(f"✅ Layer 3 READY: Acceptable mid-range entry (J={kdj_j:.0f})")
                    else:
                        setup_ready = False

                if not setup_ready:
                    four_layer_result['blocking_reason'] = "15m setup not ready"
                    log.info("⏳ Layer 3 WAIT: 15m setup not ready")
                else:
                    four_layer_result['layer3_pass'] = True
                    log.info("✅ Layer 3 PASS: 15m setup ready")

                    if self.agent_config.trigger_detector_agent:
                        from src.agents.trigger_detector_agent import TriggerDetector
                        trigger_detector = TriggerDetector()

                        df_5m = processed_dfs['5m']
                        trigger_result = trigger_detector.detect_trigger(df_5m, direction=trend_1h)
                        four_layer_result['trigger_pattern'] = trigger_result.get('pattern_type') or 'None'
                        rvol = trigger_result.get('rvol', 1.0)
                        four_layer_result['trigger_rvol'] = rvol

                        if rvol < 0.5:
                            log.warning(f"⚠️ Low Volume Warning (RVOL {rvol:.1f}x < 0.5) - Trend validation may be unreliable")
                            if not four_layer_result.get('data_anomalies'):
                                four_layer_result['data_anomalies'] = []
                            four_layer_result['data_anomalies'].append(f"Low Volume (RVOL {rvol:.1f}x)")

                        if not trigger_result['triggered']:
                            four_layer_result['blocking_reason'] = f"5min trigger not confirmed (RVOL={trigger_result.get('rvol', 1.0):.1f}x)"
                            log.info(f"⏳ Layer 4 WAIT: No engulfing or breakout pattern (RVOL={trigger_result.get('rvol', 1.0):.1f}x)")
                        else:
                            log.info(f"🎯 Layer 4 TRIGGER: {trigger_result['pattern_type']} detected")
                            sentiment_score = sentiment.get('total_sentiment_score', 0)

                            if sentiment_score > 80:
                                four_layer_result['tp_multiplier'] = 0.5
                                four_layer_result['sl_multiplier'] = 1.0
                                log.warning(f"🔴 Extreme Greed ({sentiment_score:.0f}): TP target halved")
                            elif sentiment_score < -80:
                                four_layer_result['tp_multiplier'] = 1.5
                                four_layer_result['sl_multiplier'] = 0.8
                                log.info(f"🟢 Extreme Fear ({sentiment_score:.0f}): Be greedy when others are fearful")
                            else:
                                four_layer_result['tp_multiplier'] = 1.0
                                four_layer_result['sl_multiplier'] = 1.0

                            if trend_1h == 'long' and funding_rate > 0.05:
                                four_layer_result['tp_multiplier'] *= 0.7
                                log.warning(f"💰 High Funding Rate ({funding_rate:.3f}%): Longs crowded, TP reduced")
                            elif trend_1h == 'short' and funding_rate < -0.05:
                                four_layer_result['tp_multiplier'] *= 0.7
                                log.warning(f"💰 Negative Funding Rate ({funding_rate:.3f}%): Shorts crowded, TP reduced")

                            four_layer_result['layer4_pass'] = True
                            four_layer_result['final_action'] = trend_1h
                            four_layer_result['trigger_pattern'] = trigger_result['pattern_type']
                            log.info(f"✅ Layer 4 PASS: Sentiment {sentiment_score:.0f}, Trigger={trigger_result['pattern_type']}")
                            log.info(f"🎯 ALL LAYERS PASSED: {trend_1h.upper()} with {70 + four_layer_result['confidence_boost']}% confidence")
                    else:
                        four_layer_result['trigger_pattern'] = 'disabled'
                        four_layer_result['trigger_rvol'] = None
                        four_layer_result['layer4_pass'] = True
                        four_layer_result['final_action'] = trend_1h
                        log.info("⏭️ Layer 4 SKIP: TriggerDetectorAgent disabled")

        global_state.four_layer_result = four_layer_result
        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="four_layer_filter",
            phase="end",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol,
            data={
                "final_action": four_layer_result.get('final_action', 'wait'),
                "layer4_pass": bool(four_layer_result.get('layer4_pass'))
            }
        )
        return regime_result, four_layer_result, trend_1h

    async def _run_post_filter_stage(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        current_price: float,
        trend_1h: str,
        four_layer_result: Dict[str, Any],
        quant_analysis: Dict[str, Any],
        processed_dfs: Dict[str, "pd.DataFrame"]
    ) -> None:
        """Run semantic agents + multi-period parser after four-layer filtering."""
        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="post_filter_stage",
            phase="start",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol
        )
        try:
            await self._run_semantic_analysis(
                run_id=run_id,
                cycle_id=cycle_id,
                current_price=current_price,
                trend_1h=trend_1h,
                four_layer_result=four_layer_result,
                processed_dfs=processed_dfs
            )

            try:
                multi_period_result = self.multi_period_agent.analyze(
                    quant_analysis=quant_analysis,
                    four_layer_result=four_layer_result,
                    semantic_analyses=getattr(global_state, 'semantic_analyses', {}) or {}
                )
                global_state.multi_period_result = multi_period_result
                summary = multi_period_result.get('summary')
                if summary:
                    global_state.add_agent_message("multi_period_agent", summary, level="info")
            except Exception as e:
                log.error(f"❌ Multi-Period Parser Agent failed: {e}")
                global_state.multi_period_result = {}

            emit_global_runtime_event(
                run_id=run_id,
                stream="lifecycle",
                agent="post_filter_stage",
                phase="end",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol
            )
        except Exception as e:
            emit_global_runtime_event(
                run_id=run_id,
                stream="error",
                agent="post_filter_stage",
                phase="error",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"error": str(e)}
            )
            raise

    def _record_decision_observability(
        self,
        *,
        decision_payload: Dict[str, Any],
        decision_source: str,
        vote_result: Any,
        snapshot_id: str,
        cycle_id: Optional[str]
    ) -> None:
        """Persist LLM logs and emit decision observability logs."""
        if decision_source == 'llm' and os.environ.get('ENABLE_DETAILED_LLM_LOGS', 'false').lower() == 'true':
            full_log_content = f"""
================================================================================
🕐 Timestamp: {datetime.now().isoformat()}
💱 Symbol: {self.symbol_manager.current_symbol}
🔄 Cycle: #{cycle_id}
================================================================================

--------------------------------------------------------------------------------
📤 INPUT (PROMPT)
--------------------------------------------------------------------------------
[SYSTEM PROMPT]
{decision_payload.get('system_prompt', '(Missing System Prompt)')}

[USER PROMPT]
{decision_payload.get('user_prompt', '(Missing User Prompt)')}

--------------------------------------------------------------------------------
🧠 PROCESSING (REASONING)
--------------------------------------------------------------------------------
{decision_payload.get('reasoning_detail', '(No reasoning detail)')}

--------------------------------------------------------------------------------
📥 OUTPUT (DECISION)
--------------------------------------------------------------------------------
{decision_payload.get('raw_response', '(No raw response)')}
"""
            self.saver.save_llm_log(
                content=full_log_content,
                symbol=self.symbol_manager.current_symbol,
                snapshot_id=snapshot_id,
                cycle_id=cycle_id
            )

        bull_conf = decision_payload.get('bull_perspective', {}).get('bull_confidence', 50)
        bear_conf = decision_payload.get('bear_perspective', {}).get('bear_confidence', 50)
        bull_stance = decision_payload.get('bull_perspective', {}).get('stance', 'UNKNOWN')
        bear_stance = decision_payload.get('bear_perspective', {}).get('stance', 'UNKNOWN')
        global_state.add_log(f"[🐂 Long Case] [{bull_stance}] Conf={bull_conf}%")
        global_state.add_log(f"[🐻 Short Case] [{bear_stance}] Conf={bear_conf}%")

        decision_label = "FAST Decision" if decision_source == 'fast_trend' else ("RULE Decision" if decision_source == 'decision_core' else "Final Decision")
        global_state.add_log(f"[⚖️ {decision_label}] Action={vote_result.action.upper()} | Conf={decision_payload.get('confidence', 0)}%")

        self.saver.save_decision(asdict(vote_result), self.symbol_manager.current_symbol, snapshot_id, cycle_id=cycle_id)

    def _handle_passive_decision(
        self,
        *,
        vote_result: Any,
        quant_analysis: Dict[str, Any],
        predict_result: Any,
        current_position_info: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle passive decision (wait/hold). Return cycle result if handled."""
        if not is_passive_action(vote_result.action):
            return None

        has_position = False
        if current_position_info:
            try:
                qty = float(current_position_info.get('quantity', 0) or 0)
                has_position = abs(qty) > 0
            except (TypeError, ValueError):
                has_position = True
        if not has_position and self.test_mode:
            has_position = self.symbol_manager.current_symbol in global_state.virtual_positions
        actual_action = 'hold' if has_position else 'wait'

        action_display = '持仓观望' if actual_action == 'hold' else '观望'
        print(f"\n✅ 决策: {action_display} ({actual_action})")

        regime_txt = vote_result.regime.get('regime', 'Unknown') if vote_result.regime else 'Unknown'
        pos_txt = f"{min(max(vote_result.position.get('position_pct', 0), 0), 100):.0f}%" if vote_result.position else 'N/A'
        global_state.add_log(f"⚖️ DecisionCoreAgent (The Critic): Context(Regime={regime_txt}, Pos={pos_txt}) => Vote: {actual_action.upper()} ({vote_result.reason})")

        decision_dict = self._build_decision_snapshot(
            vote_result=vote_result,
            quant_analysis=quant_analysis,
            predict_result=predict_result,
            risk_level='safe',
            guardian_passed=True,
            action_override=actual_action
        )
        global_state.update_decision(decision_dict)

        return {
            'status': actual_action,
            'action': actual_action,
            'details': {
                'reason': vote_result.reason,
                'confidence': vote_result.confidence
            }
        }

    def _finalize_risk_audit_decision(
        self,
        *,
        vote_result: Any,
        quant_analysis: Dict[str, Any],
        predict_result: Any,
        audit_result: Any,
        order_params: Dict[str, Any],
        snapshot_id: str,
        cycle_id: Optional[str],
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Persist audit outcome, apply corrections, update state, and return blocked result if needed."""
        decision_dict = self._build_decision_snapshot(
            vote_result=vote_result,
            quant_analysis=quant_analysis,
            predict_result=predict_result,
            risk_level=audit_result.risk_level.value,
            guardian_passed=audit_result.passed,
            guardian_reason=audit_result.blocked_reason
        )

        self.saver.save_risk_audit(
            audit_result={
                'passed': audit_result.passed,
                'risk_level': audit_result.risk_level.value,
                'blocked_reason': audit_result.blocked_reason,
                'corrections': audit_result.corrections,
                'warnings': audit_result.warnings,
                'order_params': order_params,
                'cycle_id': cycle_id
            },
            symbol=self.symbol_manager.current_symbol,
            snapshot_id=snapshot_id,
            cycle_id=cycle_id
        )

        print(f"  ✅ 审计结果: {'✅ 通过' if audit_result.passed else '❌ 拦截'}")
        print(f"  ✅ 风险等级: {audit_result.risk_level.value}")

        if audit_result.corrections:
            print("  ⚠️  自动修正:")
            for key, value in audit_result.corrections.items():
                print(f"     {key}: {order_params[key]} -> {value}")
                order_params[key] = value

        if audit_result.warnings:
            print("  ⚠️  警告信息:")
            for warning in audit_result.warnings:
                print(f"     {warning}")

        decision_dict['order_params'] = order_params
        global_state.update_decision(decision_dict)

        if not audit_result.passed:
            print(f"\n❌ 决策被风控拦截: {audit_result.blocked_reason}")
            return {
                'status': 'blocked',
                'action': vote_result.action,
                'details': {
                    'reason': audit_result.blocked_reason,
                    'risk_level': audit_result.risk_level.value
                },
                'current_price': current_price
            }

        return None

    def _build_analyze_only_suggestion(
        self,
        *,
        analyze_only: bool,
        vote_result: Any,
        order_params: Dict[str, Any],
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Return suggested result for open actions in analyze_only mode."""
        if not (analyze_only and vote_result.action in ('open_long', 'open_short')):
            return None

        log.info(f"🔍 [Analyze Only] Strategy suggests {vote_result.action.upper()} for {self.symbol_manager.current_symbol}, skipping execution for selector")
        return {
            'status': 'suggested',
            'action': vote_result.action,
            'confidence': vote_result.confidence,
            'order_params': order_params,
            'vote_result': vote_result,
            'current_price': current_price
        }

    async def _run_decision_pipeline_stage(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        snapshot_id: str,
        processed_dfs: Dict[str, "pd.DataFrame"],
        quant_analysis: Dict[str, Any],
        predict_result: PredictResult,
        reflection_text: Optional[str],
        current_price: float,
        current_position_info: Optional[Dict[str, Any]],
        regime_result: Optional[Dict[str, Any]]
    ) -> StageResult:
        """Run decision stage + observability + passive handling."""
        decision_payload, decision_source, fast_signal, vote_result, selected_agent_outputs = await self._run_decision_stage(
            run_id=run_id,
            cycle_id=cycle_id,
            processed_dfs=processed_dfs,
            quant_analysis=quant_analysis,
            predict_result=predict_result,
            reflection_text=reflection_text,
            current_price=current_price,
            current_position_info=current_position_info,
            regime_result=regime_result
        )

        self._record_decision_observability(
            decision_payload=decision_payload,
            decision_source=decision_source,
            vote_result=vote_result,
            snapshot_id=snapshot_id,
            cycle_id=cycle_id
        )

        passive_result = self._handle_passive_decision(
            vote_result=vote_result,
            quant_analysis=quant_analysis,
            predict_result=predict_result,
            current_position_info=current_position_info
        )
        if passive_result is not None:
            return StageResult(early_result=passive_result)

        return StageResult(payload={
            'decision_payload': decision_payload,
            'decision_source': decision_source,
            'fast_signal': fast_signal,
            'vote_result': vote_result,
            'selected_agent_outputs': selected_agent_outputs
        })

    async def _run_action_pipeline_stage(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        snapshot_id: str,
        analyze_only: bool,
        vote_result: Any,
        quant_analysis: Dict[str, Any],
        predict_result: Any,
        processed_dfs: Dict[str, "pd.DataFrame"],
        current_price: float,
        current_position_info: Optional[Dict[str, Any]],
        regime_result: Optional[Dict[str, Any]],
        market_snapshot: Any
    ) -> Dict[str, Any]:
        """Run risk-audit -> analyze-only gate -> execution as one stage."""
        if not (hasattr(self, '_headless_mode') and self._headless_mode):
            print("[Step 4/5] 👮 The Guardian (Risk Audit) - Final review...")
        order_params, audit_result, account_balance, current_position = await self._run_risk_audit_stage(
            run_id=run_id,
            cycle_id=cycle_id,
            vote_result=vote_result,
            quant_analysis=quant_analysis,
            processed_dfs=processed_dfs,
            current_price=current_price,
            current_position_info=current_position_info,
            regime_result=regime_result
        )

        blocked_result = self._finalize_risk_audit_decision(
            vote_result=vote_result,
            quant_analysis=quant_analysis,
            predict_result=predict_result,
            audit_result=audit_result,
            order_params=order_params,
            snapshot_id=snapshot_id,
            cycle_id=cycle_id,
            current_price=current_price
        )
        if blocked_result is not None:
            return blocked_result

        analyze_only_result = self._build_analyze_only_suggestion(
            analyze_only=analyze_only,
            vote_result=vote_result,
            order_params=order_params,
            current_price=current_price
        )
        if analyze_only_result is not None:
            return analyze_only_result

        return await self._run_execution_stage(
            run_id=run_id,
            cycle_id=cycle_id,
            vote_result=vote_result,
            order_params=order_params,
            current_price=current_price,
            current_position_info=current_position_info,
            current_position=current_position,
            account_balance=account_balance,
            market_snapshot=market_snapshot
        )

    async def _run_execution_stage(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        vote_result: Any,
        order_params: Dict[str, Any],
        current_price: float,
        current_position_info: Optional[Dict[str, Any]],
        current_position: Optional[PositionInfo],
        account_balance: float,
        market_snapshot: Any
    ) -> Dict[str, Any]:
        """Run order execution stage (test/live) with unified lifecycle events."""
        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="executor",
            phase="start",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol,
            data={"mode": "test" if self.test_mode else "live"}
        )

        if self.test_mode:
            return self._execute_test_mode_order(
                run_id=run_id,
                cycle_id=cycle_id,
                vote_result=vote_result,
                order_params=order_params,
                current_price=current_price,
                current_position_info=current_position_info
            )

        return self._execute_live_mode_order(
            run_id=run_id,
            cycle_id=cycle_id,
            vote_result=vote_result,
            order_params=order_params,
            current_price=current_price,
            current_position=current_position,
            account_balance=account_balance,
            market_snapshot=market_snapshot
        )

    def _execute_test_mode_order(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        vote_result: Any,
        order_params: Dict[str, Any],
        current_price: float,
        current_position_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute simulated order path for test mode."""
        if not (hasattr(self, '_headless_mode') and self._headless_mode):
            print("\n[Step 5/5] 🧪 TestMode - 模拟执行...")
        print(f"  模拟订单: {order_params['action']} {order_params['quantity']} @ {current_price}")
        global_state.add_log(f"[🚀 EXECUTOR] Test: {order_params['action'].upper()} {order_params['quantity']} @ {current_price:.2f}")

        self.saver.save_execution({
            'symbol': self.symbol_manager.current_symbol,
            'action': 'SIMULATED_EXECUTION',
            'params': order_params,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'cycle_id': cycle_id
        }, self.symbol_manager.current_symbol, cycle_id=cycle_id)

        realized_pnl = 0.0
        exit_test_price = 0.0
        normalized_action = normalize_action(
            vote_result.action,
            position_side=(current_position_info or {}).get('side')
        )

        if is_close_action(normalized_action):
            if self.symbol_manager.current_symbol in global_state.virtual_positions:
                pos = global_state.virtual_positions[self.symbol_manager.current_symbol]
                entry_price = pos['entry_price']
                qty = pos['quantity']
                side = pos['side']

                if side.upper() == 'LONG':
                    realized_pnl = (current_price - entry_price) * qty
                else:
                    realized_pnl = (entry_price - current_price) * qty

                exit_test_price = current_price
                global_state.virtual_balance += realized_pnl
                del global_state.virtual_positions[self.symbol_manager.current_symbol]
                self._save_virtual_state()
                log.info(f"💰 [TEST] Closed {side} {self.symbol_manager.current_symbol}: PnL=${realized_pnl:.2f}, Bal=${global_state.virtual_balance:.2f}")
            else:
                log.warning(f"⚠️ [TEST] Close ignored - No position for {self.symbol_manager.current_symbol}")
        elif is_open_action(normalized_action):
            side = 'LONG' if normalized_action == 'open_long' else 'SHORT'
            position_value = order_params['quantity'] * current_price
            global_state.virtual_positions[self.symbol_manager.current_symbol] = {
                'entry_price': current_price,
                'quantity': order_params['quantity'],
                'side': side,
                'entry_time': datetime.now().isoformat(),
                'stop_loss': order_params.get('stop_loss_price', 0),
                'take_profit': order_params.get('take_profit_price', 0),
                'leverage': order_params.get('leverage', 1),
                'position_value': position_value
            }
            self._save_virtual_state()
            log.info(f"💰 [TEST] Opened {side} {self.symbol_manager.current_symbol} @ ${current_price:,.2f}")

        is_close_trade_action = is_close_action(vote_result.action)
        self._persist_trade_history(
            order_params=order_params,
            cycle_id=cycle_id,
            entry_price=current_price,
            exit_price=exit_test_price,
            pnl=realized_pnl,
            is_close_trade_action=is_close_trade_action,
            open_status='SIMULATED',
            entry_field='entry_price',
            include_timestamp=True
        )

        if is_open_action(vote_result.action):
            global_state.cycle_positions_opened += 1
            log.info(f"Positions opened this cycle: {global_state.cycle_positions_opened}/1")

        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="executor",
            phase="end",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol,
            data={"status": "success", "mode": "test", "action": vote_result.action}
        )
        return {
            'status': 'success',
            'action': vote_result.action,
            'details': order_params,
            'current_price': current_price
        }

    def _execute_live_mode_order(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        vote_result: Any,
        order_params: Dict[str, Any],
        current_price: float,
        current_position: Optional[PositionInfo],
        account_balance: float,
        market_snapshot: Any
    ) -> Dict[str, Any]:
        """Execute live order path."""
        if not (hasattr(self, '_headless_mode') and self._headless_mode):
            print("\n[Step 5/5] 🚀 LiveTrade - 实盘执行...")

        try:
            is_success = self._execute_order(order_params)
            status_icon = "✅" if is_success else "❌"
            status_txt = "SENT" if is_success else "FAILED"
            global_state.add_log(f"[🚀 EXECUTOR] Live: {order_params['action'].upper()} {order_params['quantity']} => {status_icon} {status_txt}")
            executed = {'status': 'filled' if is_success else 'failed', 'avgPrice': current_price, 'executedQty': order_params['quantity']}
        except Exception as e:
            log.error(f"Live order execution failed: {e}", exc_info=True)
            global_state.add_log(f"[Execution] ❌ Live Order Failed: {e}")
            emit_global_runtime_event(
                run_id=run_id,
                stream="error",
                agent="executor",
                phase="error",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"status": "failed", "mode": "live", "error": str(e)}
            )
            return {
                'status': 'failed',
                'action': vote_result.action,
                'details': {'error': str(e)}
            }

        self.saver.save_execution({
            'symbol': self.symbol_manager.current_symbol,
            'action': 'REAL_EXECUTION',
            'params': order_params,
            'status': 'success' if executed else 'failed',
            'timestamp': datetime.now().isoformat(),
            'cycle_id': cycle_id
        }, self.symbol_manager.current_symbol, cycle_id=cycle_id)

        if executed:
            print("  ✅ 订单执行成功!")
            log_price = order_params.get('entry_price', current_price)
            global_state.add_log(f"✅ Order: {order_params['action'].upper()} {order_params['quantity']} @ ${log_price}")

            trade_logger.log_open_position(
                symbol=self.symbol_manager.current_symbol,
                side=order_params['action'].upper(),
                decision=order_params,
                execution_result={
                    'success': True,
                    'entry_price': order_params['entry_price'],
                    'quantity': order_params['quantity'],
                    'stop_loss': order_params['stop_loss'],
                    'take_profit': order_params['take_profit'],
                    'order_id': 'real_order'
                },
                market_state=market_snapshot.live_5m,
                account_info={'available_balance': account_balance}
            )

            pnl = 0.0
            exit_price = 0.0
            entry_price = order_params['entry_price']
            if is_close_action(order_params.get('action')) and current_position:
                exit_price = current_price
                entry_price = current_position.entry_price
                direction = 1 if current_position.side == 'long' else -1
                pnl = (exit_price - entry_price) * current_position.quantity * direction

            is_close_trade_action = is_close_action(order_params.get('action'))
            self._persist_trade_history(
                order_params=order_params,
                cycle_id=cycle_id,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                is_close_trade_action=is_close_trade_action,
                open_status='EXECUTED',
                entry_field='price',
                include_timestamp=False
            )

            emit_global_runtime_event(
                run_id=run_id,
                stream="lifecycle",
                agent="executor",
                phase="end",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"status": "success", "mode": "live", "action": vote_result.action}
            )
            return {
                'status': 'success',
                'action': vote_result.action,
                'details': order_params,
                'current_price': current_price
            }

        print("  ❌ 订单执行失败")
        global_state.add_log(f"❌ Order Failed: {order_params['action'].upper()}")
        emit_global_runtime_event(
            run_id=run_id,
            stream="error",
            agent="executor",
            phase="error",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol,
            data={"status": "failed", "mode": "live", "error": "execution_failed"}
        )
        return {
            'status': 'failed',
            'action': vote_result.action,
            'details': {'error': 'execution_failed'},
            'current_price': current_price
        }

    def _persist_trade_history(
        self,
        *,
        order_params: Dict[str, Any],
        cycle_id: Optional[str],
        entry_price: float,
        exit_price: float,
        pnl: float,
        is_close_trade_action: bool,
        open_status: str,
        entry_field: str,
        include_timestamp: bool
    ) -> bool:
        """Persist/merge trade record to storage + in-memory history."""
        update_success = False
        if is_close_trade_action:
            update_success = self.saver.update_trade_exit(
                symbol=self.symbol_manager.current_symbol,
                exit_price=exit_price,
                pnl=pnl,
                exit_time=datetime.now().strftime("%H:%M:%S"),
                close_cycle=global_state.cycle_counter
            )
            if update_success:
                for trade in global_state.trade_history:
                    if trade.get('symbol') == self.symbol_manager.current_symbol and trade.get('exit_price', 0) == 0:
                        trade['exit_price'] = exit_price
                        trade['pnl'] = pnl
                        trade['close_cycle'] = global_state.cycle_counter
                        trade['status'] = 'CLOSED'
                        log.info(f"✅ Synced global_state.trade_history: {self.symbol_manager.current_symbol} PnL ${pnl:.2f}")
                        break
                global_state.cumulative_realized_pnl += pnl
                log.info(f"📊 Cumulative Realized PnL: ${global_state.cumulative_realized_pnl:.2f}")

        if not update_success:
            is_open_trade_action = is_open_action(order_params.get('action'))
            original_open_cycle = 0
            if not is_open_trade_action:
                for trade in global_state.trade_history:
                    if trade.get('symbol') == self.symbol_manager.current_symbol and trade.get('exit_price', 0) == 0:
                        original_open_cycle = trade.get('open_cycle', 0)
                        break

            trade_record = {
                'open_cycle': global_state.cycle_counter if is_open_trade_action else original_open_cycle,
                'close_cycle': 0 if is_open_trade_action else global_state.cycle_counter,
                'action': order_params['action'].upper(),
                'symbol': self.symbol_manager.current_symbol,
                entry_field: entry_price,
                'quantity': order_params['quantity'],
                'cost': entry_price * order_params['quantity'],
                'exit_price': exit_price,
                'pnl': pnl,
                'confidence': order_params['confidence'],
                'status': open_status,
                'cycle': cycle_id
            }
            if include_timestamp:
                trade_record['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if is_close_trade_action:
                trade_record['status'] = 'CLOSED (Fallback)'

            self.saver.save_trade(trade_record)
            global_state.trade_history.insert(0, trade_record)
            if len(global_state.trade_history) > 50:
                global_state.trade_history.pop()

        return update_success

    async def _resolve_auto3_symbols(self):
        """
        🔝 AUTO3 Dynamic Resolution via Backtest
        
        Gets AI500 Top 5 by volume, backtests each, and selects top 2
        """
        selector = get_selector()
        account_equity = self.client.get_account_equity_estimate()
        if hasattr(selector, 'account_equity') and account_equity:
            selector.account_equity = account_equity
        top3 = await selector.select_top3(force_refresh=False, account_equity=account_equity)
        
        log.info(f"🔝 AUTO3 resolved to: {', '.join(top3)}")
        return top3

    def _get_closed_klines(self, klines: List[Dict]) -> List[Dict]:
        """Return klines confirmed closed (avoid dropping already-closed bars)."""
        if not klines:
            return []

        last = klines[-1]
        if last.get('is_closed') is True:
            return klines

        close_time = last.get('close_time')
        if close_time is not None:
            try:
                now_ms = int(datetime.now().timestamp() * 1000)
                if int(close_time) <= now_ms:
                    return klines
            except (TypeError, ValueError):
                pass

        return klines[:-1]

    def _assess_data_readiness(self, processed_dfs: Dict[str, object]) -> Dict:
        """Check warmup/is_valid flags for all timeframes before making decisions."""
        readiness = {
            'is_ready': True,
            'details': {},
            'blocking_reason': None
        }

        min_valid_index = self.processor._get_min_valid_index()
        required_bars = min_valid_index + 1

        for tf in ['5m', '15m', '1h']:
            df = processed_dfs.get(tf)
            if df is None or df.empty:
                readiness['is_ready'] = False
                readiness['details'][tf] = {
                    'is_valid': False,
                    'bars': 0,
                    'bars_needed': required_bars,
                    'bars_remaining': required_bars,
                    'reason': 'no_data'
                }
                continue

            reason = None
            if 'is_valid' in df.columns:
                latest_valid = bool(df['is_valid'].iloc[-1] == True)
            elif 'is_warmup' in df.columns:
                latest_valid = bool(df['is_warmup'].iloc[-1] == False)
            else:
                latest_valid = False
                reason = 'missing_valid_flag'

            bars = len(df)
            bars_remaining = max(0, required_bars - bars)

            readiness['details'][tf] = {
                'is_valid': latest_valid,
                'bars': bars,
                'bars_needed': required_bars,
                'bars_remaining': bars_remaining,
                'reason': reason
            }

            if not latest_valid:
                readiness['is_ready'] = False

        if not readiness['is_ready']:
            reasons = []
            for tf, info in readiness['details'].items():
                if not info['is_valid']:
                    if info.get('reason'):
                        reasons.append(f"{tf}:{info['reason']}")
                    elif info.get('bars', 0) > 0:
                        reasons.append(f"{tf}:warmup {info['bars']}/{info['bars_needed']}")
                    else:
                        reasons.append(f"{tf}:no_data")
            readiness['blocking_reason'] = "data_warmup: " + ", ".join(reasons)

        return readiness
    
    def _init_accounts(self):
        """
        Initialize trading accounts from config or legacy .env
        
        Priority:
        1. Load from config/accounts.json if exists
        2. Auto-create default account from legacy .env if no accounts loaded
        """
        import os
        from pathlib import Path
        
        config_path = Path(__file__).parent / "config" / "accounts.json"
        
        # Try to load from config file
        loaded = self.account_manager.load_from_file(str(config_path))
        
        if loaded == 0:
            # No accounts.json found - create default from legacy .env
            log.info("No accounts.json found, creating default account from .env")
            
            api_key = os.environ.get('BINANCE_API_KEY', '')
            secret_key = os.environ.get('BINANCE_SECRET_KEY', '')
            testnet = os.environ.get('BINANCE_TESTNET', 'true').lower() == 'true'
            
            if api_key:
                default_account = ExchangeAccount(
                    id='main-binance',
                    user_id='default',
                    exchange_type=ExchangeType.BINANCE,
                    account_name='Main Binance Account',
                    enabled=True,
                    api_key=api_key,
                    secret_key=secret_key,
                    testnet=testnet or self.test_mode
                )
                self.account_manager.add_account(default_account)
                log.info(f"✅ Created default account: {default_account.account_name}")
            else:
                log.warning("No API key found in .env - running in demo mode")
        
        # Log summary
        accounts = self.account_manager.list_accounts(enabled_only=True)
        if accounts:
            print(f"  📊 Loaded {len(accounts)} trading accounts:")
            for acc in accounts:
                print(f"     - {acc.account_name} ({acc.exchange_type.value}, testnet={acc.testnet})")
    
    def get_accounts(self):
        """Get list of enabled trading accounts."""
        return self.account_manager.list_accounts(enabled_only=True)
    
    async def get_trader(self, account_id: str):
        """Get trader instance for a specific account."""
        return await self.account_manager.get_trader(account_id)

    def _begin_cycle_context(self) -> CycleContext:
        """Initialize cycle-scoped context and emit system-start observability."""
        if hasattr(self, '_headless_mode') and self._headless_mode:
            self._terminal_display.print_log(f"🔍 Analyzing {self.symbol_manager.current_symbol}...", "INFO")
        else:
            print(f"\n{'='*80}")
            print(f"🔄 启动交易审计循环 | {datetime.now().strftime('%H:%M:%S')} | {self.symbol_manager.current_symbol}")
            print(f"{'='*80}")

        global_state.is_running = True
        global_state.current_symbol = self.symbol_manager.current_symbol    # maybe not needed
        run_id = f"run_{int(time.time() * 1000)}:{self.symbol_manager.current_symbol}"

        cycle_num = global_state.cycle_counter
        cycle_id = global_state.current_cycle_id
        run_id = f"{cycle_id}:{self.symbol_manager.current_symbol}" if cycle_id else run_id
        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="system",
            phase="start",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol,
            data={"cycle": cycle_num, "symbol": self.symbol_manager.current_symbol}
        )

        global_state.add_log(f"[📊 SYSTEM] {self.symbol_manager.current_symbol} analysis started")
        global_state.agent_messages = [msg for msg in global_state.agent_messages if msg.get('symbol') != self.symbol_manager.current_symbol]
        snapshot_id = f"snap_{int(time.time())}"

        return CycleContext(
            run_id=run_id,
            cycle_id=cycle_id,
            snapshot_id=snapshot_id,
            cycle_num=cycle_num,
            symbol=self.symbol_manager.current_symbol
        )

    async def _run_cycle_pipeline(self, *, context: CycleContext, analyze_only: bool) -> Dict[str, Any]:
        """Run the full trading pipeline using a prepared cycle context."""
        emit_global_runtime_event(
            run_id=context.run_id,
            stream="lifecycle",
            agent="cycle_pipeline",
            phase="start",
            cycle_id=context.cycle_id,
            symbol=self.symbol_manager.current_symbol,
            data={"symbol": context.symbol, "cycle": context.cycle_num}
        )
        try:
            oracle_result = await self._run_oracle_stage(
                run_id=context.run_id,
                cycle_id=context.cycle_id,
                snapshot_id=context.snapshot_id
            )
            if oracle_result.early_result is not None:
                early = oracle_result.early_result
                emit_cycle_pipeline_end(context=context, result=early)
                return early

            market_snapshot = oracle_result.payload['market_snapshot']
            processed_dfs = oracle_result.payload['processed_dfs']
            current_price = oracle_result.payload['current_price']
            current_position_info = oracle_result.payload['current_position_info']

            quant_analysis, predict_result, _reflection_result, reflection_text = await self._run_agent_analysis_stage(
                run_id=context.run_id,
                cycle_id=context.cycle_id,
                snapshot_id=context.snapshot_id,
                market_snapshot=market_snapshot,
                processed_dfs=processed_dfs
            )

            regime_result, four_layer_result, trend_1h = self._run_four_layer_filter_stage(
                run_id=context.run_id,
                cycle_id=context.cycle_id,
                quant_analysis=quant_analysis,
                predict_result=predict_result,
                processed_dfs=processed_dfs,
                current_price=current_price
            )

            await self._run_post_filter_stage(
                run_id=context.run_id,
                cycle_id=context.cycle_id,
                current_price=current_price,
                trend_1h=trend_1h,
                four_layer_result=four_layer_result,
                quant_analysis=quant_analysis,
                processed_dfs=processed_dfs
            )

            decision_result = await self._run_decision_pipeline_stage(
                run_id=context.run_id,
                cycle_id=context.cycle_id,
                snapshot_id=context.snapshot_id,
                processed_dfs=processed_dfs,
                quant_analysis=quant_analysis,
                predict_result=predict_result,
                reflection_text=reflection_text,
                current_price=current_price,
                current_position_info=current_position_info,
                regime_result=regime_result
            )
            if decision_result.early_result is not None:
                early = decision_result.early_result
                emit_cycle_pipeline_end(context=context, result=early)
                return early
            vote_result = decision_result.payload['vote_result']

            result = await self._run_action_pipeline_stage(
                run_id=context.run_id,
                cycle_id=context.cycle_id,
                snapshot_id=context.snapshot_id,
                analyze_only=analyze_only,
                vote_result=vote_result,
                quant_analysis=quant_analysis,
                predict_result=predict_result,
                processed_dfs=processed_dfs,
                current_price=current_price,
                current_position_info=current_position_info,
                regime_result=regime_result,
                market_snapshot=market_snapshot
            )
            emit_cycle_pipeline_end(context=context, result=result)
            return result
        except Exception as e:
            emit_global_runtime_event(
                run_id=context.run_id,
                stream="error",
                agent="cycle_pipeline",
                phase="error",
                cycle_id=context.cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"error": str(e)}
            )
            raise

    async def run_trading_cycle(self, analyze_only: bool = False) -> Dict:
        """
        执行完整的交易循环（异步版本）
        Returns:
            {
                'status': 'success/failed/wait/blocked/suggested',
                'action': 'open_long/open_short/close_long/close_short/wait/hold',
                'details': {...}
            }
        """
        cycle_context: Optional[CycleContext] = None
        run_id = f"run_{int(time.time() * 1000)}:{self.symbol_manager.current_symbol}"
        try:
            cycle_context = self._begin_cycle_context()
            run_id = cycle_context.run_id
            return await self._run_cycle_pipeline(context=cycle_context, analyze_only=analyze_only)
        
        except Exception as e:
            log.error(f"Trading cycle exception: {e}", exc_info=True)
            global_state.add_log(f"Error: {e}")
            emit_global_runtime_event(
                run_id=run_id,
                stream="error",
                agent="system",
                phase="error",
                cycle_id=(cycle_context.cycle_id if cycle_context else global_state.current_cycle_id),
                symbol=self.symbol_manager.current_symbol,
                data={"status": "error", "error": str(e)}
            )
            return {
                'status': 'error',
                'details': {'error': str(e)}
            }
    
    def _build_order_params(
        self, 
        action: str, 
        current_price: float,
        confidence: float,
        position_info: Optional[Dict] = None
    ) -> Dict:
        """
        构建订单参数
        
        Args:
            action: trading action
            current_price: 当前价格
            confidence: 决策置信度 (0-100)
        
        Returns:
            订单参数字典
        """
        action = normalize_action(action, position_side=(position_info or {}).get('side'))

        # 获取可用余额
        if self.test_mode:
            available_balance = global_state.virtual_balance
        else:
            available_balance = self.client.get_account_balance()
        
        # 动态仓位计算：置信度 100% 时使用可用余额的 33%
        # 公式: 仓位比例 = 基础比例(33%) × 置信度
        base_position_pct = 1 / 3  # 最大仓位比例 33%
        conf_pct = confidence
        if isinstance(conf_pct, (int, float)) and 0 < conf_pct <= 1:
            conf_pct *= 100
        conf_pct = max(0.0, min(float(conf_pct or 0.0), 100.0))
        position_pct = base_position_pct * (conf_pct / 100)  # 根据置信度调整
        
        # 计算仓位金额（完全基于可用余额百分比）
        adjusted_position = available_balance * position_pct
        
        # 计算数量
        quantity = adjusted_position / current_price if current_price > 0 else 0.0
        if is_close_action(action):
            if position_info and isinstance(position_info.get('quantity'), (int, float)):
                quantity = float(position_info.get('quantity', 0) or 0)
            elif self.test_mode:
                pos = (global_state.virtual_positions or {}).get(self.symbol_manager.current_symbol, {})
                quantity = float(pos.get('quantity', 0) or 0)
        
        # 计算止损止盈
        if action == 'open_long':
            stop_loss = current_price * (1 - self.stop_loss_pct / 100)
            take_profit = current_price * (1 + self.take_profit_pct / 100)
        elif action == 'open_short':
            stop_loss = current_price * (1 + self.stop_loss_pct / 100)
            take_profit = current_price * (1 - self.take_profit_pct / 100)
        else:
            stop_loss = current_price
            take_profit = current_price
        
        return {
            'action': action,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'quantity': quantity,
            'position_value': adjusted_position,  # 新增：实际仓位金额
            'position_pct': position_pct * 100,   # 新增：仓位百分比
            'leverage': self.leverage,
            'confidence': confidence
        }

    def _get_symbol_trade_stats(self, symbol: str, max_trades: int = 5) -> Dict:
        """Summarize recent closed trades for symbol to support risk filters."""
        history = global_state.trade_history or []
        loss_streak = 0
        loss_streak_active = True
        recent_pnl = 0.0
        recent_count = 0
        recent_wins = 0

        for trade in history:
            if trade.get('symbol') != symbol:
                continue

            pnl = trade.get('pnl')
            if pnl is None:
                continue

            status = str(trade.get('status', '')).upper()
            close_cycle = trade.get('close_cycle', 0)
            exit_price = trade.get('exit_price', 0)
            is_closed = (
                'CLOSED' in status or
                (isinstance(close_cycle, (int, float)) and close_cycle > 0) or
                (isinstance(exit_price, (int, float)) and exit_price > 0)
            )
            if not is_closed:
                continue

            try:
                pnl_value = float(pnl)
            except Exception:
                continue

            if loss_streak_active:
                if pnl_value < 0:
                    loss_streak += 1
                else:
                    loss_streak_active = False

            if recent_count < max_trades:
                recent_pnl += pnl_value
                recent_count += 1
                if pnl_value > 0:
                    recent_wins += 1

            if not loss_streak_active and recent_count >= max_trades:
                break

        win_rate = (recent_wins / recent_count) if recent_count > 0 else None
        return {
            'symbol_loss_streak': loss_streak,
            'symbol_recent_pnl': recent_pnl,
            'symbol_recent_trades': recent_count,
            'symbol_win_rate': win_rate
        }

    def _get_open_trade_meta(self, symbol: str) -> Optional[Dict]:
        """Return latest open trade record for symbol, if any."""
        history = global_state.trade_history or []
        for trade in history:
            if trade.get('symbol') != symbol:
                continue
            status = str(trade.get('status', '')).upper()
            close_cycle = trade.get('close_cycle', 0)
            exit_price = trade.get('exit_price', 0)
            is_closed = (
                'CLOSED' in status or
                (isinstance(close_cycle, (int, float)) and close_cycle > 0) or
                (isinstance(exit_price, (int, float)) and exit_price > 0)
            )
            if not is_closed:
                return trade
        return None

    def _get_holding_cycles(self, open_trade: Optional[Dict]) -> Optional[int]:
        """Estimate holding cycles from open trade metadata."""
        if not open_trade:
            return None
        open_cycle = open_trade.get('open_cycle')
        if isinstance(open_cycle, (int, float)):
            return max(0, int(global_state.cycle_counter) - int(open_cycle))
        return None

    def _get_holding_hours(self, symbol: str, open_trade: Optional[Dict]) -> Optional[float]:
        """Estimate holding duration in hours using cycle counter or entry_time."""
        if open_trade:
            hold_cycles = self._get_holding_cycles(open_trade)
            if hold_cycles is not None:
                cycle_interval = max(1, int(getattr(global_state, 'cycle_interval', 3) or 3))
                hold_minutes = hold_cycles * cycle_interval
                return hold_minutes / 60.0
        if self.test_mode:
            v_pos = global_state.virtual_positions.get(symbol)
            entry_time = v_pos.get('entry_time') if isinstance(v_pos, dict) else None
            if entry_time:
                try:
                    started = datetime.fromisoformat(entry_time)
                    return max(0.0, (datetime.now() - started).total_seconds() / 3600.0)
                except Exception:
                    pass
        return None

    def _check_forced_exit(self, position_info: Optional[Dict]) -> Optional[Dict]:
        """Force exit for stale or losing positions to cap drawdowns."""
        if not position_info:
            return None
        symbol = position_info.get('symbol') or self.symbol_manager.current_symbol
        side = str(position_info.get('side', '')).lower()
        close_action = 'close_long' if side == 'long' else ('close_short' if side == 'short' else 'wait')
        pnl_pct = position_info.get('pnl_pct')
        if pnl_pct is None:
            return None

        open_trade = self._get_open_trade_meta(symbol)
        hold_cycles = self._get_holding_cycles(open_trade)
        hold_hours = self._get_holding_hours(symbol, open_trade)
        max_hold_cycles = self.config.get('risk.max_holding_cycles')
        max_hold_hours = self.config.get('risk.max_holding_hours')
        if max_hold_cycles is None:
            max_hold_cycles = 180
        if max_hold_hours is None and hold_cycles is not None:
            cycle_interval = max(1, int(getattr(global_state, 'cycle_interval', 3) or 3))
            max_hold_hours = (max_hold_cycles * cycle_interval) / 60.0
        hold_tag = f"{hold_hours:.1f}h" if hold_hours is not None else "n/a"

        if hold_cycles is not None and isinstance(max_hold_cycles, (int, float)):
            if hold_cycles >= int(max_hold_cycles):
                return {
                    'action': close_action,
                    'confidence': 92,
                    'reasoning': f"Forced exit: holding cycles cap {int(max_hold_cycles)} hit ({hold_cycles} cycles, {hold_tag})"
                }
        if hold_hours is not None and isinstance(max_hold_hours, (int, float)):
            if hold_hours >= float(max_hold_hours):
                return {
                    'action': close_action,
                    'confidence': 92,
                    'reasoning': f"Forced exit: holding hours cap {float(max_hold_hours):.1f}h hit ({hold_tag})"
                }

        # Immediate loss cut
        if pnl_pct <= -5:
            return {
                'action': close_action,
                'confidence': 95,
                'reasoning': f"Forced exit: loss {pnl_pct:+.2f}% exceeds -5% cap (hold {hold_tag})"
            }

        # Time-based exit for losing/stale positions
        if hold_hours is not None:
            if hold_hours >= 6 and pnl_pct < -1:
                return {
                    'action': close_action,
                    'confidence': 90,
                    'reasoning': f"Forced exit: loss {pnl_pct:+.2f}% with stale hold {hold_tag}"
                }
            if hold_hours >= 12 and pnl_pct <= 0.3:
                return {
                    'action': close_action,
                    'confidence': 85,
                    'reasoning': f"Forced exit: capital tie-up {hold_tag} with low edge ({pnl_pct:+.2f}%)"
                }
        return None

    def _get_cycle_logger(self):
        if self._cycle_logger is None:
            try:
                from src.monitoring.logger import TradingLogger
                self._cycle_logger = TradingLogger()
            except Exception as e:
                log.error(f"Cycle logger init failed: {e}")
                self._cycle_logger = False
        return self._cycle_logger if self._cycle_logger is not False else None

    def _record_cycle_summary(
        self,
        cycle_number: int,
        cycle_id: Optional[str],
        timestamp_start: str,
        timestamp_end: str,
        symbols: List[str],
        traded: bool,
        trade_symbol: Optional[str],
        trade_action: Optional[str],
        trade_status: Optional[str]
    ) -> None:
        logger = self._get_cycle_logger()
        if not logger:
            return

        realized_total = float(getattr(global_state, 'cumulative_realized_pnl', 0.0) or 0.0)
        cycle_realized = realized_total - (self._last_cycle_realized_pnl or 0.0)
        self._last_cycle_realized_pnl = realized_total

        if self.test_mode:
            unrealized = sum(
                float(pos.get('unrealized_pnl', 0) or 0)
                for pos in global_state.virtual_positions.values()
            )
            balance = float(global_state.virtual_balance or 0.0)
            equity = balance + unrealized
        else:
            acc = global_state.account_overview or {}
            balance = float(acc.get('wallet_balance') or acc.get('available_balance') or 0.0)
            equity = float(acc.get('total_equity') or 0.0)
            if balance and equity:
                unrealized = equity - balance
            else:
                unrealized = float(acc.get('total_pnl') or 0.0)

        total_pnl = realized_total + unrealized

        try:
            logger.log_cycle({
                'cycle_number': cycle_number,
                'cycle_id': cycle_id,
                'timestamp_start': timestamp_start,
                'timestamp_end': timestamp_end,
                'symbols': ','.join(symbols) if symbols else '',
                'traded': traded,
                'trade_symbol': trade_symbol,
                'trade_action': trade_action,
                'trade_status': trade_status,
                'realized_pnl': realized_total,
                'unrealized_pnl': unrealized,
                'total_pnl': total_pnl,
                'cycle_realized_pnl': cycle_realized,
                'equity': equity,
                'balance': balance,
                'notes': None
            })
        except Exception as e:
            log.error(f"Cycle log insert failed: {e}")
    
    def _get_current_position(self) -> Optional[PositionInfo]:
        """获取当前持仓 (支持实盘 + Test Mode)"""
        try:
            # 1. Test Mode Support
            if self.test_mode:
                if self.symbol_manager.current_symbol in global_state.virtual_positions:
                    v_pos = global_state.virtual_positions[self.symbol_manager.current_symbol]
                    return PositionInfo(
                        symbol=self.symbol_manager.current_symbol,
                        side=v_pos['side'].lower(), # ensure lowercase 'long'/'short'
                        entry_price=v_pos['entry_price'],
                        quantity=v_pos['quantity'],
                        unrealized_pnl=v_pos.get('unrealized_pnl', 0)
                    )
                return None

            # 2. Live Mode Support
            pos = self.client.get_futures_position(self.symbol_manager.current_symbol)
            if pos and abs(pos['position_amt']) > 0:
                return PositionInfo(
                    symbol=self.symbol_manager.current_symbol,
                    side='long' if pos['position_amt'] > 0 else 'short',
                    entry_price=pos['entry_price'],
                    quantity=abs(pos['position_amt']),
                    unrealized_pnl=pos['unrealized_profit']
                )
            return None
        except Exception as e:
            log.error(f"Failed to get positions: {e}")
            return None

    def _execute_suggested_open_trade(self, symbol: str, suggested: Any, cycle_id: Optional[str]) -> Dict:
        """Execute an already-audited open suggestion without re-running full analysis."""
        if isinstance(suggested, SuggestedTrade):
            suggestion_symbol = suggested.symbol
            order_params = dict(suggested.order_params or {})
            suggested_price = suggested.current_price
        else:
            suggestion_symbol = symbol
            order_params = dict((suggested or {}).get('order_params') or {})
            suggested_price = (suggested or {}).get('current_price')

        if not order_params:
            return {'status': 'failed', 'action': 'wait', 'details': {'error': 'missing_order_params'}}

        action = normalize_action(order_params.get('action'))
        if not is_open_action(action):
            return {'status': 'failed', 'action': action, 'details': {'error': 'not_open_action'}}

        order_params['action'] = action
        order_params['symbol'] = suggestion_symbol
        self.symbol_manager.current_symbol = suggestion_symbol

        try:
            current_price = float(
                suggested_price
                or order_params.get('entry_price')
                or global_state.current_price.get(suggestion_symbol, 0)
                or 0
            )
        except (TypeError, ValueError):
            current_price = 0.0
        if current_price <= 0:
            return {'status': 'failed', 'action': action, 'details': {'error': 'invalid_price'}}

        if self.test_mode:
            side = 'LONG' if action == 'open_long' else 'SHORT'
            quantity = float(order_params.get('quantity', 0) or 0)
            position_value = quantity * current_price
            global_state.virtual_positions[suggestion_symbol] = {
                'entry_price': current_price,
                'quantity': quantity,
                'side': side,
                'entry_time': datetime.now().isoformat(),
                'stop_loss': order_params.get('stop_loss', 0),
                'take_profit': order_params.get('take_profit', 0),
                'leverage': order_params.get('leverage', 1),
                'position_value': position_value,
            }
            self._save_virtual_state()

            self.saver.save_execution({
                'symbol': suggestion_symbol,
                'action': 'SIMULATED_EXECUTION',
                'params': order_params,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'cycle_id': cycle_id,
            }, suggestion_symbol, cycle_id=cycle_id)

            trade_record = {
                'open_cycle': global_state.cycle_counter,
                'close_cycle': 0,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': action.upper(),
                'symbol': suggestion_symbol,
                'entry_price': current_price,
                'quantity': quantity,
                'cost': position_value,
                'exit_price': 0,
                'pnl': 0.0,
                'confidence': order_params.get('confidence'),
                'status': 'SIMULATED',
                'cycle': cycle_id,
            }
            self.saver.save_trade(trade_record)
            global_state.trade_history.insert(0, trade_record)
            if len(global_state.trade_history) > 50:
                global_state.trade_history.pop()
            global_state.cycle_positions_opened += 1
            global_state.add_log(f"[🚀 EXECUTOR] Test: {action.upper()} {quantity} @ {current_price:.2f}")
            return {'status': 'success', 'action': action, 'details': order_params, 'current_price': current_price}

        is_success = self._execute_order(order_params)
        self.saver.save_execution({
            'symbol': symbol,
            'action': 'REAL_EXECUTION',
            'params': order_params,
            'status': 'success' if is_success else 'failed',
            'timestamp': datetime.now().isoformat(),
            'cycle_id': cycle_id,
        }, suggestion_symbol, cycle_id=cycle_id)

        if not is_success:
            global_state.add_log(f"[🚀 EXECUTOR] Live: {action.upper()} => ❌ FAILED")
            return {'status': 'failed', 'action': action, 'details': {'error': 'execution_failed'}}

        quantity = float(order_params.get('quantity', 0) or 0)
        trade_record = {
            'open_cycle': global_state.cycle_counter,
            'close_cycle': 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'action': action.upper(),
            'symbol': suggestion_symbol,
            'entry_price': current_price,
            'quantity': quantity,
            'cost': current_price * quantity,
            'exit_price': 0,
            'pnl': 0.0,
            'confidence': order_params.get('confidence'),
            'status': 'EXECUTED',
            'cycle': cycle_id,
        }
        self.saver.save_trade(trade_record)
        global_state.trade_history.insert(0, trade_record)
        if len(global_state.trade_history) > 50:
            global_state.trade_history.pop()
        global_state.cycle_positions_opened += 1
        global_state.add_log(f"[🚀 EXECUTOR] Live: {action.upper()} {quantity} => ✅ SENT")
        return {'status': 'success', 'action': action, 'details': order_params, 'current_price': current_price}
    
    def _execute_order(self, order_params: Dict) -> bool:
        """
        执行订单
        
        Args:
            order_params: 订单参数
        
        Returns:
            是否成功
        """
        try:
            current_pos = self._get_current_position()
            pos_side = current_pos.side if current_pos else None
            action = normalize_action(order_params.get('action'), position_side=pos_side)
            order_params['action'] = action

            if is_passive_action(action):
                return True

            # 设置杠杆
            self.client.set_leverage(
                symbol=self.symbol_manager.current_symbol,
                leverage=order_params['leverage']
            )
            
            # 市价开仓
            if action == 'open_long':
                side = 'BUY'
            elif action == 'open_short':
                side = 'SELL'
            elif action == 'close_long':
                side = 'SELL'
            elif action == 'close_short':
                side = 'BUY'
            else:
                return False
            order = self.client.place_futures_market_order(
                symbol=self.symbol_manager.current_symbol,
                side=side,
                quantity=order_params['quantity']
            )
            
            if not order:
                return False
            
            # 仅开仓动作设置止损止盈
            if action in ('open_long', 'open_short'):
                self.execution_engine.set_stop_loss_take_profit(
                    symbol=self.symbol_manager.current_symbol,
                    position_side='LONG' if action == 'open_long' else 'SHORT',
                    stop_loss=order_params['stop_loss'],
                    take_profit=order_params['take_profit']
                )
            
            return True
            
        except Exception as e:
            log.error(f"Order execution failed: {e}", exc_info=True)
            return False

    def _format_agent_output_for_context(self, key: str, value: Any) -> str:
        import json
        try:
            payload = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            payload = str(value)
        if len(payload) > 800:
            payload = payload[:800] + "...(truncated)"
        return f"- {key}: {payload}\n"

    def _collect_selected_agent_outputs(
        self,
        predict_result: PredictResult = None,
        reflection_text: Optional[str] = None
    ) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}

        if getattr(self.agent_config, 'symbol_selector_agent', False):
            if getattr(global_state, 'symbol_selector', None):
                outputs['symbol_selector'] = global_state.symbol_selector

        if getattr(self.agent_config, 'predict_agent', False) and predict_result:
            try:
                outputs['predict_agent'] = asdict(predict_result)
            except Exception:
                outputs['predict_agent'] = getattr(predict_result, '__dict__', str(predict_result))

        semantic = getattr(global_state, 'semantic_analyses', {}) or {}

        if self.agent_config.trend_agent_llm or self.agent_config.trend_agent_local:
            if 'trend' in semantic:
                outputs['trend_agent'] = semantic.get('trend')
        if self.agent_config.setup_agent_llm or self.agent_config.setup_agent_local:
            if 'setup' in semantic:
                outputs['setup_agent'] = semantic.get('setup')
        if self.agent_config.trigger_agent_llm or self.agent_config.trigger_agent_local:
            if 'trigger' in semantic:
                outputs['trigger_agent'] = semantic.get('trigger')

        if self.agent_config.reflection_agent_llm or self.agent_config.reflection_agent_local:
            outputs['reflection_agent'] = {
                'count': getattr(global_state, 'reflection_count', 0),
                'text': reflection_text,
                'raw': getattr(global_state, 'last_reflection', None)
            }

        multi_period = getattr(global_state, 'multi_period_result', {}) or {}
        if multi_period:
            outputs['multi_period_agent'] = multi_period

        return outputs
    
    def _build_market_context(
        self,
        quant_analysis: Dict,
        predict_result,
        market_data: Dict,
        regime_info: Dict = None,
        position_info: Dict = None,
        selected_agent_outputs: Optional[Dict] = None
    ) -> str:
        """
        构建 DeepSeek LLM 所需的市场上下文文本
        """
        # 提取关键数据
        current_price = market_data['current_price']
        
        # 格式化趋势分析
        trend = quant_analysis.get('trend', {})
        trend_details = trend.get('details', {})
        
        oscillator = quant_analysis.get('oscillator', {})
        
        sentiment = quant_analysis.get('sentiment', {})
        
        # Prophet 预测 (语义化转换)
        prob_pct = (predict_result.probability_up if predict_result else 0.5) * 100
        prophet_signal = predict_result.signal if predict_result else "neutral"
        
        # 语义转换逻辑 (Prophet)
        if prob_pct >= 80:
            prediction_desc = f"Strong Uptrend Forecast (High Probability of Rising > 80%, Value: {prob_pct:.1f}%)"
        elif prob_pct >= 60:
            prediction_desc = f"Bullish Bias (Likely to Rise 60-80%, Value: {prob_pct:.1f}%)"
        elif prob_pct <= 20:
            prediction_desc = f"Strong Downtrend Forecast (High Probability of Falling > 80%, Value: {prob_pct:.1f}%)"
        elif prob_pct <= 40:
            prediction_desc = f"Bearish Bias (Likely to Fall 60-80%, Value: {prob_pct:.1f}%)"
        else:
            prediction_desc = f"Uncertain/Neutral (40-60%, Value: {prob_pct:.1f}%)"

        # 语义化转换 (Technical Indicators)
        t_score_total = trend.get('total_trend_score')  # Default to None
        t_semantic = SemanticConverter.get_trend_semantic(t_score_total)
        # Individual Trend Scores
        t_1h_score = trend.get('trend_1h_score') 
        t_15m_score = trend.get('trend_15m_score')
        t_5m_score = trend.get('trend_5m_score')
        t_1h_sem = SemanticConverter.get_trend_semantic(t_1h_score)
        t_15m_sem = SemanticConverter.get_trend_semantic(t_15m_score)
        t_5m_sem = SemanticConverter.get_trend_semantic(t_5m_score)
        
        o_score_total = oscillator.get('total_osc_score')
        o_semantic = SemanticConverter.get_oscillator_semantic(o_score_total)
        
        s_score_total = sentiment.get('total_sentiment_score')
        s_semantic = SemanticConverter.get_sentiment_score_semantic(s_score_total)

        rsi_15m = oscillator.get('oscillator_15m', {}).get('details', {}).get('rsi_value')
        rsi_1h = oscillator.get('oscillator_1h', {}).get('details', {}).get('rsi_value')
        rsi_15m_semantic = SemanticConverter.get_rsi_semantic(rsi_15m)
        rsi_1h_semantic = SemanticConverter.get_rsi_semantic(rsi_1h)
        
        # MACD
        macd_15m = trend.get('details', {}).get('15m_macd_diff')
        macd_semantic = SemanticConverter.get_macd_semantic(macd_15m)
        
        oi_change = sentiment.get('oi_change_24h_pct', 0)
        oi_semantic = SemanticConverter.get_oi_change_semantic(oi_change)
        
        # 市场状态与价格位置
        regime_type = "Unknown"
        regime_confidence = 0
        price_position = "Unknown"
        price_position_pct = 50
        if regime_info:
            regime_type = regime_info.get('regime', 'unknown')
            regime_confidence = regime_info.get('confidence', 0)
            position_info_regime = regime_info.get('position', {})
            price_position = position_info_regime.get('location', 'unknown')
            price_position_pct = position_info_regime.get('position_pct', 50)
        
        # Helper to format values safely
        def fmt_val(val, fmt="{:.2f}"):
            return fmt.format(val) if val is not None else "N/A"

            
        # 构建持仓信息文本 (New)
        position_section = ""
        if position_info:
            side_icon = "🟢" if position_info['side'] == 'LONG' else "🔴"
            pnl_icon = "💰" if position_info['unrealized_pnl'] > 0 else "💸"
            position_section = f"""
## 💼 CURRENT POSITION STATUS (Virtual Sub-Agent Logic)
> ⚠️ CRITICAL: YOU ARE HOLDING A POSITION. EVALUATE EXIT CONDITIONS FIRST.

- **Status**: {side_icon} {position_info['side']}
- **Entry Price**: ${position_info['entry_price']:,.2f}
- **Current Price**: ${current_price:,.2f}
- **PnL**: {pnl_icon} ${position_info['unrealized_pnl']:.2f} ({position_info['pnl_pct']:+.2f}%)
- **Quantity**: {position_info['quantity']}
- **Leverage**: {position_info['leverage']}x

**EXIT JUDGMENT INSTRUCTION**:
1. **Trend Reversal**: If current trend contradicts position side (e.g. Long but Trend turned Bearish), consider CLOSE.
2. **Profit/Risk**: Check if PnL is satisfactory or risk is increasing.
3. **If Closing**:
   - current side is LONG -> return `close_long`
   - current side is SHORT -> return `close_short`
"""
        
        context = f"""
## 1. Snapshot
- Symbol: {self.symbol_manager.current_symbol}
- Price: ${current_price:,.2f}
"""

        if position_section:
            context += f"\n{position_section}\n"

        context += "\n## 2. Four-Layer Status\n"
        # Build four-layer status summary with smart grouping
        blocking_reason = global_state.four_layer_result.get('blocking_reason', 'None')
        layer1_pass = global_state.four_layer_result.get('layer1_pass')
        layer2_pass = global_state.four_layer_result.get('layer2_pass')
        layer3_pass = global_state.four_layer_result.get('layer3_pass')
        layer4_pass = global_state.four_layer_result.get('layer4_pass')
        
        layer_status = []
        
        # Smart grouping: if both Layer 1 and 2 fail with same reason, merge them
        if not layer1_pass and not layer2_pass:
            layer_status.append(f"❌ **Layers 1-2 BLOCKED**: {blocking_reason}")
        else:
            if layer1_pass:
                layer_status.append("✅ **Trend/Fuel**: PASS")
            else:
                layer_status.append(f"❌ **Trend/Fuel**: FAIL - {blocking_reason}")
            
            if layer2_pass:
                layer_status.append("✅ **AI Filter**: PASS")
            else:
                layer_status.append(f"❌ **AI Filter**: VETO - {blocking_reason}")
        
        # Layer 3 & 4
        layer_status.append(f"{'✅' if layer3_pass else '⏳'} **Setup (15m)**: {'READY' if layer3_pass else 'WAIT'}")
        layer_status.append(f"{'✅' if layer4_pass else '⏳'} **Trigger (5m)**: {'CONFIRMED' if layer4_pass else 'WAITING'}")
        
        # Add risk adjustment
        tp_mult = global_state.four_layer_result.get('tp_multiplier', 1.0)
        sl_mult = global_state.four_layer_result.get('sl_multiplier', 1.0)
        if tp_mult != 1.0 or sl_mult != 1.0:
            layer_status.append(f"⚖️ **Risk Adjustment**: TP x{tp_mult} | SL x{sl_mult}")
        
        context += "\n".join(layer_status)
        
        # Add data anomaly warning
        if global_state.four_layer_result.get('data_anomalies'):
            anomalies = ', '.join(global_state.four_layer_result.get('data_anomalies', []))
            context += f"\n\n⚠️ **DATA ANOMALY**: {anomalies}"

        # Multi-Period Parser Summary (only if selected)
        multi_period = None
        if selected_agent_outputs and selected_agent_outputs.get('multi_period_agent'):
            multi_period = selected_agent_outputs.get('multi_period_agent') or {}
        if multi_period:
            trend_scores = multi_period.get('trend_scores', {}) or {}
            four_layer = multi_period.get('four_layer', {}) or {}
            layer_pass = four_layer.get('layer_pass', {}) or {}
            context += "\n\n## 3. Multi-Period\n"
            context += (
                f"- Alignment: {multi_period.get('alignment_reason', 'N/A')}\n"
                f"- Bias: {multi_period.get('bias', 'N/A')}\n"
                f"- Trend(1h/15m/5m): "
                f"{trend_scores.get('trend_1h', 0):+.0f}/"
                f"{trend_scores.get('trend_15m', 0):+.0f}/"
                f"{trend_scores.get('trend_5m', 0):+.0f}\n"
                f"- Four-Layer: {four_layer.get('final_action', 'WAIT')} "
                f"(L1:{'Y' if layer_pass.get('L1') else 'N'}, "
                f"L2:{'Y' if layer_pass.get('L2') else 'N'}, "
                f"L3:{'Y' if layer_pass.get('L3') else 'N'}, "
                f"L4:{'Y' if layer_pass.get('L4') else 'N'})"
            )

        # Selected agent outputs (explicitly inject for Decision Core)
        if selected_agent_outputs:
            context += "\n\n## 4. Enabled Agent Outputs (Compact)\n"
            for key, val in selected_agent_outputs.items():
                context += self._format_agent_output_for_context(key, val)

        context += "\n\n## 5. Market Summary\n"
        
        # Extract analysis results (respect selected agents)
        trend_result = {}
        setup_result = {}
        trigger_result = {}
        if selected_agent_outputs:
            trend_result = selected_agent_outputs.get('trend_agent', {})
            setup_result = selected_agent_outputs.get('setup_agent', {})
            trigger_result = selected_agent_outputs.get('trigger_agent', {})
        else:
            trend_result = getattr(global_state, 'semantic_analyses', {}).get('trend', {})
            setup_result = getattr(global_state, 'semantic_analyses', {}).get('setup', {})
            trigger_result = getattr(global_state, 'semantic_analyses', {}).get('trigger', {})
        
        if isinstance(trend_result, dict):
            trend_stance = trend_result.get('stance', 'UNKNOWN')
            trend_meta = trend_result.get('metadata', {})
            trend_line = f"- Trend: {trend_stance} | Strength={trend_meta.get('strength', 'N/A')} | ADX={trend_meta.get('adx', 'N/A')}"
        else:
            trend_line = "- Trend: N/A"

        if isinstance(setup_result, dict):
            setup_stance = setup_result.get('stance', 'UNKNOWN')
            setup_meta = setup_result.get('metadata', {})
            setup_line = f"- Setup: {setup_stance} | Zone={setup_meta.get('zone', 'N/A')} | KDJ={setup_meta.get('kdj_j', 'N/A')} | MACD={setup_meta.get('macd_signal', 'N/A')}"
        else:
            setup_line = "- Setup: N/A"

        if isinstance(trigger_result, dict):
            trigger_stance = trigger_result.get('stance', 'UNKNOWN')
            trigger_meta = trigger_result.get('metadata', {})
            trigger_line = f"- Trigger: {trigger_stance} | Pattern={trigger_meta.get('pattern', 'NONE')} | RVOL={trigger_meta.get('rvol', 'N/A')}x"
        else:
            trigger_line = "- Trigger: N/A"

        context += f"{trend_line}\n{setup_line}\n{trigger_line}\n"
        
        # Note: Market Regime and Price Position are already calculated by TREND and SETUP agents
        # and included in their respective analyses above, so we don't duplicate them here.
        
        return context

# ... locating where vote_result is processed to add semantic analysis

    def run_once(self) -> Dict:
        """运行一次交易循环（同步包装）"""
        result = asyncio.run(self.run_trading_cycle())
        self._display_recent_trades()
        return result

    def _display_recent_trades(self):
        """显示最近的交易记录 (增强版表格)"""
        trades = self.saver.get_recent_trades(limit=10)
        if not trades:
            return
            
        print("\n" + "─"*100)
        print("📜 最近 10 次成交审计 (The Executor History)")
        print("─"*100)
        header = f"{'时间':<12} | {'币种':<8} | {'方向':<10} | {'成交价':<10} | {'成本':<10} | {'卖出价':<10} | {'盈亏':<10} | {'状态'}"
        print(header)
        print("─"*100)
        
        for t in trades:
            # 简化时间
            fmt_time = str(t.get('record_time', 'N/A'))[5:16]
            symbol = t.get('symbol', 'BTC')[:7]
            action = t.get('action', 'N/A')
            price = f"${float(t.get('price', 0)):,.1f}"
            cost = f"${float(t.get('cost', 0)):,.1f}"
            exit_p = f"${float(t.get('exit_price', 0)):,.1f}" if float(t.get('exit_price', 0)) > 0 else "-"
            
            pnl_val = float(t.get('pnl', 0))
            pnl_str = f"{'+' if pnl_val > 0 else ''}${pnl_val:,.2f}" if pnl_val != 0 else "-"
            
            status = t.get('status', 'N/A')
            
            row = f"{fmt_time:<12} | {symbol:<8} | {action:<10} | {price:<10} | {cost:<10} | {exit_p:<10} | {pnl_str:<10} | {status}"
            print(row)
        print("─"*100)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'risk_audit': self.risk_audit.get_audit_report(),
        }
        # DeepSeek 模式下没有 decision_core
        if hasattr(self, 'strategy_engine'):
            # self.strategy_engine 目前没有 get_statistics 方法，但可以返回基本信息
            stats['strategy_engine'] = {
                'provider': self.strategy_engine.provider,
                'model': self.strategy_engine.model
            }
        return stats

    def start_account_monitor(self):
        """Start a background thread to monitor account equity in real-time"""
        def _monitor():
            if self.test_mode:
                log.info("💰 Account Monitor Thread: Disabled in Test Mode")
                return
                
            log.info("💰 Account Monitor Thread Started")
            while True:
                # Check Control State
                if global_state.execution_mode == "Stopped":
                    break
                
                # We update even if Paused, to see PnL of open positions
                try:
                    acc = self.client.get_futures_account()
                    
                    wallet = float(acc.get('total_wallet_balance', 0))
                    pnl = float(acc.get('total_unrealized_profit', 0))
                    avail = float(acc.get('available_balance', 0))
                    equity = wallet + pnl
                    
                    global_state.update_account(equity, avail, wallet, pnl)
                    global_state.record_account_success()  # Track success
                except Exception as e:
                    log.error(f"Account Monitor Error: {e}")
                    global_state.record_account_failure()  # Track failure
                    global_state.add_log(f"❌ Account info fetch failed: {str(e)}")  # Dashboard log
                    time.sleep(5) # Backoff on error
                
                time.sleep(3) # Update every 3 seconds

        t = threading.Thread(target=_monitor, daemon=True)
        t.start()

    def run_continuous(self, interval_minutes: int = 3, headless: bool = False):
        """
        持续运行模式
        
        Args:
            interval_minutes: 运行间隔（分钟）
            headless: 是否为无头模式（不使用 Web Dashboard，在终端显示）
        """
        log.info(f"🚀 Starting continuous mode (interval: {interval_minutes}min)")
        global_state.is_running = True
        
        # 🖥️ Headless Mode: Initialize terminal display and configure logging
        self._headless_mode = headless
        if headless:
            from src.cli.terminal_display import get_display
            self._terminal_display = get_display(self.symbol_manager.symbols)
            self._terminal_display.print_header(test_mode=self.test_mode)
            
            # Configure minimal logging for headless mode using a custom filter
            # This ensures Web Dashboard mode is not affected
            import logging
            
            class HeadlessFilter(logging.Filter):
                """Filter to suppress verbose logs only in headless mode"""
                def filter(self, record):
                    # Only suppress INFO level logs from specific modules
                    if record.levelno == logging.INFO:
                        suppressed_modules = [
                            'src.features.technical_features',
                            'src.utils.logger',
                            'src.agents.data_sync_agent',
                            'src.agents.trend_agent',
                            'src.agents.setup_agent',
                            'src.agents.trigger_agent',
                            'src.strategy.llm_engine',
                            'src.models.prophet_model',
                            'src.server.state',
                            '__main__'
                        ]
                        return record.name not in suppressed_modules
                    return True  # Allow WARNING and above
            
            # Add filter to root logger
            headless_filter = HeadlessFilter()
            logging.getLogger().addFilter(headless_filter)
            
            # Store filter reference for cleanup
            self._headless_filter = headless_filter
        
        # Logger is configured in src.utils.logger, no need to override here.
        # Dashboard logging is handled via global_state.add_log -> log.bind(dashboard=True)

        # Start Real-time Monitors
        self.start_account_monitor()
        
        # 🔮 启动 Prophet 自动训练器 (每 2 小时重新训练)
        from src.models.prophet_model import HAS_LIGHTGBM
        if HAS_LIGHTGBM and self.agent_config.predict_agent:
            self.predict_agents_provider.start_auto_trainer()
        
        # 设置初始间隔 (优先使用 CLI 参数，后续 API 可覆盖)
        global_state.cycle_interval = interval_minutes
        
        log.info(f"🚀 Starting continuous trading mode (interval: {global_state.cycle_interval}m)")
        
        # 🧪 Test Mode: Initialize Virtual Account for Chart
        if self.test_mode:
            log.info("🧪 Test Mode: Initializing Virtual Account...")
            initial_balance = global_state.virtual_initial_balance
            current_balance = global_state.virtual_balance
            global_state.init_balance(current_balance, initial_balance=initial_balance)  # Initialize balance tracking
            global_state.update_account(
                equity=current_balance,
                available=current_balance,
                wallet=current_balance,
                pnl=current_balance - initial_balance
            )
        
        try:
            while global_state.is_running:
                # 🔄 Check for configuration changes
                # Method 1: .env file changed (Local mode)
                if self._env_exists:
                    try:
                        current_mtime = os.path.getmtime(self._env_path)
                        if current_mtime > self._env_mtime:
                            if self._env_mtime > 0: # Avoid reload on first pass as it's already loaded
                                log.info("📝 .env file change detected, reloading symbols...")
                                self._reload_symbols()
                            self._env_mtime = current_mtime
                    except Exception as e:
                        log.warning(f"Error checking .env mtime: {e}")
                
                # Method 2: Runtime config changed (Railway mode)
                if global_state.config_changed:
                    log.info("⚙️ Runtime config change detected, reloading symbols...")
                    self._reload_symbols()
                    # Reload LLM engine after config updates
                    try:
                        if hasattr(self, "strategy_engine"):
                            self.strategy_engine.reload_config()
                            self._update_llm_metadata()
                    except Exception as e:
                        log.warning(f"⚠️ Failed to reload LLM engine: {e}")
                    # Re-evaluate agent config from env/config on runtime updates
                    from src.agents.agent_config import AgentConfig
                    refreshed = AgentConfig.from_dict({'agents': self.config.get('agents', {})})
                    refreshed_map = refreshed.get_enabled_agents()
                    if refreshed_map != self._last_agent_config:
                        log.info(f"🔧 Runtime agent config refreshed: {refreshed_map}")
                        self._apply_agent_config(refreshed_map)
                    global_state.config_changed = False  # Reset flag
                
                runtime_agents = getattr(global_state, 'agent_config', None)
                if runtime_agents and runtime_agents != self._last_agent_config:
                    log.info(f"🔧 Runtime agent config updated: {runtime_agents}")
                    self._apply_agent_config(runtime_agents)

                # Check stop state FIRST - must break before continue
                if global_state.execution_mode == 'Stopped':
                    # Fix: Do not break, just wait.
                    if not hasattr(self, '_stop_logged') or not self._stop_logged:
                        print("\n⏹️ System stopped (waiting for start)")
                        global_state.add_log("⏹️ System STOPPED - Waiting for Start...")
                        self._stop_logged = True
                    time.sleep(1)
                    continue
                else:
                    self._stop_logged = False
                
                # Check pause state - continue waiting
                if global_state.execution_mode == 'Paused':
                    # 首次进入暂停时打印日志
                    if not hasattr(self, '_pause_logged') or not self._pause_logged:
                        print("\n⏸️ System paused, waiting to resume...")
                        global_state.add_log("⏸️ System PAUSED - waiting for resume...")
                        self._pause_logged = True
                    time.sleep(1)
                    continue
                else:
                    self._pause_logged = False  # 重置暂停日志标记

                # ✅ 统一周期计数: 在遍历币种前递增一次
                global_state.cycle_counter += 1
                cycle_num = global_state.cycle_counter
                cycle_id = f"cycle_{cycle_num:04d}_{int(time.time())}"
                global_state.current_cycle_id = cycle_id
                cycle_start_ts = datetime.now().isoformat()
                cycle_traded = False
                cycle_trade_symbol = None
                cycle_trade_action = None
                cycle_trade_status = None

                # 🧹 Clear chatroom messages each cycle (show current cycle only)
                global_state.clear_agent_messages()
                global_state.clear_agent_events()

                # 🧪 Test Mode: reset per-cycle baseline for PnL display
                if self.test_mode:
                    baseline = global_state.account_overview.get('total_equity', 0)
                    if not baseline:
                        unrealized = sum(
                            float(pos.get('unrealized_pnl', 0) or 0)
                            for pos in global_state.virtual_positions.values()
                        )
                        baseline = global_state.virtual_balance + unrealized
                    global_state.virtual_initial_balance = baseline
                    global_state.initial_balance = baseline

                # 🧹 Clear initialization logs when Cycle 1 starts (sync with Recent Decisions)
                if cycle_num == 1:
                    global_state.clear_init_logs()

                # 🔒 Position lock: if any active position exists, lock analysis to it.
                active_symbols = self.symbol_manager.get_active_position_symbols()
                locked_symbols = [s for s in self.symbol_manager.symbols if s in active_symbols]
                if active_symbols and not locked_symbols:
                    locked_symbols = sorted(set(active_symbols))
                has_lock = bool(locked_symbols)

                # 🔝 Symbol Selector Agent: run once at startup, then every 10 minutes during wait
                if not has_lock:
                    self.symbol_manager.run_symbol_selector(reason="startup", check_for_startup_done=True)

                symbols_for_cycle = locked_symbols if has_lock else self.symbol_manager.symbols
                if has_lock:
                    self.symbol_manager.current_symbol = symbols_for_cycle[0]
                    global_state.add_log(f"[🔒 SYSTEM] Active position lock: {', '.join(symbols_for_cycle)}")

                # 🧪 Test Mode: Record start of cycle account state (for Net Value Curve)
                if self.test_mode:
                    # Re-log current state with new cycle number so chart shows start of cycle
                    global_state.update_account(
                        equity=global_state.account_overview['total_equity'],
                        available=global_state.account_overview['available_balance'],
                        wallet=global_state.account_overview['wallet_balance'],
                        pnl=global_state.account_overview['total_pnl']
                    )
                
                # 🖥️ Headless Mode: Use terminal display
                if self._headless_mode:
                    self._terminal_display.print_cycle_start(cycle_num, symbols_for_cycle)
                else:
                    print(f"\n{'='*80}")
                    print(f"🔄 Cycle #{cycle_num} | 分析 {len(symbols_for_cycle)} 个交易对")
                    print(f"{'='*80}")
                global_state.add_log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                global_state.add_log(f"[📊 SYSTEM] Cycle #{cycle_num} | {', '.join(symbols_for_cycle)}")

                # 🎯 重置周期开仓计数器
                global_state.cycle_positions_opened = 0
                
                # 🔄 多币种顺序处理: 依次分析每个交易对
                # Step 1: 收集所有交易对的决策
                all_decisions: List[SuggestedTrade] = []
                latest_prices = {}  # Store latest prices for PnL calculation
                for symbol in symbols_for_cycle:
                    self.symbol_manager.current_symbol = symbol  # 设置当前处理的交易对
                    
                    # Analyze each symbol first without executing OPEN actions
                    result = asyncio.run(self.run_trading_cycle(analyze_only=True))
                    
                    latest_prices[symbol] = global_state.current_price.get(symbol, 0)
                    
                    print(f"  [{symbol}] 结果: {result['status']}")
                    
                    # Collect viable open opportunities
                    suggested_trade = SuggestedTrade.from_cycle_result(symbol=symbol, result=result)
                    if suggested_trade:
                        all_decisions.append(suggested_trade)
                
                # Step 2: 从所有开仓决策中选择信心度最高的一个
                if all_decisions:
                    # 按信心度排序
                    all_decisions.sort(key=lambda x: x.confidence, reverse=True)
                    best_decision = all_decisions[0]
                    
                    print(f"\n🎯 本周期最优开仓机会: {best_decision.symbol} (信心度: {best_decision.confidence:.1f}%)")
                    global_state.add_log(f"[🎯 SYSTEM] Best: {best_decision.symbol} (Conf: {best_decision.confidence:.1f}%)")
                    
                    # 只执行最优的一个（直接执行已审计建议，避免重复跑完整流程）
                    try:
                        self.symbol_manager.current_symbol = best_decision.symbol
                        exec_result = self._execute_suggested_open_trade(
                            symbol=self.symbol_manager.current_symbol,
                            suggested=best_decision,
                            cycle_id=cycle_id
                        )
                        exec_action = exec_result.get('action', 'unknown')
                        exec_status = exec_result.get('status', 'unknown')
                        if exec_action and str(exec_action).lower() != 'unknown' and not is_passive_action(exec_action):
                            cycle_traded = exec_status == 'success'
                            cycle_trade_symbol = self.symbol_manager.current_symbol
                            cycle_trade_action = exec_action
                            cycle_trade_status = exec_status
                        global_state.add_log(
                            f"[🎯 SYSTEM] Executed: {self.symbol_manager.current_symbol} {exec_action} ({exec_status})"
                        )
                    except Exception as e:
                        log.error(f"❌ Best decision execution failed: {e}", exc_info=True)
                        global_state.add_log(f"[🎯 SYSTEM] Execution failed: {e}")
                    
                    # 如果有其他开仓机会被跳过，记录下来
                    if len(all_decisions) > 1:
                        skipped = [f"{d.symbol}({d.confidence:.1f}%)" for d in all_decisions[1:]]
                        print(f"  ⏭️  跳过其他机会: {', '.join(skipped)}")
                        global_state.add_log(f"⏭️  Skipped opportunities: {', '.join(skipped)} (1 position per cycle limit)")
                
                        global_state.add_log(f"⏭️  Skipped opportunities: {', '.join(skipped)} (1 position per cycle limit)")
                
                # 💰 Update Virtual Account PnL (Mark-to-Market)
                if self.test_mode:
                    self._update_virtual_account_stats(latest_prices)
                
                # 🖥️ Headless Mode: Print account summary after each cycle
                if self._headless_mode:
                    acc = global_state.account_overview
                    # Get current positions
                    positions = global_state.virtual_positions if self.test_mode else {}
                    self._terminal_display.print_account_summary(
                        equity=acc['total_equity'],
                        available=acc['available_balance'],
                        pnl=acc['total_pnl'],
                        initial=global_state.initial_balance,
                        cycle=global_state.cycle_counter,
                        positions=positions,
                        symbols=symbols_for_cycle
                    )

                # 📋 Persist cycle summary to DB
                self._record_cycle_summary(
                    cycle_number=cycle_num,
                    cycle_id=cycle_id,
                    timestamp_start=cycle_start_ts,
                    timestamp_end=datetime.now().isoformat(),
                    symbols=symbols_for_cycle,
                    traded=cycle_traded,
                    trade_symbol=cycle_trade_symbol,
                    trade_action=cycle_trade_action,
                    trade_status=cycle_trade_status
                )
                
                # Dynamic Interval: specific to new requirement
                current_interval = global_state.cycle_interval
                
                # 等待下一次检查
                if self._headless_mode:
                    self._terminal_display.print_waiting(current_interval)
                else:
                    print(f"\n⏳ 等待 {current_interval} 分钟...")
                
                # Sleep in chunks to allow responsive PAUSE/STOP and INTERVAL changes
                # Check every 1 second during the wait interval
                elapsed_seconds = 0
                while True:
                    # 每秒检查当前间隔设置 (支持动态调整)
                    current_interval = global_state.cycle_interval
                    wait_seconds = current_interval * 60

                    # Run symbol selector on schedule (every 10 minutes, skip if holding positions)
                    # Note: symbol_manager.run_symbol_selector also has internal position check for safety
                    has_positions = bool(self.symbol_manager.get_active_position_symbols())
                    if ((time.time() - self.selector_last_run) >= self.selector_interval_sec
                            and global_state.execution_mode == "Running"
                            and not has_positions):
                        self.symbol_manager.run_symbol_selector(reason="scheduled")
                    
                    # 如果已经等待足够时间，结束等待
                    if elapsed_seconds >= wait_seconds:
                        break
                    
                    # 检查执行模式
                    if global_state.execution_mode != "Running":
                        break
                    
                    # Heartbeat every 60s
                    if elapsed_seconds > 0 and elapsed_seconds % 60 == 0:
                        remaining = int((wait_seconds - elapsed_seconds) / 60)
                        if remaining > 0:
                            print(f"⏳ Next cycle in {remaining}m...")
                            global_state.add_log(f"[📊 SYSTEM] Waiting next cycle... ({remaining}m)")

                    time.sleep(1)
                    elapsed_seconds += 1
                
        except KeyboardInterrupt:
            if self._headless_mode:
                # Display shutdown summary
                stats = {
                    'cycles': global_state.cycle_counter,
                    'trades': len(global_state.trade_history),
                    'total_pnl': global_state.account_overview.get('total_pnl', 0)
                }
                self._terminal_display.print_shutdown(stats)
                
                # Clean up headless filter
                import logging
                if hasattr(self, '_headless_filter'):
                    logging.getLogger().removeFilter(self._headless_filter)
            else:
                print(f"\n\n⚠️  收到停止信号，退出...")
            global_state.is_running = False

    def _update_virtual_account_stats(self, latest_prices: Dict[str, float]):
        """
        [Test Mode] 更新虚拟账户统计 (权益、PnL) 并推送到 Global State
        """
        if not self.test_mode:
            return

        total_unrealized_pnl = 0.0
        
        # 遍历持仓计算未实现盈亏
        for symbol, pos in global_state.virtual_positions.items():
            current_price = latest_prices.get(symbol)
            if not current_price:
                 # Fallback to stored price if current not available
                 current_price = pos.get('current_price', pos['entry_price'])
                
            entry_price = pos['entry_price']
            quantity = pos['quantity']
            side = pos['side']  # LONG or SHORT
            
            # PnL Calc
            if side.upper() == 'LONG':
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
                
            pos['unrealized_pnl'] = pnl
            pos['current_price'] = current_price
            total_unrealized_pnl += pnl

        # 更新权益
        # Equity = Balance (Realized) + Unrealized PnL
        total_equity = global_state.virtual_balance + total_unrealized_pnl
        
        # 计算真实总盈亏 (相比初始资金)
        # Total PnL = Current Equity - Initial Balance
        real_total_pnl = total_equity - global_state.virtual_initial_balance
        
        # 更新 Global State
        global_state.update_account(
            equity=total_equity,
            available=global_state.virtual_balance,
            wallet=global_state.virtual_balance,
            pnl=real_total_pnl  # ✅ Fix: Pass total profit/loss from start
        )

    def _save_virtual_state(self):
        """Helper to persist virtual account state"""
        if self.test_mode:
            self.saver.save_virtual_account(
                balance=global_state.virtual_balance,
                positions=global_state.virtual_positions
            )
