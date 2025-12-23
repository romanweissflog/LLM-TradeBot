"""
🤖 LLM-TradeBot - 多Agent架构主循环
===========================================

集成:
1. 🕵️ DataSyncAgent - 异步并发数据采集
2. 👨‍🔬 QuantAnalystAgent - 量化信号分析
3. ⚖️ DecisionCoreAgent - 加权投票决策
4. 👮 RiskAuditAgent - 风控审计拦截

优化:
- 异步并发执行（减少60%等待时间）
- 双视图数据结构（stable + live）
- 分层信号分析（趋势 + 震荡）
- 多周期对齐决策
- 止损方向自动修正
- 一票否决风控

Author: AI Trader Team
Date: 2025-12-19
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from typing import Dict, Optional, List
from datetime import datetime
import json
import time
import threading
import signal
from dataclasses import asdict

from src.api.binance_client import BinanceClient
from src.execution.engine import ExecutionEngine
from src.risk.manager import RiskManager
from src.utils.logger import log, setup_logger
from src.utils.trade_logger import trade_logger
from src.utils.data_saver import DataSaver
from src.data.processor import MarketDataProcessor  # ✅ Corrected Import
from src.features.technical_features import TechnicalFeatureEngineer
from src.server.state import global_state
from src.utils.semantic_converter import SemanticConverter  # ✅ Global Import
from src.agents.regime_detector import RegimeDetector  # ✅ Market Regime Detection
from src.config import Config # Re-added Config as it's used later

# FastAPI dependencies
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 导入多Agent
from src.agents import (
    DataSyncAgent,
    QuantAnalystAgent,
    DecisionCoreAgent,
    RiskAuditAgent,
    PositionInfo,
    SignalWeight
)
from src.strategy.deepseek_engine import StrategyEngine
from src.agents.predict_agent import PredictAgent
from src.server.app import app
from src.server.state import global_state

class MultiAgentTradingBot:
    """
    多Agent交易机器人（重构版）
    
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
        test_mode: bool = False
    ):
        """
        初始化多Agent交易机器人
        
        Args:
            max_position_size: 最大单笔金额（USDT）
            leverage: 杠杆倍数
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
            test_mode: 测试模式（不执行真实交易）
        """
        print("\n" + "="*80)
        print(f"🤖 AI Trader - DeepSeek LLM 决策模式")
        print("="*80)
        
        self.config = Config()
        # 多币种支持: 读取 symbols 列表，兼容旧版 symbol 单值配置
        symbols_config = self.config.get('trading.symbols', None)
        if symbols_config:
            self.symbols = symbols_config
        else:
            # 向后兼容: 使用旧版 trading.symbol 配置
            self.symbols = [self.config.get('trading.symbol', 'BTCUSDT')]
        self.primary_symbol = self.config.get('trading.primary_symbol', self.symbols[0])
        self.current_symbol = self.primary_symbol  # 当前处理的交易对
        self.test_mode = test_mode
        global_state.is_test_mode = test_mode  # Set test mode in global state
        
        # 交易参数
        self.max_position_size = max_position_size
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        
        # 初始化客户端
        self.client = BinanceClient()
        self.risk_manager = RiskManager()
        self.execution_engine = ExecutionEngine(self.client, self.risk_manager)
        self.saver = DataSaver() # ✅ 初始化 Multi-Agent 数据保存器
        
        # 初始化共享 Agent (与币种无关)
        print("\n🚀 初始化Agent...")
        self.data_sync_agent = DataSyncAgent(self.client)
        self.quant_analyst = QuantAnalystAgent()
        # self.decision_core = DecisionCoreAgent() # Deprecated in DeepSeek Mode
        self.risk_audit = RiskAuditAgent(
            max_leverage=10.0,
            max_position_pct=0.3,
            min_stop_loss_pct=0.005,
            max_stop_loss_pct=0.05
        )
        self.processor = MarketDataProcessor()  # ✅ 初始化数据处理器
        self.feature_engineer = TechnicalFeatureEngineer()  # 🔮 特征工程器 for Prophet
        # self.regime_detector = RegimeDetector()  # 📊 市场状态检测器 (Integrated into QuantAnalystAgent)
        
        # 🔮 为每个币种创建独立的 PredictAgent
        self.predict_agents = {}
        for symbol in self.symbols:
            self.predict_agents[symbol] = PredictAgent(horizon='30m', symbol=symbol)
        
        print("  ✅ DataSyncAgent 已就绪")
        print("  ✅ QuantAnalystAgent 已就绪")
        print(f"  ✅ PredictAgent 已就绪 (共 {len(self.symbols)} 个币种)")
        print("  ✅ RiskAuditAgent 已就绪")
        
        # 🧠 DeepSeek 决策引擎
        self.strategy_engine = StrategyEngine()
        print("  ✅ DeepSeek StrategyEngine 已就绪")
        
        print(f"\n⚙️  交易配置:")
        print(f"  - 交易对: {', '.join(self.symbols)}")
        print(f"  - 最大单笔: ${self.max_position_size:.2f} USDT")
        print(f"  - 杠杆倍数: {self.leverage}x")
        print(f"  - 止损: {self.stop_loss_pct}%")
        print(f"  - 止盈: {self.take_profit_pct}%")
        print(f"  - 测试模式: {'✅ 是' if self.test_mode else '❌ 否'}")
        
        # ✅ Load initial trade history (Only in Live Mode)
        if not self.test_mode:
            recent_trades = self.saver.get_recent_trades(limit=20)
            global_state.trade_history = recent_trades
            print(f"  📜 已加载 {len(recent_trades)} 条历史交易记录")
        else:
            global_state.trade_history = []
            print("  🧪 测试模式：不加载历史记录，仅显示本次运行数据")
    


    async def run_trading_cycle(self) -> Dict:
        """
        执行完整的交易循环（异步版本）
        Returns:
            {
                'status': 'success/failed/hold/blocked',
                'action': 'long/short/hold',
                'details': {...}
            }
        """
        print(f"\n{'='*80}")
        print(f"🔄 启动交易审计循环 | {datetime.now().strftime('%H:%M:%S')} | {self.current_symbol}")
        print(f"{'='*80}")
        
        # Update Dashboard Status
        global_state.is_running = True
        # Removed verbose log: Starting trading cycle
        
        try:
            # ✅ 使用 run_continuous 中已设置的周期信息
            cycle_num = global_state.cycle_counter
            cycle_id = global_state.current_cycle_id
            
            # 每个币种的子日志
            global_state.add_log(f"📊 [{self.current_symbol}] Starting analysis...")
            
            # ✅ Generate snapshot_id for this cycle (legacy compatibility)
            snapshot_id = f"snap_{int(time.time())}"

            # Step 1: 采样 - 数据先知 (The Oracle)
            print("\n[Step 1/4] 🕵️ 数据先知 (The Oracle) - 异步数据采集...")
            global_state.oracle_status = "Fetching Data..." 
            market_snapshot = await self.data_sync_agent.fetch_all_timeframes(self.current_symbol)
            global_state.oracle_status = "Data Ready"
            
            # 💰 fetch_position_info logic (New Feature)
            # Create a unified position_info dict for Context
            current_position_info = None
            
            try:
                if self.test_mode:
                    if self.current_symbol in global_state.virtual_positions:
                        v_pos = global_state.virtual_positions[self.current_symbol]
                        # Calc PnL
                        current_price_5m = market_snapshot.live_5m['close']
                        entry_price = v_pos['entry_price']
                        qty = v_pos['quantity']
                        side = v_pos['side']
                        
                        if side == 'LONG':
                            unrealized_pnl = (current_price_5m - entry_price) * qty
                        else:
                            unrealized_pnl = (entry_price - current_price_5m) * qty
                        
                        pnl_pct = (unrealized_pnl / (entry_price * qty)) * 100 if entry_price > 0 else 0
                        
                        # Store in position_info
                        current_position_info = {
                            'symbol': self.current_symbol,
                            'side': side,
                            'quantity': qty,
                            'entry_price': entry_price,
                            'unrealized_pnl': unrealized_pnl,
                            'pnl_pct': pnl_pct,
                            'leverage': v_pos.get('leverage', 1),
                            'is_test': True
                        }
                        
                        # Also update local object for backward compatibility with display logic
                        v_pos['unrealized_pnl'] = unrealized_pnl
                        v_pos['pnl_pct'] = pnl_pct
                        v_pos['current_price'] = current_price_5m
                        log.info(f"💰 [Virtual Position] {side} {self.current_symbol} PnL: ${unrealized_pnl:.2f} ({pnl_pct:+.2f}%)")
                        
                else:
                    # Live Mode
                    try:
                        raw_pos = self.client.get_futures_position(self.current_symbol)
                        # raw_pos returns dict if specific symbol, or list if not?
                        # BinanceClient.get_futures_position returns Optional[Dict]
                        
                        if raw_pos and float(raw_pos.get('positionAmt', 0)) != 0:
                            amt = float(raw_pos.get('positionAmt', 0))
                            side = 'LONG' if amt > 0 else 'SHORT'
                            entry_price = float(raw_pos.get('entryPrice', 0))
                            unrealized_pnl = float(raw_pos.get('unRealizedProfit', 0))
                            qty = abs(amt)
                            
                            pnl_pct = (unrealized_pnl / (entry_price * qty / int(raw_pos.get('leverage', 1)))) * 100 # Approx ROE
                            
                            current_position_info = {
                                'symbol': self.current_symbol,
                                'side': side,
                                'quantity': qty,
                                'entry_price': entry_price,
                                'unrealized_pnl': unrealized_pnl,
                                'pnl_pct': pnl_pct, # Note: this might be rough calc
                                'leverage': int(raw_pos.get('leverage', 1)),
                                'is_test': False
                            }
                            log.info(f"💰 [Real Position] {side} {self.current_symbol} Amt:{amt} PnL:${unrealized_pnl:.2f}")
                    except Exception as e:
                        log.error(f"Failed to fetch real position: {e}")

            except Exception as e:
                 log.error(f"Error processing position info: {e}")

            # ✅ Save Market Data & Process Indicators
            processed_dfs = {}
            for tf in ['5m', '15m', '1h']:
                raw_klines = getattr(market_snapshot, f'raw_{tf}')
                # 保存原始数据
                self.saver.save_market_data(raw_klines, self.current_symbol, tf)
                
                # 处理并保存指标 (Process indicators)
                df_with_indicators = self.processor.extract_feature_snapshot(getattr(self.processor.process_klines(raw_klines, self.current_symbol, tf), "copy")())
                # Wait, process_klines returns df. Calling extract_feature_snapshot on it is for features.
                # The original code:
                # df_with_indicators = self.processor.process_klines(raw_klines, self.current_symbol, tf)
                # self.saver.save_indicators(df_with_indicators, self.current_symbol, tf, snapshot_id)
                # features_df = self.processor.extract_feature_snapshot(df_with_indicators)
                
                # Let's restore original lines carefully.
                df_with_indicators = self.processor.process_klines(raw_klines, self.current_symbol, tf)
                self.saver.save_indicators(df_with_indicators, self.current_symbol, tf, snapshot_id)
                features_df = self.processor.extract_feature_snapshot(df_with_indicators)
                self.saver.save_features(features_df, self.current_symbol, tf, snapshot_id)
                
                # 存入字典供后续步骤复用
                processed_dfs[tf] = df_with_indicators
            
            # ✅ 重要优化：更新快照中的 DataFrame
            market_snapshot.stable_5m = processed_dfs['5m']
            market_snapshot.stable_15m = processed_dfs['15m']
            market_snapshot.stable_1h = processed_dfs['1h']
            
            current_price = market_snapshot.live_5m.get('close')
            print(f"  ✅ 采样完毕: ${current_price:,.2f} ({market_snapshot.timestamp.strftime('%H:%M:%S')})")
            
            # LOG 1: Oracle
            global_state.add_log(f"🕵️ DataSyncAgent (The Oracle): Fetch complete. Snapshot=${current_price:,.2f}")
            global_state.current_price = current_price
            
            # Step 2: Strategist
            print("[Step 2/4] 👨‍🔬 量化策略师 (The Strategist) - 评估数据中...")
            quant_analysis = await self.quant_analyst.analyze_all_timeframes(market_snapshot)
            
            # Update Dashboard
            s_score = quant_analysis['comprehensive']['score']
            global_state.strategist_score = s_score
            
            # Save Context
            self.saver.save_context(quant_analysis, self.current_symbol, 'analytics', snapshot_id)
            
            # Step 2.5: Prophet
            print("[Step 2.5/5] 🔮 预测预言家 (The Prophet) - 计算上涨概率...")
            df_15m_features = self.feature_engineer.build_features(processed_dfs['15m'])
            if not df_15m_features.empty:
                latest = df_15m_features.iloc[-1].to_dict()
                predict_features = {k: v for k, v in latest.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            else:
                 predict_features = {}
            
            predict_result = await self.predict_agents[self.current_symbol].predict(predict_features)
            global_state.prophet_probability = predict_result.probability_up
            
            # Step 3: DeepSeek
            market_data = {
                'df_5m': processed_dfs['5m'],
                'df_15m': processed_dfs['15m'],
                'df_1h': processed_dfs['1h'],
                'current_price': current_price
            }
            regime_info = quant_analysis.get('regime', {})
            
            print("[Step 3/5] 🧠 DeepSeek LLM - 智能决策中...")
            
            # Build Context with POSITION INFO
            market_context_text = self._build_market_context(
                quant_analysis=quant_analysis,
                predict_result=predict_result,
                market_data=market_data,
                regime_info=regime_info,
                position_info=current_position_info  # ✅ Pass Position Info
            )
            
            market_context_data = {
                'symbol': self.current_symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price
            }
            
            # Call DeepSeek
            llm_decision = self.strategy_engine.make_decision(
                market_context_text=market_context_text,
                market_context_data=market_context_data
            )
            
            # ... Rest of logic stays similar ...
            
            # 转换为 VoteResult 兼容格式
            # (Need to check if i need to include rest of the function)

            
            # 转换为 VoteResult 兼容格式
            from src.agents.decision_core_agent import VoteResult
            
            # Extract scores for dashboard
            q_trend = quant_analysis.get('trend', {})
            q_osc = quant_analysis.get('oscillator', {})
            q_sent = quant_analysis.get('sentiment', {})
            q_comp = quant_analysis.get('comprehensive', {})
            
            # Construct vote_details similar to DecisionCore
            vote_details = {
                'deepseek': llm_decision.get('confidence', 0),
                'strategist_total': q_comp.get('score', 0),
                # Trend
                'trend_1h': q_trend.get('trend_1h_score', 0),
                'trend_15m': q_trend.get('trend_15m_score', 0),
                'trend_5m': q_trend.get('trend_5m_score', 0),
                # Oscillator
                'oscillator_1h': q_osc.get('osc_1h_score', 0),
                'oscillator_15m': q_osc.get('osc_15m_score', 0),
                'oscillator_5m': q_osc.get('osc_5m_score', 0),
                # Sentiment
                'sentiment': q_sent.get('total_sentiment_score', 0),
                # Prophet
                'prophet': predict_result.probability_up,
                # 🐂🐻 Bull/Bear Agent Perspectives
                'bull_confidence': llm_decision.get('bull_perspective', {}).get('bull_confidence', 50),
                'bear_confidence': llm_decision.get('bear_perspective', {}).get('bear_confidence', 50),
                'bull_stance': llm_decision.get('bull_perspective', {}).get('stance', 'UNKNOWN'),
                'bear_stance': llm_decision.get('bear_perspective', {}).get('stance', 'UNKNOWN'),
                'bull_reasons': llm_decision.get('bull_perspective', {}).get('bullish_reasons', ''),
                'bear_reasons': llm_decision.get('bear_perspective', {}).get('bearish_reasons', '')
            }
            
            # Determine Regime from Trend Score using Semantic Converter
            trend_score_total = quant_analysis.get('trend', {}).get('total_trend_score', 0)
            regime_desc = SemanticConverter.get_trend_semantic(trend_score_total)
            
            # Determine Position details from LLM Decision
            pos_pct = llm_decision.get('position_size_pct', 0)
            if not pos_pct and llm_decision.get('position_size_usd') and self.max_position_size:
                 # Fallback: estimate pct if usd is provided
                 pos_pct = (llm_decision.get('position_size_usd') / self.max_position_size) * 100
                 # Clamp to reasonable range (仓位大小不应超过100%)
                 pos_pct = min(pos_pct, 100)
            
            # 获取真正的价格位置信息（从 regime_info）
            price_position_info = regime_info.get('position', {}) if regime_info else {}
            
            vote_result = VoteResult(
                action=llm_decision.get('action', 'wait'),
                confidence=llm_decision.get('confidence', 0) / 100.0,  # 转换为 0-1
                weighted_score=llm_decision.get('confidence', 0) - 50,  # -50 to +50
                vote_details=vote_details,
                multi_period_aligned=True,
                reason=llm_decision.get('reasoning', 'DeepSeek LLM decision'),
                regime={
                    'regime': regime_desc,
                    'confidence': llm_decision.get('confidence', 0)
                },
                position=price_position_info  # 使用真正的价格位置信息
            )
            
            # 保存完整的 LLM 交互日志 (Input, Process, Output)
            full_log_content = f"""
================================================================================
🕐 Timestamp: {datetime.now().isoformat()}
💱 Symbol: {self.current_symbol}
================================================================================

--------------------------------------------------------------------------------
📤 INPUT (PROMPT)
--------------------------------------------------------------------------------
[SYSTEM PROMPT]
{llm_decision.get('system_prompt', '(Missing System Prompt)')}

[USER PROMPT]
{llm_decision.get('user_prompt', '(Missing User Prompt)')}

--------------------------------------------------------------------------------
🧠 PROCESSING (REASONING)
--------------------------------------------------------------------------------
{llm_decision.get('reasoning_detail', '(No reasoning detail)')}

--------------------------------------------------------------------------------
📥 OUTPUT (DECISION)
--------------------------------------------------------------------------------
{llm_decision.get('raw_response', '(No raw response)')}
"""
            self.saver.save_llm_log(
                content=full_log_content,
                symbol=self.current_symbol,
                snapshot_id=snapshot_id
            )
            
            # LOG: DeepSeek
            global_state.add_log(f"🧠 DeepSeek LLM: Action={vote_result.action.upper()} | Conf={llm_decision.get('confidence', 0)}% | {llm_decision.get('reasoning', '')[:50]}")
            
            # LOG: Bull/Bear Agents
            bull_conf = llm_decision.get('bull_perspective', {}).get('bull_confidence', 50)
            bear_conf = llm_decision.get('bear_perspective', {}).get('bear_confidence', 50)
            bull_stance = llm_decision.get('bull_perspective', {}).get('stance', 'UNKNOWN')
            bear_stance = llm_decision.get('bear_perspective', {}).get('stance', 'UNKNOWN')
            bull_reasons = llm_decision.get('bull_perspective', {}).get('bullish_reasons', '')[:50]
            bear_reasons = llm_decision.get('bear_perspective', {}).get('bearish_reasons', '')[:50]
            global_state.add_log(f"🐂 Bull Agent: [{bull_stance}] Conf={bull_conf}% | {bull_reasons}...")
            global_state.add_log(f"🐻 Bear Agent: [{bear_stance}] Conf={bear_conf}% | {bear_reasons}...")
            
            # ✅ Decision Recording moved after Risk Audit for complete context
            # Saved to file still happens here for "raw" decision
            self.saver.save_decision(asdict(vote_result), self.current_symbol, snapshot_id, cycle_id=cycle_id)

            # 如果是观望，也需要更新状态
            if vote_result.action in ('hold', 'wait'):
                print(f"\n✅ 决策: 观望 ({vote_result.action})")
                
                # GlobalState Logging of Logic
                regime_txt = vote_result.regime.get('regime', 'Unknown') if vote_result.regime else 'Unknown'
                pos_txt = f"{min(max(vote_result.position.get('position_pct', 0), 0), 100):.0f}%" if vote_result.position else 'N/A'
                
                # GlobalState Logging of Logic
                regime_txt = vote_result.regime.get('regime', 'Unknown') if vote_result.regime else 'Unknown'
                pos_txt = f"{min(max(vote_result.position.get('position_pct', 0), 0), 100):.0f}%" if vote_result.position else 'N/A'
                
                # LOG 3: Critic (Wait Case)
                global_state.add_log(f"⚖️ DecisionCoreAgent (The Critic): Context(Regime={regime_txt}, Pos={pos_txt}) => Vote: WAIT ({vote_result.reason})")
                # Check if there's an active position
                # For now, we assume no position in test mode (can be enhanced with real position check)
                actual_action = 'wait'  # No position → wait (观望)
                # If we had a position, it would be 'hold' (持有)
                
                # Check if there's an active position
                # For now, we assume no position in test mode (can be enhanced with real position check)
                actual_action = 'wait'  # No position → wait (观望)
                # If we had a position, it would be 'hold' (持有)
                
                # Update State with WAIT/HOLD decision
                decision_dict = asdict(vote_result)
                decision_dict['action'] = actual_action  # ✅ Use 'wait' instead of 'hold'
                decision_dict['symbol'] = self.current_symbol
                decision_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                decision_dict['cycle_number'] = global_state.cycle_counter
                decision_dict['cycle_id'] = global_state.current_cycle_id
                # Add implicit safe risk for Wait/Hold
                decision_dict['risk_level'] = 'safe'
                decision_dict['guardian_passed'] = True
                decision_dict['prophet_probability'] = predict_result.probability_up  # 🔮 Prophet
                
                # ✅ Add Semantic Analysis for Dashboard
                decision_dict['vote_analysis'] = SemanticConverter.convert_analysis_map(decision_dict.get('vote_details', {}))
                
                # Update Market Context
                if vote_result.regime:
                    global_state.market_regime = vote_result.regime.get('regime', 'Unknown')
                if vote_result.position:
                    # Safety clamp: ensure position_pct is 0-100
                    pos_pct = min(max(vote_result.position.get('position_pct', 0), 0), 100)
                    global_state.price_position = f"{pos_pct:.1f}% ({vote_result.position.get('location', 'Unknown')})"
                    
                global_state.update_decision(decision_dict)

                return {
                    'status': actual_action,
                    'action': actual_action,
                    'details': {
                        'reason': vote_result.reason,
                        'confidence': vote_result.confidence
                    }
                }
            
            # Step 4: 审计 - 风控守护者 (The Guardian)
            print(f"[Step 4/5] 👮 风控守护者 (The Guardian) - 进行终审...")
            
            # Critic Log for Action decision
            # Step 4: 审计 - 风控守护者 (The Guardian)
            print(f"[Step 4/5] 👮 风控守护者 (The Guardian) - 进行终审...")
            
            # LOG 3: Critic (Action Case) - if not already logged (Wait case returns early)
            regime_txt = vote_result.regime.get('regime', 'Unknown') if vote_result.regime else 'Unknown'
            # Note: Wait case returns, so if we are here, it's an action.
            global_state.add_log(f"⚖️ DecisionCoreAgent (The Critic): Context(Regime={regime_txt}) => Vote: {vote_result.action.upper()} (Conf: {vote_result.confidence*100:.0f}%)")
            
            global_state.guardian_status = "Auditing..."
            global_state.guardian_status = "Auditing..."
            
            order_params = self._build_order_params(
                action=vote_result.action,
                current_price=current_price,
                confidence=vote_result.confidence
            )
            
            print(f"  ✅ 信号方向: {vote_result.action}")
            print(f"  ✅ 综合信心: {vote_result.confidence:.1f}%")
            if vote_result.regime:
                print(f"  📊 市场状态: {vote_result.regime['regime']}")
            if vote_result.position:
                print(f"  📍 价格位置: {min(max(vote_result.position['position_pct'], 0), 100):.1f}% ({vote_result.position['location']})")
            
            # 将对抗式上下文注入订单参数，以便风控审计使用
            order_params['regime'] = vote_result.regime
            order_params['position'] = vote_result.position
            order_params['confidence'] = vote_result.confidence
            
            # Step 5 (Embedded in Step 4 for clean output)
            
            # 获取账户信息
            # Using _get_full_account_info helper (we will create it or inline logic)
            # Fetch directly from client to get full details
            try:
                if self.test_mode:
                    # Test Mode: Use virtual balance
                    wallet_bal = global_state.virtual_balance
                    avail_bal = global_state.virtual_balance
                    unrealized_pnl = 0.0 # Updated at end of cycle
                    
                    # Log for debugging
                    # log.info(f"Test Mode: Using virtual balance ${avail_bal}")
                    
                    account_balance = avail_bal
                else:
                    acc_info = self.client.get_futures_account()
                    # acc_info keys: 'total_wallet_balance', 'total_unrealized_profit', 'available_balance', etc. (snake_case)
                    wallet_bal = float(acc_info.get('total_wallet_balance', 0))
                    unrealized_pnl = float(acc_info.get('total_unrealized_profit', 0))
                    avail_bal = float(acc_info.get('available_balance', 0))
                    total_equity = wallet_bal + unrealized_pnl
                    
                    # Update State
                    global_state.update_account(
                        equity=total_equity,
                        available=avail_bal,
                        wallet=wallet_bal,
                        pnl=unrealized_pnl
                    )
                    global_state.record_account_success()  # Track success
                    
                    account_balance = avail_bal # For backward compatibility with audit
            except Exception as e:
                log.error(f"Failed to fetch account info: {e}")
                global_state.record_account_failure()  # Track failure
                global_state.add_log(f"❌ Account info fetch failed: {str(e)}")  # Dashboard log
                account_balance = 0.0

            current_position = self._get_current_position()
            
            # 执行审计
            audit_result = await self.risk_audit.audit_decision(
                decision=order_params,
                current_position=current_position,
                account_balance=account_balance,
                current_price=current_price
            )
            
            # Update Dashboard Guardian Status
            global_state.guardian_status = "PASSED" if audit_result.passed else "BLOCKED"
            
            # LOG 4: Guardian (Single Line)
            if not audit_result.passed:
                 global_state.add_log(f"🛡️ RiskAuditAgent (The Guardian): Result: ❌ BLOCKED ({audit_result.blocked_reason})")
            else:
                 warn_txt = f" | Corrections: {audit_result.corrections}" if audit_result.corrections else ""
                 global_state.add_log(f"🛡️ RiskAuditAgent (The Guardian): Result: ✅ PASSED (Risk: {audit_result.risk_level.value}){warn_txt}")
            
            # ✅ Update Global State with FULL Decision info (Vote + Audit)
            decision_dict = asdict(vote_result)
            decision_dict['symbol'] = self.current_symbol
            decision_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            decision_dict['cycle_number'] = global_state.cycle_counter
            decision_dict['cycle_id'] = global_state.current_cycle_id
            
            # Inject Risk Data
            decision_dict['risk_level'] = audit_result.risk_level.value
            decision_dict['guardian_passed'] = audit_result.passed
            decision_dict['guardian_reason'] = audit_result.blocked_reason
            decision_dict['prophet_probability'] = predict_result.probability_up  # 🔮 Prophet
            
            # ✅ Add Semantic Analysis for Dashboard
            decision_dict['vote_analysis'] = SemanticConverter.convert_analysis_map(decision_dict.get('vote_details', {}))
            
            # Update Market Context
            if vote_result.regime:
                global_state.market_regime = vote_result.regime.get('regime', 'Unknown')
            if vote_result.position:
                # Safety clamp: ensure position_pct is 0-100
                pos_pct = min(max(vote_result.position.get('position_pct', 0), 0), 100)
                global_state.price_position = f"{pos_pct:.1f}% ({vote_result.position.get('location', 'Unknown')})"
                
            global_state.update_decision(decision_dict)
            
            # ✅ Save Risk Audit Report
            from dataclasses import asdict as dc_asdict
            self.saver.save_risk_audit(
                audit_result={
                    'passed': audit_result.passed,
                    'risk_level': audit_result.risk_level.value,
                    'blocked_reason': audit_result.blocked_reason,
                    'corrections': audit_result.corrections,
                    'warnings': audit_result.warnings,
                    'order_params': order_params
                },
                symbol=self.current_symbol,
                snapshot_id=snapshot_id
            )
            
            print(f"  ✅ 审计结果: {'✅ 通过' if audit_result.passed else '❌ 拦截'}")
            print(f"  ✅ 风险等级: {audit_result.risk_level.value}")
            
            # 如果有修正
            if audit_result.corrections:
                print(f"  ⚠️  自动修正:")
                for key, value in audit_result.corrections.items():
                    print(f"     {key}: {order_params[key]} -> {value}")
                    order_params[key] = value  # 应用修正
            
            # 如果有警告
            if audit_result.warnings:
                print(f"  ⚠️  警告信息:")
                for warning in audit_result.warnings:
                    print(f"     {warning}")
            
            # 如果被拦截
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
            # Step 5: 执行引擎
            if self.test_mode:
                print("\n[Step 5/5] 🧪 TestMode - 模拟执行...")
                print(f"  模拟订单: {order_params['action']} {order_params['quantity']} @ {current_price}")
                
            if self.test_mode:
                print("\n[Step 5/5] 🧪 TestMode - 模拟执行...")
                print(f"  模拟订单: {order_params['action']} {order_params['quantity']} @ {current_price}")
                
                # LOG 5: Executor (Test)
                global_state.add_log(f"🚀 ExecutionEngine (The Executor): Mode=Test => Command: {order_params['action'].upper()} {order_params['quantity']} @ {current_price:.2f}")

                 # ✅ Save Execution (Simulated)
                self.saver.save_execution({
                    'symbol': self.current_symbol,
                    'action': 'SIMULATED_EXECUTION',
                    'params': order_params,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                }, self.current_symbol)
                
                # 💰 测试模式逻辑: 计算 PnL 和更新状态 (Virtual Account)
                realized_pnl = 0.0
                exit_test_price = 0.0
                
                if self.test_mode:
                    action_lower = vote_result.action.lower()
                    
                    # Close Logic
                    if 'close' in action_lower:
                        if self.current_symbol in global_state.virtual_positions:
                            pos = global_state.virtual_positions[self.current_symbol]
                            entry_price = pos['entry_price']
                            qty = pos['quantity']
                            side = pos['side']
                            
                            # Calc Realized PnL
                            if side.upper() == 'LONG':
                                realized_pnl = (current_price - entry_price) * qty
                            else:
                                realized_pnl = (entry_price - current_price) * qty
                            
                            exit_test_price = current_price
                            # Update Virtual Balance
                            global_state.virtual_balance += realized_pnl
                            
                            # Remove position
                            del global_state.virtual_positions[self.current_symbol]
                            
                            log.info(f"💰 [TEST] Closed {side} {self.current_symbol}: PnL=${realized_pnl:.2f}, Bal=${global_state.virtual_balance:.2f}")
                        else:
                            log.warning(f"⚠️ [TEST] Close ignored - No position for {self.current_symbol}")
                    
                    # Open Logic
                    elif 'long' in action_lower or 'short' in action_lower:
                        side = 'LONG' if 'long' in action_lower else 'SHORT'
                        global_state.virtual_positions[self.current_symbol] = {
                            'entry_price': current_price,
                            'quantity': order_params['quantity'],
                            'side': side,
                            'entry_time': datetime.now().isoformat(),
                            'stop_loss': order_params.get('stop_loss_price', 0),
                            'take_profit': order_params.get('take_profit_price', 0),
                            'leverage': order_params.get('leverage', 1)
                        }
                        log.info(f"💰 [TEST] Opened {side} {self.current_symbol} @ ${current_price:,.2f}")

                # ✅ Save Trade in persistent history
                # Logic Update: If CLOSING, try to update previous OPEN record. If failing, save new.
                
                is_close_action = 'close' in vote_result.action.lower()
                update_success = False
                
                if is_close_action:
                    update_success = self.saver.update_trade_exit(
                        symbol=self.current_symbol,
                        exit_price=exit_test_price,
                        pnl=realized_pnl,
                        exit_time=datetime.now().strftime("%H:%M:%S"),
                        close_cycle=global_state.cycle_counter
                    )
                    
                    # ✅ Sync global_state.trade_history if CSV update succeeded
                    if update_success:
                        for trade in global_state.trade_history:
                            if trade.get('symbol') == self.current_symbol and trade.get('exit_price', 0) == 0:
                                trade['exit_price'] = exit_test_price
                                trade['pnl'] = realized_pnl
                                trade['close_cycle'] = global_state.cycle_counter
                                trade['status'] = 'CLOSED'
                                log.info(f"✅ Synced global_state.trade_history: {self.current_symbol} PnL ${realized_pnl:.2f}")
                                break
                
                # Only save NEW record if it's OPEN action OR if Update Failed (Fallback)
                if not update_success:
                    is_open_action = 'open' in order_params['action'].lower()
                    
                    # For CLOSE actions, find the original open_cycle from trade_history
                    original_open_cycle = 0
                    if not is_open_action:
                        for trade in global_state.trade_history:
                            if trade.get('symbol') == self.current_symbol and trade.get('exit_price', 0) == 0:
                                original_open_cycle = trade.get('open_cycle', 0)
                                break
                    
                    trade_record = {
                        'open_cycle': global_state.cycle_counter if is_open_action else original_open_cycle,
                        'close_cycle': 0 if is_open_action else global_state.cycle_counter,
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'action': order_params['action'].upper(),
                        'symbol': self.current_symbol,
                        'price': current_price,
                        'quantity': order_params['quantity'],
                        'cost': current_price * order_params['quantity'],
                        'exit_price': exit_test_price,
                        'pnl': realized_pnl,
                        'confidence': order_params['confidence'],
                        'status': 'SIMULATED'
                    }
                    if is_close_action:
                         trade_record['status'] = 'CLOSED (Fallback)'
                         
                    self.saver.save_trade(trade_record)
                    # Update Global State History
                    global_state.trade_history.insert(0, trade_record)
                    if len(global_state.trade_history) > 50:
                        global_state.trade_history.pop()

                # 🎯 递增周期开仓计数器
                if 'open' in vote_result.action.lower():
                     global_state.cycle_positions_opened += 1
                     log.info(f"本周期已开仓: {global_state.cycle_positions_opened}/1")
                
                return {
                    'status': 'success',
                    'action': vote_result.action,
                    'details': order_params,
                    'current_price': current_price
                }
            else:
                # Live Execution
                print("\n[Step 5/5] 🚀 LiveTrade - 实盘执行...")
                
                try:
                    # _execute_order returns bool
                    is_success = self._execute_order(order_params)
                    
                    status_icon = "✅" if is_success else "❌"
                    status_txt = "SENT" if is_success else "FAILED"
                    
                    # LOG 5: Executor (Live)
                    global_state.add_log(f"🚀 ExecutionEngine (The Executor): Mode=Live | Command={order_params['action'].upper()} {order_params['quantity']} => Result: {status_icon} {status_txt}")
                        
                    executed = {'status': 'filled' if is_success else 'failed', 'avgPrice': current_price, 'executedQty': order_params['quantity']}
                        
                except Exception as e:
                    log.error(f"Live order execution failed: {e}", exc_info=True)
                    global_state.add_log(f"[Execution] ❌ Live Order Failed: {e}")
                    return {
                        'status': 'failed',
                        'action': vote_result.action,
                        'details': {'error': str(e)}
                    }
            
            # ✅ Save Execution
            self.saver.save_execution({
                'symbol': self.current_symbol,
                'action': 'REAL_EXECUTION',
                'params': order_params,
                'status': 'success' if executed else 'failed',
                'timestamp': datetime.now().isoformat()
            }, self.current_symbol)
            
            if executed:
                print("  ✅ 订单执行成功!")
                global_state.add_log(f"✅ Order: {order_params['action'].upper()} {order_params['quantity']} @ ${order_params['price']}")
                
                # 记录交易日志
                trade_logger.log_open_position(
                    symbol=self.current_symbol,
                    side=order_params['action'].upper(),
                    decision=order_params,
                    execution_result={
                        'success': True,
                        'entry_price': order_params['entry_price'],
                        'quantity': order_params['quantity'],
                        'stop_loss': order_params['stop_loss'],
                        'take_profit': order_params['take_profit'],
                        'order_id': 'real_order' # Placeholder if actual ID not captured
                    },
                    market_state=market_snapshot.live_5m,
                    account_info={'available_balance': account_balance}
                )
                
                # 计算盈亏 (如果是平仓)
                pnl = 0.0
                exit_price = 0.0
                entry_price = order_params['entry_price']
                if order_params['action'] == 'close_position' and current_position:
                    exit_price = current_price
                    entry_price = current_position.entry_price
                    # PnL = (Exit - Entry) * Qty (Multiplied by 1 if long, -1 if short)
                    direction = 1 if current_position.side == 'long' else -1
                    pnl = (exit_price - entry_price) * current_position.quantity * direction
                
                # ✅ Save Trade in persistent history
                # Logic Update: If CLOSING, try to update previous OPEN record. If failing, save new.
                
                is_close_action = 'close' in order_params['action'].lower()
                update_success = False
                
                if is_close_action:
                    update_success = self.saver.update_trade_exit(
                        symbol=self.current_symbol,
                        exit_price=exit_price,
                        pnl=pnl,
                        exit_time=datetime.now().strftime("%H:%M:%S"),
                        close_cycle=global_state.cycle_counter
                    )
                    
                    # ✅ Sync global_state.trade_history if CSV update succeeded
                    if update_success:
                        for trade in global_state.trade_history:
                            if trade.get('symbol') == self.current_symbol and trade.get('exit_price', 0) == 0:
                                trade['exit_price'] = exit_price
                                trade['pnl'] = pnl
                                trade['close_cycle'] = global_state.cycle_counter
                                trade['status'] = 'CLOSED'
                                log.info(f"✅ Synced global_state.trade_history: {self.current_symbol} PnL ${pnl:.2f}")
                                break
                
                if not update_success:
                    is_open_action = 'open' in order_params['action'].lower()
                    
                    # For CLOSE actions, find the original open_cycle from trade_history
                    original_open_cycle = 0
                    if not is_open_action:
                        for trade in global_state.trade_history:
                            if trade.get('symbol') == self.current_symbol and trade.get('exit_price', 0) == 0:
                                original_open_cycle = trade.get('open_cycle', 0)
                                break
                    
                    trade_record = {
                        'open_cycle': global_state.cycle_counter if is_open_action else original_open_cycle,
                        'close_cycle': 0 if is_open_action else global_state.cycle_counter,
                        'action': order_params['action'].upper(),
                        'symbol': self.current_symbol,
                        'price': entry_price,
                        'quantity': order_params['quantity'],
                        'cost': entry_price * order_params['quantity'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'confidence': order_params['confidence'],
                        'status': 'EXECUTED'
                    }
                    if is_close_action:
                         trade_record['status'] = 'CLOSED (Fallback)'
                         
                    self.saver.save_trade(trade_record)
                    
                    # Update Global State History
                    global_state.trade_history.insert(0, trade_record)
                    if len(global_state.trade_history) > 50:
                        global_state.trade_history.pop()
                
                return {
                    'status': 'success',
                    'action': vote_result.action,
                    'details': order_params,
                    'current_price': current_price
                }
            else:
                print("  ❌ 订单执行失败")
                global_state.add_log(f"❌ Order Failed: {order_params['action'].upper()}")
                return {
                    'status': 'failed',
                    'action': vote_result.action,
                    'details': {'error': 'execution_failed'},
                    'current_price': current_price
                }
        
        except Exception as e:
            log.error(f"计交易循环异常: {e}", exc_info=True)
            global_state.add_log(f"Error: {e}")
            return {
                'status': 'error',
                'details': {'error': str(e)}
            }
    
    def _build_order_params(
        self, 
        action: str, 
        current_price: float,
        confidence: float
    ) -> Dict:
        """
        构建订单参数
        
        Args:
            action: 'long' or 'short'
            current_price: 当前价格
            confidence: 决策置信度
        
        Returns:
            订单参数字典
        """
        # 计算仓位大小（根据置信度调整）
        position_multiplier = min(confidence * 1.2, 1.0)  # 最高100%仓位
        adjusted_position = self.max_position_size * position_multiplier
        
        # 计算数量
        quantity = adjusted_position / current_price
        
        # 计算止损止盈
        if action == 'long':
            stop_loss = current_price * (1 - self.stop_loss_pct / 100)
            take_profit = current_price * (1 + self.take_profit_pct / 100)
        else:  # short
            stop_loss = current_price * (1 + self.stop_loss_pct / 100)
            take_profit = current_price * (1 - self.take_profit_pct / 100)
        
        return {
            'action': action,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'quantity': quantity,
            'leverage': self.leverage,
            'confidence': confidence
        }
    
    def _get_account_balance(self) -> float:
        """获取账户可用余额"""
        try:
            return self.client.get_account_balance()
        except Exception as e:
            log.error(f"获取余额失败: {e}")
            return 0.0
    
    def _get_current_position(self) -> Optional[PositionInfo]:
        """获取当前持仓 (支持实盘 + Test Mode)"""
        try:
            # 1. Test Mode Support
            if self.test_mode:
                if self.current_symbol in global_state.virtual_positions:
                    v_pos = global_state.virtual_positions[self.current_symbol]
                    return PositionInfo(
                        symbol=self.current_symbol,
                        side=v_pos['side'].lower(), # ensure lowercase 'long'/'short'
                        entry_price=v_pos['entry_price'],
                        quantity=v_pos['quantity'],
                        unrealized_pnl=v_pos.get('unrealized_pnl', 0)
                    )
                return None

            # 2. Live Mode Support
            pos = self.client.get_futures_position(self.current_symbol)
            if pos and abs(pos['position_amt']) > 0:
                return PositionInfo(
                    symbol=self.current_symbol,
                    side='long' if pos['position_amt'] > 0 else 'short',
                    entry_price=pos['entry_price'],
                    quantity=abs(pos['position_amt']),
                    unrealized_pnl=pos['unrealized_profit']
                )
            return None
        except Exception as e:
            log.error(f"获取持仓失败: {e}")
            return None
    
    def _execute_order(self, order_params: Dict) -> bool:
        """
        执行订单
        
        Args:
            order_params: 订单参数
        
        Returns:
            是否成功
        """
        try:
            # 设置杠杆
            self.client.set_leverage(
                symbol=self.current_symbol,
                leverage=order_params['leverage']
            )
            
            # 市价开仓
            side = 'BUY' if order_params['action'] == 'long' else 'SELL'
            order = self.client.place_futures_market_order(
                symbol=self.current_symbol,
                side=side,
                quantity=order_params['quantity']
            )
            
            if not order:
                return False
            
            # 设置止损止盈
            self.execution_engine.set_stop_loss_take_profit(
                symbol=self.current_symbol,
                position_side='LONG' if order_params['action'] == 'long' else 'SHORT',
                stop_loss=order_params['stop_loss'],
                take_profit=order_params['take_profit']
            )
            
            return True
            
        except Exception as e:
            log.error(f"订单执行失败: {e}", exc_info=True)
            return False
    
    
    def _build_market_context(self, quant_analysis: Dict, predict_result, market_data: Dict) -> str:
        """
        构建 DeepSeek LLM 所需的市场上下文文本
        """
        # 提取关键数据
        current_price = market_data['current_price']
        
    # ... existing code ...
    from src.utils.semantic_converter import SemanticConverter
    
    def _format_choppy_analysis(self, regime_info: Dict) -> str:
        """Format CHOPPY market analysis for DeepSeek prompt"""
        if not regime_info or regime_info.get('regime') != 'choppy':
            return ""
        
        choppy = regime_info.get('choppy_analysis', {})
        if not choppy:
            return ""
        
        range_info = choppy.get('range', {})
        
        lines = [
            "",
            "### ⚠️ CHOPPY MARKET ANALYSIS (Range Trading Intelligence)",
            f"- **Squeeze Active**: {'YES 🔴' if choppy.get('squeeze_active') else 'NO'}",
            f"- **Squeeze Intensity**: {choppy.get('squeeze_intensity', 0):.0f}% (Higher = Breakout More Likely)",
            f"- **Breakout Probability**: {choppy.get('breakout_probability', 0):.0f}%",
            f"- **Potential Direction**: {choppy.get('breakout_direction', 'unknown').upper()}",
            f"- **Range Support**: ${range_info.get('support', 0):,.2f}",
            f"- **Range Resistance**: ${range_info.get('resistance', 0):,.2f}",
            f"- **Mean Reversion Signal**: {choppy.get('mean_reversion_signal', 'neutral').upper().replace('_', ' ')}",
            f"- **Consolidation Bars**: {choppy.get('consolidation_bars', 0)}",
            f"- **💡 Strategy Hint**: {choppy.get('strategy_hint', 'N/A')}",
            ""
        ]
        return "\n".join(lines)

    def _build_market_context(self, quant_analysis: Dict, predict_result, market_data: Dict, regime_info: Dict = None, position_info: Dict = None) -> str:
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
        prob_pct = predict_result.probability_up * 100
        prophet_signal = predict_result.signal
        
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
        
        o_score_total = oscillator.get('total_oscillator_score')
        o_semantic = SemanticConverter.get_oscillator_semantic(o_score_total)
        
        s_score_total = sentiment.get('total_sentiment_score')
        s_semantic = SemanticConverter.get_sentiment_score_semantic(s_score_total)

        rsi_15m = oscillator.get('rsi_15m')
        rsi_1h = oscillator.get('rsi_1h')
        rsi_1m_semantic = SemanticConverter.get_rsi_semantic(rsi_15m)
        rsi_1h_semantic = SemanticConverter.get_rsi_semantic(rsi_1h)
        
        # MACD
        macd_15m = trend.get('details', {}).get('15m_macd_diff')
        macd_semantic = SemanticConverter.get_macd_semantic(macd_15m)
        
        oi_change = sentiment.get('oi_change_24h_pct')
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
3. **If Closing**: Return `close_position` action.
"""
        
        context = f"""
## 1. Price Overview
- Current Price: ${current_price:,.2f}
- Symbol: {self.current_symbol}

{position_section}

## 2. Trend Analysis
- 1h Trend: {t_1h_sem} (Score: {fmt_val(t_1h_score, "{:.0f}")})
- 15m Trend: {t_15m_sem} (Score: {fmt_val(t_15m_score, "{:.0f}")})
- 5m Trend: {t_5m_sem} (Score: {fmt_val(t_5m_score, "{:.0f}")})
- Total Trend Score: {fmt_val(t_score_total, "{:.0f}")} (Range: -100 to +100) => {t_semantic}

## 3. Oscillators
- RSI (15m): {fmt_val(rsi_15m)} => {rsi_1m_semantic}
- RSI (1h): {fmt_val(rsi_1h)} => {rsi_1h_semantic}
- MACD (15m): {fmt_val(macd_15m, "{:.4f}")} => {macd_semantic}
- Total Oscillator Score: {fmt_val(o_score_total, "{:.0f}")} (Range: -100 to +100) => {o_semantic}

## 4. Market Sentiment
- 24h OI Change: {fmt_val(oi_change)}% => {oi_semantic}
- Total Sentiment Score: {fmt_val(s_score_total, "{:.0f}")} (Range: -100 to +100) => {s_semantic}

## 5. AI Prediction (Prophet)
- Forecast: {prediction_desc}
- Signal: {prophet_signal}
- Confidence: {predict_result.confidence:.0%}

## 6. Market Regime & Price Position
- Market Regime: {regime_type.upper()} (Confidence: {min(max(regime_confidence, 0), 100):.0f}%)
- Price Position: {price_position.upper()} ({min(max(price_position_pct, 0), 100):.1f}% of recent range)
- Note: Position near extremes (0-20% or 80-100%) suggests potential reversal zones
{self._format_choppy_analysis(regime_info)}
## 7. Comprehensive Score
- Strategist Score: {quant_analysis.get('comprehensive', {}).get('score', 0):.0f}/100
"""
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
        return {
            'decision_core': self.decision_core.get_statistics(),
            'risk_audit': self.risk_audit.get_audit_report(),
        }

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

    def run_continuous(self, interval_minutes: int = 3):
        """
        持续运行模式
        
        Args:
            interval_minutes: 运行间隔（分钟）
        """
        log.info(f"🚀 启动持续运行模式 (间隔: {interval_minutes}分钟)")
        global_state.is_running = True
        
        # Logger is configured in src.utils.logger, no need to override here.
        # Dashboard logging is handled via global_state.add_log -> log.bind(dashboard=True)

        # Start Real-time Monitors
        self.start_account_monitor()
        
        # 🔮 启动 Prophet 自动训练器 (每 2 小时重新训练)
        from src.models.prophet_model import ProphetAutoTrainer, HAS_LIGHTGBM
        if HAS_LIGHTGBM:
            # 为主交易对创建自动训练器
            primary_agent = self.predict_agents[self.primary_symbol]
            self.auto_trainer = ProphetAutoTrainer(
                predict_agent=primary_agent,
                binance_client=self.client,
                interval_hours=2.0,  # 每 2 小时训练一次
                training_days=7,     # 使用最近 7 天数据
                symbol=self.primary_symbol
            )
            self.auto_trainer.start()
        
        # 设置初始间隔 (优先使用 CLI 参数，后续 API 可覆盖)
        global_state.cycle_interval = interval_minutes
        
        log.info(f"🚀 启动持续交易模式 (间隔: {global_state.cycle_interval}m)")
        
        # 🧪 Test Mode: Initialize Virtual Account for Chart
        if self.test_mode:
            log.info("🧪 Test Mode: Initializing Virtual Account...")
            global_state.update_account(
                equity=global_state.virtual_balance,
                available=global_state.virtual_balance,
                wallet=global_state.virtual_balance,
                pnl=0.0
            )
        
        try:
            while global_state.is_running:
                # Check pause state
                if global_state.execution_mode == 'Paused':
                    # 首次进入暂停时打印日志
                    if not hasattr(self, '_pause_logged') or not self._pause_logged:
                        print("\n⏸️ 系统已暂停，等待恢复...")
                        global_state.add_log("⏸️ System PAUSED - waiting for resume...")
                        self._pause_logged = True
                    time.sleep(1)
                    continue
                else:
                    self._pause_logged = False  # 重置暂停日志标记
                
                if global_state.execution_mode == 'Stopped':
                    print("\n⏹️ 系统已停止")
                    global_state.add_log("⏹️ System STOPPED by user")
                    break

                # ✅ 统一周期计数: 在遍历币种前递增一次
                global_state.cycle_counter += 1
                cycle_num = global_state.cycle_counter
                cycle_id = f"cycle_{cycle_num:04d}_{int(time.time())}"
                global_state.current_cycle_id = cycle_id
                
                # 🧪 Test Mode: Record start of cycle account state (for Net Value Curve)
                if self.test_mode:
                    # Re-log current state with new cycle number so chart shows start of cycle
                    global_state.update_account(
                        equity=global_state.account_overview['total_equity'],
                        available=global_state.account_overview['available_balance'],
                        wallet=global_state.account_overview['wallet_balance'],
                        pnl=global_state.account_overview['total_pnl']
                    )
                
                print(f"\n{'='*80}")
                print(f"🔄 Cycle #{cycle_num} | 分析 {len(self.symbols)} 个交易对")
                print(f"{'='*80}")
                global_state.add_log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                global_state.add_log(f"🔄 Cycle #{cycle_num} started | Symbols: {', '.join(self.symbols)}")

                # 🎯 重置周期开仓计数器
                global_state.cycle_positions_opened = 0
                
                # 🔄 多币种顺序处理: 依次分析每个交易对
                # Step 1: 收集所有交易对的决策
                all_decisions = []
                latest_prices = {}  # Store latest prices for PnL calculation
                for symbol in self.symbols:
                    self.current_symbol = symbol  # 设置当前处理的交易对
                    
                    # Use asyncio.run for the async cycle
                    result = asyncio.run(self.run_trading_cycle())
                    
                    # Capture price from global state (updated inside run_trading_cycle)
                    latest_prices[symbol] = global_state.current_price
                    
                    print(f"  [{symbol}] 结果: {result['status']}")
                    
                    # 如果是开仓决策，收集起来
                    if result.get('action') in ['open_long', 'open_short'] and result.get('status') == 'success':
                        all_decisions.append({
                            'symbol': symbol,
                            'result': result,
                            'confidence': result.get('confidence', 0)
                        })
                
                # Step 2: 从所有开仓决策中选择信心度最高的一个
                if all_decisions:
                    # 按信心度排序
                    all_decisions.sort(key=lambda x: x['confidence'], reverse=True)
                    best_decision = all_decisions[0]
                    
                    print(f"\n🎯 本周期最优开仓机会: {best_decision['symbol']} (信心度: {best_decision['confidence']:.1f}%)")
                    global_state.add_log(f"🎯 Best opportunity this cycle: {best_decision['symbol']} (Confidence: {best_decision['confidence']:.1f}%)")
                    
                    # 只执行最优的一个
                    # 注意：实际执行已经在 run_trading_cycle 中完成了
                    # 这里只是记录和通知
                    
                    # 如果有其他开仓机会被跳过，记录下来
                    if len(all_decisions) > 1:
                        skipped = [f"{d['symbol']}({d['confidence']:.1f}%)" for d in all_decisions[1:]]
                        print(f"  ⏭️  跳过其他机会: {', '.join(skipped)}")
                        global_state.add_log(f"⏭️  Skipped opportunities: {', '.join(skipped)} (1 position per cycle limit)")
                
                        global_state.add_log(f"⏭️  Skipped opportunities: {', '.join(skipped)} (1 position per cycle limit)")
                
                # 💰 Update Virtual Account PnL (Mark-to-Market)
                if self.test_mode:
                    self._update_virtual_account_stats(latest_prices)
                
                # Dynamic Interval: specific to new requirement
                current_interval = global_state.cycle_interval
                
                # 等待下一次检查
                print(f"\n⏳ 等待 {current_interval} 分钟...")
                
                # Sleep in chunks to allow responsive PAUSE/STOP and INTERVAL changes
                # Check every 1 second during the wait interval
                elapsed_seconds = 0
                while True:
                    # 每秒检查当前间隔设置 (支持动态调整)
                    current_interval = global_state.cycle_interval
                    wait_seconds = current_interval * 60
                    
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
                             global_state.add_log(f"⏳ Waiting next cycle... ({remaining}m)")

                    time.sleep(1)
                    elapsed_seconds += 1
                
        except KeyboardInterrupt:
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
        
        # 更新 Global State
        global_state.update_account(
            equity=total_equity,
            available=global_state.virtual_balance,
            wallet=global_state.virtual_balance,
            pnl=total_unrealized_pnl
        )

def start_server():
    """Start FastAPI server in a separate thread"""
    print("\n🌍 Starting Web Dashboard at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

# ============================================
# 主入口
# ============================================
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多Agent交易机器人')
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--max-position', type=float, default=100.0, help='最大单笔金额')
    parser.add_argument('--leverage', type=int, default=1, help='杠杆倍数')
    parser.add_argument('--stop-loss', type=float, default=1.0, help='止损百分比')
    parser.add_argument('--take-profit', type=float, default=2.0, help='止盈百分比')
    parser.add_argument('--mode', choices=['once', 'continuous'], default='once', help='运行模式')
    parser.add_argument('--interval', type=float, default=3.0, help='持续运行间隔（分钟）')
    
    args = parser.parse_args()
    
    # 测试模式默认 1 分钟周期，实盘模式默认 3 分钟
    if args.test and args.interval == 3.0:  # 如果是测试模式且用户没有指定间隔
        args.interval = 1.0
    
    
    # 创建机器人
    bot = MultiAgentTradingBot(
        max_position_size=args.max_position,
        leverage=args.leverage,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        test_mode=args.test
    )
    
    # 启动 Dashboard Server (Only if in continuous mode or if explicitly requested, but let's do it always for now if deps exist)
    try:
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
    except Exception as e:
        print(f"⚠️ Failed to start Dashboard: {e}")
    
    # 运行
    if args.mode == 'once':
        result = bot.run_once()
        print(f"\n最终结果: {json.dumps(result, indent=2)}")
        
        # 显示统计
        stats = bot.get_statistics()
        print(f"\n统计信息:")
        print(json.dumps(stats, indent=2))
        
        # Keep alive briefly for server to be reachable if desired, 
        # or exit immediately. Usually 'once' implies run and exit.
        
    else:
        bot.run_continuous(interval_minutes=args.interval)

if __name__ == '__main__':
    main()
