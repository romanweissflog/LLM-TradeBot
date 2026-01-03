"""
回测引擎核心 (Backtest Engine)
================================

协调数据回放、策略执行和性能评估

Author: AI Trader Team
Date: 2025-12-31
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import pandas as pd

from src.backtest.data_replay import DataReplayAgent
from src.backtest.portfolio import BacktestPortfolio, Side, Trade
from src.backtest.metrics import PerformanceMetrics, MetricsResult
from src.backtest.report import BacktestReport
from src.utils.logger import log


@dataclass
class BacktestConfig:
    """回测配置"""
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    max_position_size: float = 1000000.0
    leverage: int = 1
    stop_loss_pct: float = 1.0
    take_profit_pct: float = 2.0
    slippage: float = 0.001
    commission: float = 0.0004
    step: int = 1  # 1=每5分钟, 3=每15分钟, 12=每小时
    margin_mode: str = "cross"  # "cross" 或 "isolated"
    contract_type: str = "linear"  # "linear" 或 "inverse"
    contract_size: float = 100.0  # 币本位合约面值 (BTC=100 USD)
    strategy_mode: str = "agent"  # "technical" (EMA) or "agent" (Multi-Agent) - Default: agent for prompt optimization
    use_llm: bool = False  # 是否在回测中调用 LLM（费用高、速度慢）
    llm_cache: bool = True  # 缓存 LLM 响应
    llm_throttle_ms: int = 100  # LLM 调用间隔（毫秒），避免速率限制
    
    def __post_init__(self):
        """验证配置参数"""
        from datetime import datetime
        
        # 验证日期格式
        try:
            # Try full datetime format first
            try:
                start = datetime.strptime(self.start_date, '%Y-%m-%d %H:%M')
            except ValueError:
                # Fallback to date only
                start = datetime.strptime(self.start_date, '%Y-%m-%d')
            
            try:
                end = datetime.strptime(self.end_date, '%Y-%m-%d %H:%M')
            except ValueError:
                # Fallback to date only
                end = datetime.strptime(self.end_date, '%Y-%m-%d')
                
            if start >= end:
                raise ValueError(f"start_date ({self.start_date}) must be before end_date ({self.end_date})")
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError(f"Invalid date format. Expected YYYY-MM-DD or YYYY-MM-DD HH:MM, got start_date={self.start_date}, end_date={self.end_date}")
            raise
        
        # 验证数值范围
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        
        if self.max_position_size <= 0:
            raise ValueError(f"max_position_size must be positive, got {self.max_position_size}")
        
        if self.leverage < 1 or self.leverage > 125:
            raise ValueError(f"leverage must be between 1 and 125, got {self.leverage}")
        
        if self.stop_loss_pct < 0 or self.stop_loss_pct > 100:
            raise ValueError(f"stop_loss_pct must be between 0 and 100, got {self.stop_loss_pct}")
        
        if self.take_profit_pct < 0:
            raise ValueError(f"take_profit_pct must be non-negative, got {self.take_profit_pct}")
        
        if self.slippage < 0 or self.slippage > 1:
            raise ValueError(f"slippage must be between 0 and 1, got {self.slippage}")
        
        if self.commission < 0 or self.commission > 1:
            raise ValueError(f"commission must be between 0 and 1, got {self.commission}")
        
        if self.step < 1:
            raise ValueError(f"step must be at least 1, got {self.step}")
        
        # 验证symbol格式
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("symbol must be a non-empty string")
        
        # 验证策略模式
        if self.strategy_mode not in ['technical', 'agent']:
            raise ValueError(f"strategy_mode must be 'technical' or 'agent', got {self.strategy_mode}")
        
        # 验证保证金模式
        if self.margin_mode not in ['cross', 'isolated']:
            raise ValueError(f"margin_mode must be 'cross' or 'isolated', got {self.margin_mode}")
        
        # 验证合约类型
        if self.contract_type not in ['linear', 'inverse']:
            raise ValueError(f"contract_type must be 'linear' or 'inverse', got {self.contract_type}")



@dataclass
class BacktestResult:
    """回测结果"""
    config: BacktestConfig
    metrics: MetricsResult
    equity_curve: pd.DataFrame
    trades: List[Trade]
    decisions: List[Dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        # 获取决策数据并去重
        def _get_filtered_decisions():
            """获取过滤和去重后的决策列表"""
            # 获取最后50个决策
            recent = self.decisions[-50:] if len(self.decisions) > 50 else self.decisions
            # 获取所有非hold决策
            non_hold = [d for d in self.decisions if d.get('action') != 'hold']
            
            # 合并并去重（基于timestamp）
            seen = set()
            result = []
            for d in recent + non_hold:
                # 使用timestamp作为唯一键
                key = d.get('timestamp')
                if key and key not in seen:
                    seen.add(key)
                    # 只保留需要的字段
                    filtered = {k: v for k, v in d.items() if k in ['timestamp', 'action', 'confidence', 'reason', 'price', 'vote_details']}
                    result.append(filtered)
            return result
        
        return {
            'config': {
                'symbol': self.config.symbol,
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'initial_capital': self.config.initial_capital,
            },
            'metrics': self.metrics.to_dict(),
            'total_trades': len(self.trades),
            'duration_seconds': self.duration_seconds,
            'decisions': _get_filtered_decisions(),
        }


class BacktestEngine:
    """
    回测引擎核心
    
    工作流程：
    1. 加载历史数据
    2. 初始化虚拟投资组合
    3. 遍历每个时间点
    4. 执行策略决策
    5. 模拟交易执行
    6. 记录净值和交易
    7. 计算性能指标
    8. 生成报告
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        strategy_fn: Optional[Callable] = None
    ):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
            strategy_fn: 策略函数，接收 (snapshot, portfolio) 返回 {'action': 'long/short/hold', 'confidence': 0-1}
        """
        self.config = config
        self.strategy_fn = strategy_fn or self._default_strategy
        
        # 组件
        self.data_replay: Optional[DataReplayAgent] = None
        self.portfolio: Optional[BacktestPortfolio] = None
        self.agent_runner = None
        
        # Initialize Agent Runner if needed
        if config.strategy_mode == "agent":
            from src.backtest.agent_wrapper import BacktestAgentRunner
            self.agent_runner = BacktestAgentRunner(config.__dict__)
        
        # 状态
        self.is_running = False
        self.current_timestamp: Optional[datetime] = None
        self.decisions: List[Dict] = []
        
        log.info(f"🔬 BacktestEngine initialized | {config.symbol} | "
                 f"{config.start_date} to {config.end_date}")
    
    async def run(self, progress_callback: Callable = None) -> BacktestResult:
        """
        运行完整回测
        
        Args:
            progress_callback: 进度回调函数 (current, total, pct)
            
        Returns:
            BacktestResult 对象
        """
        start_time = datetime.now()
        self.is_running = True
        
        log.info("=" * 60)
        log.info("🚀 Starting Backtest")
        log.info("=" * 60)
        
        # 1. 初始化数据回放器
        self.data_replay = DataReplayAgent(
            symbol=self.config.symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        
        success = await self.data_replay.load_data()
        if not success:
            raise RuntimeError("Failed to load historical data")
        
        # 2. 初始化投资组合
        self.portfolio = BacktestPortfolio(
            initial_capital=self.config.initial_capital,
            slippage=self.config.slippage,
            commission=self.config.commission
        )
        
        # 3. 遍历时间点
        timestamps = list(self.data_replay.iterate_timestamps(step=self.config.step))
        total = len(timestamps)
        
        log.info(f"📊 Processing {total} timestamps (step={self.config.step})")
        log.info(f"⏱️  Estimated time: {total * 72 / 60:.1f} minutes (3 LLM calls per timepoint)")
        
        for i, timestamp in enumerate(timestamps):
            if not self.is_running:
                log.warning("Backtest stopped by user")
                break
            
            self.current_timestamp = timestamp
            
            try:
                # 获取市场快照
                snapshot = self.data_replay.get_snapshot_at(timestamp)
                current_price = self.data_replay.get_current_price()
                
                # 🆕 检查并应用资金费率结算
                funding_rate = self.data_replay.get_funding_rate_for_settlement(timestamp)
                if funding_rate is not None:
                    # 获取标记价格（若有）
                    fr_record = self.data_replay.get_funding_rate_at(timestamp)
                    mark_price = fr_record.mark_price if fr_record and fr_record.mark_price > 0 else current_price
                    
                    # 对所有持仓应用资金费率
                    for symbol in list(self.portfolio.positions.keys()):
                        self.portfolio.apply_funding_fee(symbol, funding_rate, mark_price, timestamp)
                
                # 🆕 检查强平
                prices = {self.config.symbol: current_price}
                liquidated = self.portfolio.check_liquidation(prices, timestamp)
                if liquidated:
                    log.warning(f"⚠️ Positions liquidated: {liquidated}")
                    continue  # 强平后跳过本轮策略执行
                
                # 检查止损止盈
                self.portfolio.check_stop_loss_take_profit(prices, timestamp)
                
                # 执行策略
                decision = await self._execute_strategy(snapshot, current_price)
                self.decisions.append(decision)
                
                # 执行交易
                await self._execute_decision(decision, current_price, timestamp)
                
                # 记录净值 (OPTIMIZATION: Sample every 12 steps or on key events)
                should_record_equity = (i % 12 == 0) or (i == total - 1) or (decision['action'] != 'hold')
                if should_record_equity:
                    self.portfolio.record_equity(timestamp, prices)
                
                
                # 进度回调（包含实时收益数据和增量可视化数据）
                if progress_callback:
                    progress_pct = (i + 1) / total * 100  # +1 because we just completed this timepoint
                    
                    # Send progress update
                    current_equity = self.portfolio.get_current_equity(prices)
                    profit = current_equity - self.config.initial_capital
                    profit_pct = (profit / self.config.initial_capital) * 100
                    
                    # Calculate fresh equity point for real-time display (not from stale curve)
                    # Update peak for drawdown calculation
                    if current_equity > self.portfolio.peak_equity:
                        peak_equity = current_equity
                    else:
                        peak_equity = self.portfolio.peak_equity
                    
                    drawdown = peak_equity - current_equity
                    drawdown_pct = drawdown / peak_equity * 100 if peak_equity > 0 else 0
                    
                    latest_equity_point = {
                        'timestamp': timestamp.isoformat(),
                        'total_equity': float(current_equity),
                        'drawdown_pct': float(drawdown_pct)
                    }
                    
                    # 获取最新交易（最近1笔）
                    latest_trade = None
                    if self.portfolio.trades:
                        trade = self.portfolio.trades[-1]
                        latest_trade = {
                            'timestamp': trade.timestamp.isoformat(),
                            'side': trade.side.value,
                                            'action': trade.action,
                            'price': float(trade.price),
                            'pnl': float(trade.pnl),
                            'pnl_pct': float(trade.pnl_pct)
                        }
                    
                    # 计算实时指标
                    trades_count = len(self.portfolio.trades)
                    winning_trades = sum(1 for t in self.portfolio.trades if t.pnl > 0 and t.action == 'close')
                    win_rate = (winning_trades / trades_count * 100) if trades_count > 0 else 0
                    
                    callback_data = {
                        'progress': progress_pct,
                        'current_timepoint': i + 1,  # Human-readable: 1-indexed
                        'total_timepoints': total,
                        'current_equity': current_equity,
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                        'equity_point': latest_equity_point,
                        'latest_trade': latest_trade,
                        'metrics': {
                            'total_trades': trades_count,
                            'win_rate': win_rate,
                            'max_drawdown_pct': self.portfolio.equity_curve[-1].drawdown_pct if self.portfolio.equity_curve else 0
                        }
                    }

                    if asyncio.iscoroutinefunction(progress_callback):
                        await progress_callback(callback_data)
                    else:
                        progress_callback(callback_data)
                
            except (KeyError, ValueError, IndexError) as e:
                # 可恢复的数据错误：记录警告并跳过此时间点
                log.warning(f"Data error at {timestamp}: {type(e).__name__}: {e}, skipping this timestamp")
                continue
            except Exception as e:
                # 致命错误：记录错误并终止回测
                log.error(f"Fatal error at {timestamp}: {type(e).__name__}: {e}")
                log.error(f"Backtest terminated due to fatal error")
                raise RuntimeError(f"Backtest failed at {timestamp}: {e}") from e
        
        # 4. 强制平仓所有持仓
        await self._close_all_positions()
        
        # 5. 计算性能指标
        equity_curve = self.portfolio.get_equity_dataframe()
        trades = self.portfolio.trades
        
        metrics = PerformanceMetrics.calculate(
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=self.config.initial_capital
        )
        
        # 6. 生成结果
        duration = (datetime.now() - start_time).total_seconds()
        
        result = BacktestResult(
            config=self.config,
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades,
            decisions=self.decisions,
            duration_seconds=duration
        )
        
        self.is_running = False
        
        log.info("=" * 60)
        log.info("✅ Backtest Complete")
        log.info(f"   Duration: {duration:.1f}s")
        log.info(f"   Total Return: {metrics.total_return:+.2f}%")
        log.info(f"   Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        log.info(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        log.info(f"   Win Rate: {metrics.win_rate:.1f}%")
        log.info(f"   Total Trades: {metrics.total_trades}")
        log.info(f"   💸 Funding Paid: ${self.portfolio.total_funding_paid:.4f}")
        log.info(f"   💰 Fees Paid: ${self.portfolio.total_fees_paid:.2f}")
        log.info(f"   📉 Slippage Cost: ${self.portfolio.total_slippage_cost:.2f}")
        log.info(f"   🔥 Liquidations: {self.portfolio.liquidation_count}")
        log.info("=" * 60)
        
        return result
    
    async def _execute_strategy(
        self,
        snapshot,
        current_price: float
    ) -> Dict:
        """执行策略并返回决策"""
        try:
            # 调用策略函数
            # DEBUG LOG
            log.info(f"DEBUG: execute_strategy mode={self.config.strategy_mode} runner={self.agent_runner}")
            
            if self.config.strategy_mode == "agent" and self.agent_runner:
                log.info("DEBUG: Entering agent runner step")
                decision = await self.agent_runner.step(snapshot, self.portfolio)
            else:
                decision = await self.strategy_fn(
                    snapshot=snapshot,
                    portfolio=self.portfolio,
                    current_price=current_price,
                    config=self.config
                )
            
            decision['timestamp'] = self.current_timestamp
            decision['price'] = current_price
            
            return decision
            
        except Exception as e:
            log.warning(f"Strategy error: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reason': f'strategy_error: {e}',
                'timestamp': self.current_timestamp,
                'price': current_price
            }
    
    async def _execute_decision(
        self,
        decision: Dict,
        current_price: float,
        timestamp: datetime
    ):
        """执行交易决策"""
        action = decision.get('action', 'hold')
        confidence = decision.get('confidence', 0.0)
        
        # 0. Global Safety Check: Minimum Confidence 50%
        # Filters out weak mechanical signals when LLM yields (0% confidence)
        if action in ['long', 'short', 'open_long', 'open_short', 'add_position'] and confidence < 50:
            log.warning(f"🚫 Confidence {confidence}% < 50% for {action}. Forcing WAIT.")
            return {'action': 'wait', 'reason': 'low_confidence_filtering'}
        
        # NOTE: Volatile Regime Guard REMOVED - was too strict, blocking all trades
        # The LLM already provides this context in the reason field
        
        # Normalize actions
        if action == 'open_long': action = 'long'
        if action == 'open_short': action = 'short'
        
        symbol = self.config.symbol
        has_position = symbol in self.portfolio.positions
        
        # Handle Add Position (Treat as increasing existing position)
        if action == 'add_position' and has_position:
            # Re-map to long/short based on current side
            current_side = self.portfolio.positions[symbol].side
            action = 'long' if current_side == Side.LONG else 'short'
            # Fall through to Open logic
            
        # Handle Reduce Position
        if action == 'reduce_position' and has_position:
            # Partial Close logic
            current_pos = self.portfolio.positions[symbol]
            reduce_pct = 0.5 # Default reduce by 50%
            
            # Check if LLM specified size
            params = decision.get('trade_params') or {}
            if params.get('position_size_pct', 0) > 0:
                # Interpret as "Reduce BY X%" or "Reduce TO X%"? 
                # Usually "Position Size" implies target. 
                # Let's assume Reduce means "Close 50%" unless specific instructions.
                # Simplest: Reduce 50%
                pass
            
            reduce_qty = current_pos.quantity * reduce_pct
            self.portfolio.close_position(
                symbol=symbol,
                price=current_price,
                timestamp=timestamp,
                quantity=reduce_qty,
                reason='reduce_position'
            )
            return

        # Basic Action Filtering
        if action in ['close', 'close_short', 'close_long'] and has_position:
            # 平仓 (Close Position)
            # Validate direction matches if specified (close_short for SHORT, close_long for LONG)
            current_side = self.portfolio.positions[symbol].side
            if action == 'close_short' and current_side != Side.SHORT:
                log.warning(f"⚠️ close_short signal but position is {current_side}, ignoring")
                return
            if action == 'close_long' and current_side != Side.LONG:
                log.warning(f"⚠️ close_long signal but position is {current_side}, ignoring")
                return
            
            # Profit Protection: Don't close winners too early
            pos = self.portfolio.positions[symbol]
            current_pnl_pct = pos.get_pnl_pct(current_price)
            hold_hours = (timestamp - pos.entry_time).total_seconds() / 3600 if pos.entry_time else 0
            
            # If profitable AND held < 2 hours, let it run (unless trailing stop triggered)
            if current_pnl_pct > 0 and hold_hours < 2:
                log.info(f"🛡️ Profit Protection: +{current_pnl_pct:.2f}% profit, only {hold_hours:.1f}h held < 2h. Keeping position.")
                return
            
            self.portfolio.close_position(
                symbol=symbol,
                price=current_price,
                timestamp=timestamp,
                reason='signal'
            )
            log.info(f"✅ Closed {current_side.value} position via {action} signal")
            return

        if action in ['long', 'short']:
            side = Side.LONG if action == 'long' else Side.SHORT
            
            # 处理反向持仓 (Reversal)
            if has_position:
                current_side = self.portfolio.positions[symbol].side
                if current_side != side:
                    # 反向信号，先平仓
                    self.portfolio.close_position(
                        symbol=symbol,
                        price=current_price,
                        timestamp=timestamp,
                        reason='reverse_signal'
                    )
                    has_position = False # 标记为无持仓，以便下面执行开仓
            
            # 执行开/加仓 (Open / Add)
            # 此时我们要么是无持仓(New/Reversal)，要么是同向持仓(Add)
            
            # --- 复制之前的动态参数逻辑 ---
            params = decision.get('trade_params') or {}
            leverage = params.get('leverage') or self.config.leverage
            sl_pct = params.get('stop_loss_pct') or self.config.stop_loss_pct
            tp_pct = params.get('take_profit_pct') or self.config.take_profit_pct
            trailing_pct = params.get('trailing_stop_pct')
            
            available_cash = self.portfolio.cash
            
            if params.get('position_size_pct', 0) > 0:
                use_cash = available_cash * (params['position_size_pct'] / 100)
                target_position_value = use_cash * leverage
                position_size = min(
                    target_position_value,
                    available_cash * 0.98 * leverage,
                    self.config.max_position_size * leverage
                )
            else:
                # 默认逻辑
                position_size = min(
                    self.config.max_position_size * leverage,
                    available_cash * 0.95
                )
            
            # 如果是加仓，检查是否超过总上限 (Portfolio.open_position 通常处理增加，但我们需要控制总量)
            # 简化起见，我们假设 position_size 是本次下单量 (Incremental)
            # 对于 "Add"，Agent通常意味着 "Add X amount". 
            
            quantity = position_size / current_price
            
            if quantity > 0:
                self.portfolio.open_position(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=current_price,
                    timestamp=timestamp,
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct,
                    trailing_stop_pct=trailing_pct
                )
    
    async def _close_all_positions(self):
        """平仓所有持仓"""
        if self.portfolio is None:
            return
        
        for symbol in list(self.portfolio.positions.keys()):
            current_price = self.data_replay.get_current_price()
            self.portfolio.close_position(
                symbol=symbol,
                price=current_price,
                timestamp=self.current_timestamp,
                reason='backtest_end'
            )
    
    async def _default_strategy(
        self,
        snapshot,
        portfolio: BacktestPortfolio,
        current_price: float,
        config: BacktestConfig
    ) -> Dict:
        """
        默认策略（简单趋势跟踪）
        
        使用 EMA 交叉作为信号（直接计算，无外部依赖）
        """
        # 获取稳定数据
        df = snapshot.stable_5m.copy()
        
        if len(df) < 50:
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'insufficient_data'}
        
        # 计算 EMA（直接计算）
        close = df['close'].astype(float)
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        
        # 当前和前一个值
        ema_fast = ema_20.iloc[-1]
        ema_slow = ema_50.iloc[-1]
        ema_fast_prev = ema_20.iloc[-2]
        ema_slow_prev = ema_50.iloc[-2]
        
        # 金叉/死叉
        symbol = config.symbol
        has_position = symbol in portfolio.positions
        
        if ema_fast > ema_slow and ema_fast_prev <= ema_slow_prev:
            # 金叉 - 做多
            if has_position:
                current_side = portfolio.positions[symbol].side
                if current_side == Side.SHORT:
                    return {'action': 'long', 'confidence': 0.7, 'reason': 'golden_cross_reverse'}
                return {'action': 'hold', 'confidence': 0.5, 'reason': 'already_long'}
            return {'action': 'long', 'confidence': 0.7, 'reason': 'golden_cross'}
        
        elif ema_fast < ema_slow and ema_fast_prev >= ema_slow_prev:
            # 死叉 - 做空
            if has_position:
                current_side = portfolio.positions[symbol].side
                if current_side == Side.LONG:
                    return {'action': 'short', 'confidence': 0.7, 'reason': 'death_cross_reverse'}
                return {'action': 'hold', 'confidence': 0.5, 'reason': 'already_short'}
            return {'action': 'short', 'confidence': 0.7, 'reason': 'death_cross'}
        
        return {'action': 'hold', 'confidence': 0.3, 'reason': 'no_signal'}
    
    def stop(self):
        """停止回测"""
        self.is_running = False
    
    def generate_report(self, result: BacktestResult, filename: str = None) -> str:
        """
        生成回测报告
        
        Args:
            result: 回测结果
            filename: 文件名
            
        Returns:
            报告文件路径
        """
        report = BacktestReport()
        
        config_dict = {
            'symbol': self.config.symbol,
            'initial_capital': self.config.initial_capital,
            'start_date': self.config.start_date,
            'end_date': self.config.end_date,
        }
        
        trades_df = self.portfolio.get_trades_dataframe() if self.portfolio else pd.DataFrame()
        
        filepath = report.generate(
            metrics=result.metrics,
            equity_curve=result.equity_curve,
            trades_df=trades_df,
            config=config_dict,
            filename=filename
        )
        
        log.info(f"📄 Report saved to: {filepath}")
        return filepath


# CLI 入口支持
async def run_backtest_cli(
    symbol: str = "BTCUSDT",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-01",
    initial_capital: float = 10000,
    step: int = 3
) -> BacktestResult:
    """
    CLI 运行回测
    
    Args:
        symbol: 交易对
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金
        step: 时间步长
        
    Returns:
        BacktestResult
    """
    config = BacktestConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        step=step
    )
    
    engine = BacktestEngine(config)
    
    def progress(current, total, pct):
        print(f"\rProgress: {current}/{total} ({pct:.1f}%)", end="", flush=True)
    
    result = await engine.run(progress_callback=progress)
    print()  # 换行
    
    # 生成报告
    report_path = engine.generate_report(result)
    print(f"\n📄 Report: {report_path}")
    
    return result


# 测试函数
async def test_backtest_engine():
    """测试回测引擎"""
    print("\n" + "=" * 60)
    print("🧪 Testing BacktestEngine")
    print("=" * 60)
    
    config = BacktestConfig(
        symbol="BTCUSDT",
        start_date="2024-12-01",
        end_date="2024-12-07",
        initial_capital=10000,
        step=12  # 每小时一个决策点
    )
    
    engine = BacktestEngine(config)
    
    def progress(current, total, pct):
        if current % 10 == 0:
            print(f"   Progress: {pct:.1f}%")
    
    result = await engine.run(progress_callback=progress)
    
    print(f"\n📊 Results:")
    print(f"   Total Return: {result.metrics.total_return:+.2f}%")
    print(f"   Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")
    print(f"   Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"   Total Trades: {result.metrics.total_trades}")
    
    # 生成报告
    report_path = engine.generate_report(result, "test_backtest")
    print(f"\n📄 Report: {report_path}")
    
    print("\n✅ BacktestEngine test complete!")
    return result


if __name__ == "__main__":
    asyncio.run(test_backtest_engine())
