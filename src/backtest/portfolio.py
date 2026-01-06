"""
虚拟投资组合管理 (Backtest Portfolio)
======================================

管理回测中的虚拟持仓、交易记录和净值曲线

Author: AI Trader Team
Date: 2025-12-31
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

from src.utils.logger import log


class Side(Enum):
    """交易方向"""
    LONG = "long"
    SHORT = "short"


class MarginMode(Enum):
    """保证金模式"""
    CROSS = "cross"      # 全仓模式
    ISOLATED = "isolated"  # 逐仓模式


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"    # 市价单 (Taker)
    LIMIT = "limit"      # 限价单 (可能是 Maker)


@dataclass
class FeeStructure:
    """
    手续费结构
    
    Binance 期货默认费率：
    - 普通用户: Maker 0.02%, Taker 0.04%
    - VIP1: Maker 0.016%, Taker 0.04%
    - 持有 BNB 可享 10% 折扣
    """
    maker_fee: float = 0.0002   # 0.02% 挂单成交
    taker_fee: float = 0.0004   # 0.04% 吃单成交
    
    # Binance VIP 费率预设
    @classmethod
    def binance_vip0(cls) -> 'FeeStructure':
        return cls(maker_fee=0.0002, taker_fee=0.0004)
    
    @classmethod
    def binance_vip1(cls) -> 'FeeStructure':
        return cls(maker_fee=0.00016, taker_fee=0.0004)
    
    @classmethod
    def binance_vip2(cls) -> 'FeeStructure':
        return cls(maker_fee=0.00014, taker_fee=0.00035)
    
    @classmethod
    def binance_with_bnb(cls) -> 'FeeStructure':
        """使用 BNB 支付手续费 (10% 折扣)"""
        return cls(maker_fee=0.00018, taker_fee=0.00036)
    
    def get_fee(self, is_maker: bool) -> float:
        """获取费率"""
        return self.maker_fee if is_maker else self.taker_fee


@dataclass
class MarginConfig:
    """
    保证金配置
    
    用于模拟交易所的保证金和强平机制
    """
    mode: MarginMode = MarginMode.CROSS
    leverage: int = 10
    margin_type: str = "USDT"  # "USDT" 或 "COIN" (币本位)
    
    # 维持保证金率 (Binance 默认阶梯)
    # 第一档：0-50,000 USDT 仓位，维持保证金率 0.4%
    maintenance_margin_rate: float = 0.004  # 0.4%
    
    # 强平费率
    liquidation_fee: float = 0.005  # 0.5%
    
    # 阶梯保证金表 (简化版 Binance BTCUSDT)
    # 格式: [(最大仓位值, 维持保证金率), ...]
    tiered_margins: List = field(default_factory=lambda: [
        (50000, 0.004),      # 0-50k: 0.4%
        (250000, 0.005),     # 50k-250k: 0.5%
        (1000000, 0.01),     # 250k-1M: 1%
        (5000000, 0.025),    # 1M-5M: 2.5%
        (20000000, 0.05),    # 5M-20M: 5%
        (float('inf'), 0.1), # 20M+: 10%
    ])
    
    def get_maintenance_margin_rate(self, position_value: float) -> float:
        """根据仓位大小获取对应的维持保证金率"""
        for max_value, rate in self.tiered_margins:
            if position_value <= max_value:
                return rate
        return self.tiered_margins[-1][1]


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    contract_type: str = "linear"  # "linear" 或 "inverse"
    contract_size: float = 1.0     # 币本位合约面值
    trailing_stop_pct: Optional[float] = None
    highest_price: float = 0.0      # For Long Trailing
    lowest_price: float = float('inf') # For Short Trailing
    
    @property
    def notional_value(self) -> float:
        """头寸名义价值"""
        if self.contract_type == "inverse":
            # 币本位：名义价值 = 合约数 * 合约面值
            return self.quantity * self.contract_size
        return self.quantity * self.entry_price
    
    def get_pnl(self, current_price: float) -> float:
        """
        计算当前盈亏
        
        U本位: PnL = (exit - entry) * qty
        币本位: PnL = (1/entry - 1/exit) * contracts * size (以币计价)
        """
        if self.contract_type == "inverse":
            # 币本位计算 (返回币种单位，需要转换为 USD)
            if self.side == Side.LONG:
                pnl_coin = (1/self.entry_price - 1/current_price) * self.quantity * self.contract_size
            else:
                pnl_coin = (1/current_price - 1/self.entry_price) * self.quantity * self.contract_size
            # 转换为 USD
            return pnl_coin * current_price
        else:
            # U本位计算
            if self.side == Side.LONG:
                return (current_price - self.entry_price) * self.quantity
            else:
                return (self.entry_price - current_price) * self.quantity
    
    def get_pnl_pct(self, current_price: float) -> float:
        """计算盈亏百分比"""
        if self.entry_price == 0:
            return 0.0
        pnl = self.get_pnl(current_price)
        return pnl / self.notional_value * 100
    
    def should_stop_loss(self, current_price: float) -> bool:
        """是否触发止损"""
        if self.stop_loss is None:
            return False
        if self.side == Side.LONG:
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss
    
    def should_take_profit(self, current_price: float) -> bool:
        """是否触发止盈"""
        if self.take_profit is None:
            return False
        if self.side == Side.LONG:
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit

    def update_price(self, current_price: float):
        """Update high/low watermark for trailing stop"""
        if self.side == Side.LONG:
            if current_price > self.highest_price:
                self.highest_price = current_price
        else:
            if current_price < self.lowest_price:
                self.lowest_price = current_price

    def should_trailing_stop(self, current_price: float) -> bool:
        """Check if trailing stop is triggered"""
        if self.trailing_stop_pct is None:
            return False
            
        if self.side == Side.LONG:
            # If price drops X% from high
            stop_price = self.highest_price * (1 - self.trailing_stop_pct / 100)
            return current_price <= stop_price
        else:
            # If price rises X% from low
            stop_price = self.lowest_price * (1 + self.trailing_stop_pct / 100)
            return current_price >= stop_price

@dataclass
class Trade:
    """交易记录"""
    trade_id: int
    symbol: str
    side: Side
    action: str  # "open" or "close"
    quantity: float
    price: float
    timestamp: datetime
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    
    # 关联信息
    entry_price: Optional[float] = None  # 平仓时的开仓价
    holding_time: Optional[float] = None  # 持仓时间（小时）
    close_reason: Optional[str] = None  # 平仓原因：signal/stop_loss/take_profit
    
    def to_dict(self) -> Dict:
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'action': self.action,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'commission': self.commission,
            'slippage': self.slippage,
            'entry_price': self.entry_price,
            'holding_time': self.holding_time,
            'close_reason': self.close_reason,
        }


@dataclass
class EquityPoint:
    """净值曲线点"""
    timestamp: datetime
    cash: float
    position_value: float
    total_equity: float
    drawdown: float = 0.0
    drawdown_pct: float = 0.0


class BacktestPortfolio:
    """
    虚拟投资组合管理
    
    功能：
    1. 管理现金和持仓
    2. 记录所有交易
    3. 跟踪净值曲线
    4. 计算实时盈亏
    """
    
    def __init__(
        self,
        initial_capital: float,
        slippage: float = 0.001,
        commission: float = 0.0004,
        margin_config: MarginConfig = None,
        fee_structure: FeeStructure = None
    ):
        """
        初始化投资组合
        
        Args:
            initial_capital: 初始资金 (USDT)
            slippage: 基础滑点 (0.001 = 0.1%)
            commission: 默认手续费 (已弃用，使用 fee_structure)
            margin_config: 保证金配置
            fee_structure: 手续费结构（Maker/Taker）
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.slippage = slippage
        self.commission = commission
        self.margin_config = margin_config or MarginConfig()
        self.fee_structure = fee_structure or FeeStructure()
        
        # 持仓 (symbol -> Position)
        self.positions: Dict[str, Position] = {}
        
        # 交易记录
        self.trades: List[Trade] = []
        self.trade_counter = 0
        
        # 净值曲线
        self.equity_curve: List[EquityPoint] = []
        self.peak_equity = initial_capital
        
        # 资金费率追踪
        self.total_funding_paid: float = 0.0
        self.funding_history: List[Dict] = []
        
        # 强平追踪
        self.liquidation_count: int = 0
        self.liquidation_history: List[Dict] = []
        
        # 费用追踪
        self.total_fees_paid: float = 0.0
        self.total_slippage_cost: float = 0.0
        
        log.info(f"💼 Portfolio initialized | Capital: ${initial_capital:.2f} | "
                 f"Leverage: {self.margin_config.leverage}x | "
                 f"Mode: {self.margin_config.mode.value}")
    
    def apply_funding_fee(
        self,
        symbol: str,
        funding_rate: float,
        mark_price: float,
        timestamp: datetime
    ) -> float:
        """
        应用资金费率结算
        
        永续合约资金费率机制：
        - 多头持仓：funding_rate > 0 时支付，< 0 时收取
        - 空头持仓：funding_rate > 0 时收取，< 0 时支付
        
        计算公式：Funding Fee = Position Size * funding_rate
        
        Args:
            symbol: 交易对
            funding_rate: 资金费率 (e.g., 0.0001 = 0.01%)
            mark_price: 标记价格（用于计算仓位价值）
            timestamp: 结算时间
            
        Returns:
            实际支付/收取的资金费用（负数表示支付）
        """
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        
        # 计算仓位名义价值
        position_value = position.quantity * mark_price
        
        # 计算资金费用
        funding_fee = position_value * abs(funding_rate)
        
        # 根据持仓方向和费率方向决定支付/收取
        if position.side == Side.LONG:
            if funding_rate > 0:
                # 多头支付
                fee_impact = -funding_fee
            else:
                # 多头收取
                fee_impact = funding_fee
        else:  # SHORT
            if funding_rate > 0:
                # 空头收取
                fee_impact = funding_fee
            else:
                # 空头支付
                fee_impact = -funding_fee
        
        # 更新现金
        self.cash += fee_impact
        self.total_funding_paid += fee_impact
        
        # 记录
        self.funding_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'side': position.side.value,
            'position_value': position_value,
            'funding_rate': funding_rate,
            'fee_impact': fee_impact
        })
        
        log.debug(f"💸 Funding settled: {symbol} | Rate: {funding_rate*100:.4f}% | "
                  f"Impact: ${fee_impact:.4f}")
        
        return fee_impact
    
    def check_liquidation(
        self,
        prices: Dict[str, float],
        timestamp: datetime
    ) -> List[str]:
        """
        检查并执行强平
        
        强平条件：账户权益 < 维持保证金
        
        全仓模式：所有持仓共享保证金
        逐仓模式：每个持仓独立计算
        
        Args:
            prices: 当前市场价格 {symbol: price}
            timestamp: 当前时间
            
        Returns:
            被强平的symbol列表
        """
        liquidated_symbols = []
        
        if self.margin_config.mode == MarginMode.CROSS:
            # 全仓模式：计算总权益和总维持保证金
            total_position_value = 0.0
            total_unrealized_pnl = 0.0
            
            for symbol, position in self.positions.items():
                current_price = prices.get(symbol, position.entry_price)
                position_value = position.quantity * current_price
                pnl = position.get_pnl(current_price)
                
                total_position_value += position_value
                total_unrealized_pnl += pnl
            
            # 账户权益 = 现金 + 未实现盈亏
            total_equity = self.cash + total_unrealized_pnl
            
            # 维持保证金 = 仓位价值 * 维持保证金率
            mm_rate = self.margin_config.get_maintenance_margin_rate(total_position_value)
            maintenance_margin = total_position_value * mm_rate
            
            # 检查是否触发强平
            if self.positions and total_equity < maintenance_margin:
                # 强平所有持仓
                log.warning(f"⚠️ LIQUIDATION TRIGGERED | Equity: ${total_equity:.2f} < "
                           f"Maintenance: ${maintenance_margin:.2f}")
                
                for symbol in list(self.positions.keys()):
                    current_price = prices.get(symbol, 0)
                    self._execute_liquidation(symbol, current_price, timestamp, total_equity, maintenance_margin)
                    liquidated_symbols.append(symbol)
        
        else:
            # 逐仓模式：每个仓位独立检查
            for symbol, position in list(self.positions.items()):
                current_price = prices.get(symbol, position.entry_price)
                position_value = position.quantity * current_price
                pnl = position.get_pnl(current_price)
                
                # 逐仓权益 = 初始保证金 + 未实现盈亏
                initial_margin = position_value / self.margin_config.leverage
                isolated_equity = initial_margin + pnl
                
                # 维持保证金
                mm_rate = self.margin_config.get_maintenance_margin_rate(position_value)
                maintenance_margin = position_value * mm_rate
                
                if isolated_equity < maintenance_margin:
                    log.warning(f"⚠️ ISOLATED LIQUIDATION: {symbol} | "
                               f"Equity: ${isolated_equity:.2f} < MM: ${maintenance_margin:.2f}")
                    self._execute_liquidation(symbol, current_price, timestamp, isolated_equity, maintenance_margin)
                    liquidated_symbols.append(symbol)
        
        return liquidated_symbols
    
    def _execute_liquidation(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        equity: float,
        maintenance_margin: float
    ):
        """执行强平"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # 计算强平损失（接近但不超过全部保证金）
        pnl = position.get_pnl(price)
        liquidation_fee = position.quantity * price * self.margin_config.liquidation_fee
        
        # 更新现金（强平后损失）
        initial_margin = position.quantity * position.entry_price / self.margin_config.leverage
        loss = -initial_margin + min(pnl, 0) - liquidation_fee
        self.cash = max(0, self.cash + loss)
        
        # 记录强平交易
        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            side=position.side,
            action="liquidation",
            quantity=position.quantity,
            price=price,
            timestamp=timestamp,
            pnl=pnl - liquidation_fee,
            pnl_pct=position.get_pnl_pct(price),
            entry_price=position.entry_price,
            holding_time=(timestamp - position.entry_time).total_seconds() / 3600,
            close_reason="liquidation"
        )
        self.trades.append(trade)
        
        # 记录强平历史
        self.liquidation_count += 1
        self.liquidation_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'side': position.side.value,
            'price': price,
            'equity': equity,
            'maintenance_margin': maintenance_margin,
            'loss': loss
        })
        
        # 移除持仓
        del self.positions[symbol]
        
        log.error(f"🔥 LIQUIDATED: {symbol} {position.side.value} @ ${price:.2f} | "
                  f"Loss: ${loss:.2f}")
    
    def open_position(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        price: float,
        timestamp: datetime,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
        trailing_stop_pct: float = None
    ) -> Optional[Trade]:
        """
        开仓
        
        Args:
            symbol: 交易对
            side: 交易方向 (LONG/SHORT)
            quantity: 数量
            price: 开仓价格
            timestamp: 时间戳
            stop_loss_pct: 止损百分比 (e.g., 1.0 = 1%)
            take_profit_pct: 止盈百分比 (e.g., 2.0 = 2%)
            
        Returns:
            Trade 对象，或 None（如果开仓失败）
        """
        # 检查是否已有持仓
        if symbol in self.positions:
            log.warning(f"Position already exists for {symbol}, close it first")
            return None
        
        # 计算滑点后的价格
        slippage_impact = price * self.slippage
        if side == Side.LONG:
            exec_price = price + slippage_impact  # 买入时滑点向上
        else:
            exec_price = price - slippage_impact  # 卖出时滑点向下
        
        # 计算手续费
        notional = quantity * exec_price
        commission = notional * self.commission
        
        # 计算初始保证金
        initial_margin = notional / self.margin_config.leverage
        
        # 检查资金是否足够（需要支付保证金 + 手续费）
        total_cost = initial_margin + commission
        if total_cost > self.cash:
            log.warning(f"Insufficient cash: ${self.cash:.2f} < ${total_cost:.2f} (Margin: ${initial_margin:.2f}, Fee: ${commission:.2f})")
            return None
        
        # 计算止损止盈价格
        stop_loss = None
        take_profit = None
        if stop_loss_pct is not None and stop_loss_pct > 0:
            if side == Side.LONG:
                stop_loss = exec_price * (1 - stop_loss_pct / 100)
            else:
                stop_loss = exec_price * (1 + stop_loss_pct / 100)
        if take_profit_pct is not None and take_profit_pct > 0:
            if side == Side.LONG:
                take_profit = exec_price * (1 + take_profit_pct / 100)
            else:
                take_profit = exec_price * (1 - take_profit_pct / 100)
        
        # 创建持仓
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=exec_price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
            highest_price=exec_price,
            lowest_price=exec_price
        )
        self.positions[symbol] = position
        
        # 扣除资金（保证金 + 手续费）
        self.cash -= total_cost
        
        # 记录交易
        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            side=side,
            action="open",
            quantity=quantity,
            price=exec_price,
            timestamp=timestamp,
            commission=commission,
            slippage=slippage_impact * quantity
        )
        self.trades.append(trade)
        
        sl_str = f"${stop_loss:.2f}" if stop_loss else "N/A"
        tp_str = f"${take_profit:.2f}" if take_profit else "N/A"
        log.info(f"📈 Opened {side.value.upper()} | {symbol} | "
                 f"Qty: {quantity:.4f} @ ${exec_price:.2f} | "
                 f"SL: {sl_str} | TP: {tp_str}")
        
        return trade
    
    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        reason: str = "signal"
    ) -> Optional[Trade]:
        """
        平仓
        
        Args:
            symbol: 交易对
            price: 平仓价格
            timestamp: 时间戳
            reason: 平仓原因 (signal/stop_loss/take_profit)
            
        Returns:
            Trade 对象，或 None（如果平仓失败）
        """
        if symbol not in self.positions:
            log.warning(f"No position for {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # 计算滑点后的价格
        slippage_impact = price * self.slippage
        if position.side == Side.LONG:
            exec_price = price - slippage_impact  # 卖出时滑点向下
        else:
            exec_price = price + slippage_impact  # 买回时滑点向上
        
        # 计算盈亏
        pnl = position.get_pnl(exec_price)
        pnl_pct = position.get_pnl_pct(exec_price)
        
        # 计算手续费
        notional = position.quantity * exec_price
        commission = notional * self.commission
        
        # 计算持仓时间
        holding_time = (timestamp - position.entry_time).total_seconds() / 3600
        
        # 计算初始保证金
        initial_margin = position.quantity * position.entry_price / self.margin_config.leverage
        
        # 更新资金: 返还保证金 + 盈亏 - 手续费
        self.cash += initial_margin + pnl - commission
        
        # 记录交易
        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            side=position.side,
            action="close",
            quantity=position.quantity,
            price=exec_price,
            timestamp=timestamp,
            pnl=pnl,  # 原始PnL（未扣手续费）
            pnl_pct=pnl_pct,
            commission=commission,
            slippage=slippage_impact * position.quantity,
            entry_price=position.entry_price,
            holding_time=holding_time,
            close_reason=reason
        )
        self.trades.append(trade)
        
        # 删除持仓
        del self.positions[symbol]
        
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        log.info(f"📉 Closed {position.side.value.upper()} | {symbol} | "
                 f"PnL: {pnl_str} ({pnl_pct:+.2f}%) | "
                 f"Hold: {holding_time:.1f}h | Reason: {reason}")
        
        return trade
    
    def check_stop_loss_take_profit(
        self,
        current_prices: Dict[str, float],
        timestamp: datetime
    ) -> List[Trade]:
        """
        检查所有持仓的止损止盈
        
        Returns:
            触发的平仓交易列表
        """
        triggered_trades = []
        symbols_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue
            
            price = current_prices[symbol]
            
            # Update high/low watermark
            position.update_price(price)
            
            if position.should_stop_loss(price):
                symbols_to_close.append((symbol, price, "stop_loss"))
            elif position.should_take_profit(price):
                symbols_to_close.append((symbol, price, "take_profit"))
            elif position.should_trailing_stop(price):
                symbols_to_close.append((symbol, price, "trailing_stop"))
        
        for symbol, price, reason in symbols_to_close:
            trade = self.close_position(symbol, price, timestamp, reason)
            if trade:
                triggered_trades.append(trade)
        
        return triggered_trades

    def check_stop_loss_take_profit_intrabar(
        self,
        bars: Dict[str, Dict[str, float]],
        timestamp: datetime
    ) -> List[Trade]:
        """
        Intrabar SL/TP check using bar high/low.

        Uses conservative ordering: stop_loss -> take_profit -> trailing_stop.
        """
        triggered_trades = []
        symbols_to_close = []

        for symbol, position in self.positions.items():
            bar = bars.get(symbol)
            if not isinstance(bar, dict):
                continue

            high = bar.get('high')
            low = bar.get('low')
            if not isinstance(high, (int, float)) or not isinstance(low, (int, float)):
                # Fallback to close/open if high/low missing
                fallback = bar.get('close', bar.get('open', 0.0))
                high = fallback
                low = fallback

            # Update watermarks for trailing stop with intrabar extremes
            if position.side == Side.LONG:
                position.update_price(high)
                if position.stop_loss is not None and low <= position.stop_loss:
                    symbols_to_close.append((symbol, position.stop_loss, "stop_loss"))
                    continue
                if position.take_profit is not None and high >= position.take_profit:
                    symbols_to_close.append((symbol, position.take_profit, "take_profit"))
                    continue
                if position.trailing_stop_pct is not None:
                    stop_price = position.highest_price * (1 - position.trailing_stop_pct / 100)
                    if low <= stop_price:
                        symbols_to_close.append((symbol, stop_price, "trailing_stop"))
            else:
                position.update_price(low)
                if position.stop_loss is not None and high >= position.stop_loss:
                    symbols_to_close.append((symbol, position.stop_loss, "stop_loss"))
                    continue
                if position.take_profit is not None and low <= position.take_profit:
                    symbols_to_close.append((symbol, position.take_profit, "take_profit"))
                    continue
                if position.trailing_stop_pct is not None:
                    stop_price = position.lowest_price * (1 + position.trailing_stop_pct / 100)
                    if high >= stop_price:
                        symbols_to_close.append((symbol, stop_price, "trailing_stop"))

        for symbol, price, reason in symbols_to_close:
            trade = self.close_position(symbol, price, timestamp, reason)
            if trade:
                triggered_trades.append(trade)

        return triggered_trades
    
    def get_current_equity(self, current_prices: Dict[str, float]) -> float:
        """
        计算当前总净值
        
        Args:
            current_prices: 当前价格字典 {symbol: price}
            
        Returns:
            总净值 (现金/可用余额 + 占用保证金 + 持仓未实现盈亏)
        """
        unrealized_pnl = 0.0
        used_margin = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                # 1. 累加未实现盈亏
                pnl = position.get_pnl(current_prices[symbol])
                unrealized_pnl += pnl
                
                # 2. 累加占用保证金
                # 注意：目前简化假设占用保证金固定为 initial_margin
                # 实际上应该基于 position.entry_price 计算，而不是当前价格
                # 全仓模式下 margin = quantity * entry_price / leverage
                margin = position.notional_value / self.margin_config.leverage
                used_margin += margin
        
        # 净值 = 现金(可用余额) + 占用保证金 + 未实现盈亏
        return self.cash + used_margin + unrealized_pnl
    
    def record_equity(
        self,
        timestamp: datetime,
        current_prices: Dict[str, float]
    ):
        """记录净值曲线点"""
        total_equity = self.get_current_equity(current_prices)
        
        # 计算持仓价值
        position_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value += position.notional_value + position.get_pnl(current_prices[symbol])
        
        # 更新峰值
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
        
        # 计算回撤
        drawdown = self.peak_equity - total_equity
        drawdown_pct = drawdown / self.peak_equity * 100 if self.peak_equity > 0 else 0
        
        point = EquityPoint(
            timestamp=timestamp,
            cash=self.cash,
            position_value=position_value,
            total_equity=total_equity,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct
        )
        self.equity_curve.append(point)
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """获取净值曲线 DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()
        
        data = [
            {
                'timestamp': p.timestamp,
                'cash': p.cash,
                'position_value': p.position_value,
                'total_equity': p.total_equity,
                'drawdown': p.drawdown,
                'drawdown_pct': p.drawdown_pct,
            }
            for p in self.equity_curve
        ]
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """获取交易记录 DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        data = [t.to_dict() for t in self.trades]
        return pd.DataFrame(data)
    
    def get_summary(self) -> Dict:
        """获取投资组合摘要"""
        total_equity = self.cash
        for symbol, pos in self.positions.items():
            total_equity += pos.notional_value
        
        return {
            'initial_capital': self.initial_capital,
            'current_cash': self.cash,
            'total_equity': total_equity,
            'total_return': (total_equity - self.initial_capital) / self.initial_capital * 100,
            'open_positions': len(self.positions),
            'total_trades': len(self.trades),
            'peak_equity': self.peak_equity,
            'max_drawdown': max((p.drawdown for p in self.equity_curve), default=0),
            'max_drawdown_pct': max((p.drawdown_pct for p in self.equity_curve), default=0),
        }


# 测试函数
def test_portfolio():
    """测试投资组合"""
    print("\n" + "=" * 60)
    print("🧪 Testing BacktestPortfolio")
    print("=" * 60)
    
    # 创建投资组合
    portfolio = BacktestPortfolio(
        initial_capital=10000,
        slippage=0.001,
        commission=0.0004
    )
    
    # 开仓
    now = datetime.now()
    trade1 = portfolio.open_position(
        symbol="BTCUSDT",
        side=Side.LONG,
        quantity=0.01,
        price=50000,
        timestamp=now,
        stop_loss_pct=1.0,
        take_profit_pct=2.0
    )
    print(f"\n✅ Opened position: {trade1}")
    
    # 记录净值
    prices = {"BTCUSDT": 50500}
    portfolio.record_equity(now, prices)
    
    # 检查止损止盈
    prices = {"BTCUSDT": 51000}  # 价格上涨 2%
    triggered = portfolio.check_stop_loss_take_profit(prices, now)
    print(f"\n✅ Triggered trades: {len(triggered)}")
    
    # 获取摘要
    summary = portfolio.get_summary()
    print(f"\n📊 Portfolio Summary:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    
    print("\n✅ BacktestPortfolio test complete!")


if __name__ == "__main__":
    test_portfolio()
