"""
⚖️ 对抗评论员 (The Critic) Agent
===========================================

职责:
1. 加权投票机制 - 整合量化分析师的多个信号源
2. 动态权重调整 - 根据历史表现调整各信号权重
3. 多周期对齐决策 - 优先级: 1h > 15m > 5m
4. LLM决策增强 - 将量化信号作为上下文传递给DeepSeek
5. 最终决策输出 - 统一格式{action, confidence, reason}

Author: AI Trader Team
Date: 2025-12-19
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json

import pandas as pd

from src.utils.logger import log
from src.agents.position_analyzer import PositionAnalyzer
from src.agents.regime_detector import RegimeDetector
from src.agents.predict_agent import PredictResult


# ============================================
# 过度交易防护 (Overtrading Guard)
# ============================================
@dataclass
class TradeRecord:
    """交易记录"""
    symbol: str
    action: str
    timestamp: datetime
    pnl: float = 0.0


class OvertradingGuard:
    """
    过度交易防护 - 防止频繁交易和连续亏损
    
    规则:
    - 同一symbol最少间隔2个周期
    - 6小时内最多3个新仓位
    - 连续2次亏损后，需要等待4个周期
    """
    
    MIN_CYCLES_SAME_SYMBOL = 4        # 同symbol最小间隔周期 (1小时)
    MAX_POSITIONS_6H = 2              # 6小时内最多开仓数 (减少过度交易)
    LOSS_STREAK_COOLDOWN = 6          # 连续亏损后冷却周期 (增加冷却时间)
    CONSECUTIVE_LOSS_THRESHOLD = 2   # 触发冷却的连续亏损次数
    
    def __init__(self):
        self.trade_history: List[TradeRecord] = []
        self.consecutive_losses = 0
        self.last_trade_cycle: Dict[str, int] = {}  # symbol -> cycle
        self.cooldown_until_cycle: int = 0
    
    def record_trade(self, symbol: str, action: str, pnl: float = 0.0, current_cycle: int = 0):
        """记录一笔交易"""
        self.trade_history.append(TradeRecord(
            symbol=symbol,
            action=action,
            timestamp=datetime.now(),
            pnl=pnl
        ))
        self.last_trade_cycle[symbol] = current_cycle
        
        # 追踪连续亏损
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.CONSECUTIVE_LOSS_THRESHOLD:
                self.cooldown_until_cycle = current_cycle + self.LOSS_STREAK_COOLDOWN
                log.warning(f"⚠️ 连续{self.consecutive_losses}次亏损，冷却至周期 {self.cooldown_until_cycle}")
        else:
            self.consecutive_losses = 0
    
    def can_open_position(self, symbol: str, current_cycle: int = 0) -> Tuple[bool, str]:
        """
        检查是否可以开仓
        
        Returns:
            (allowed, reason)
        """
        # 检查冷却期
        if current_cycle < self.cooldown_until_cycle:
            remaining = self.cooldown_until_cycle - current_cycle
            return False, f"⛔ 连续亏损冷却中，剩余{remaining}周期"
        
        # 检查同symbol间隔
        if symbol in self.last_trade_cycle:
            cycles_since = current_cycle - self.last_trade_cycle[symbol]
            if cycles_since < self.MIN_CYCLES_SAME_SYMBOL:
                return False, f"⛔ {symbol}交易间隔不足，需等待{self.MIN_CYCLES_SAME_SYMBOL - cycles_since}周期"
        
        # 检查6小时内开仓数
        six_hours_ago = datetime.now().timestamp() - 6 * 3600
        recent_opens = sum(
            1 for t in self.trade_history 
            if t.timestamp.timestamp() > six_hours_ago and 'open' in t.action.lower()
        )
        if recent_opens >= self.MAX_POSITIONS_6H:
            return False, f"⛔ 6小时内已开{recent_opens}仓，已达上限{self.MAX_POSITIONS_6H}"
        
        return True, "✅ 允许开仓"
    
    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            'consecutive_losses': self.consecutive_losses,
            'cooldown_until': self.cooldown_until_cycle,
            'recent_trades': len(self.trade_history),
            'symbols_traded': list(self.last_trade_cycle.keys())
        }


@dataclass
class SignalWeight:
    """信号权重配置
    
    注意: 所有权重应该合计为 1.0 (不包括动态 sentiment)
    优化后配置 (2026-01-07): 基于回测分析进一步优化
    - 增加1h权重，减少短周期噪音
    - 减少prophet权重，更依赖技术指标
    """
    # 趋势信号 (合计 0.45) - 增加长周期权重
    trend_5m: float = 0.03   # 减少5m噪音影响
    trend_15m: float = 0.12  # 略增
    trend_1h: float = 0.30   # 增加1h权重 (核心趋势判断)
    # 震荡信号 (合计 0.20)
    oscillator_5m: float = 0.03  # 减少5m噪音
    oscillator_15m: float = 0.07
    oscillator_1h: float = 0.10  # 增加1h权重
    # Prophet ML 预测权重 - 进一步减少
    prophet: float = 0.05  # 减少ML权重，避免过拟合
    # 情绪信号 (动态权重)
    sentiment: float = 0.25
    # 其他扩展信号（如LLM）
    llm_signal: float = 0.0  # 待整合


@dataclass
class VoteResult:
    """投票结果"""
    action: str  # 'long', 'short', 'close_long', 'close_short', 'hold'
    confidence: float  # 0-100
    weighted_score: float  # -100 ~ +100
    vote_details: Dict[str, float]  # 各信号的贡献分
    multi_period_aligned: bool  # 多周期是否一致
    reason: str  # 决策原因
    regime: Optional[Dict] = None      # 市场状态信息
    position: Optional[Dict] = None    # 价格位置信息
    trade_params: Optional[Dict] = None # 动态交易参数 (stop_loss, take_profit, leverage, etc.)


class DecisionCoreAgent:
    """对抗评论员 (The Critic)
    
    核心功能:
    - 加权投票: 根据可配置权重整合多个信号
    - 多周期对齐: 检测多周期趋势一致性
    - 市场感知: 集成位置感知和状态检测
    - 信心增强: 基于市场状态和价格位置校准信心度
    """
    
    def __init__(self, weights: Optional[SignalWeight] = None):
        """
        初始化对抗评论员 (The Critic)
        
        Args:
            weights: 自定义信号权重（默认使用内置配置）
        """
        self.weights = weights or SignalWeight()
        self.history: List[VoteResult] = []  # 历史决策记录
        
        # 初始化辅助分析器
        self.position_analyzer = PositionAnalyzer()
        self.regime_detector = RegimeDetector()
        
        self.performance_tracker = {
            'trend_5m': {'total': 0, 'correct': 0},
            'trend_15m': {'total': 0, 'correct': 0},
            'trend_1h': {'total': 0, 'correct': 0},
            'oscillator_5m': {'total': 0, 'correct': 0},
            'oscillator_15m': {'total': 0, 'correct': 0},
            'oscillator_1h': {'total': 0, 'correct': 0},
        }
        
        # 初始化交易防护
        self.overtrading_guard = OvertradingGuard()
        self.current_cycle = 0  # 当前周期计数
        
    async def make_decision(
        self, 
        quant_analysis: Dict, 
        predict_result: Optional[PredictResult] = None,
        market_data: Optional[Dict] = None
    ) -> VoteResult:
        """
        执行加权投票决策
        
        Args:
            quant_analysis: QuantAnalystAgent的输出
            predict_result: PredictAgent的输出 (ML预测)
            market_data: 包含 df_5m, df_15m, df_1h 和 current_price 的原始市场数据
            
        Returns:
            VoteResult对象
        """
        # 更新周期计数
        self.current_cycle += 1
        symbol = quant_analysis.get('symbol', 'UNKNOWN')
        
        # ========== 过度交易检查 ==========
        overtrade_allowed, overtrade_reason = self.overtrading_guard.can_open_position(
            symbol, self.current_cycle
        )
        
        # 1. 提取各信号分数
        # Fix: Read from granular scores provided by QuantAnalystAgent
        trend_data = quant_analysis.get('trend', {})
        osc_data = quant_analysis.get('oscillator', {})
        sentiment_data = quant_analysis.get('sentiment', {})
        
        scores = {
            'trend_5m': trend_data.get('trend_5m_score', 0),
            'trend_15m': trend_data.get('trend_15m_score', 0),
            'trend_1h': trend_data.get('trend_1h_score', 0),
            'oscillator_5m': osc_data.get('osc_5m_score', 0),
            'oscillator_15m': osc_data.get('osc_15m_score', 0),
            'oscillator_1h': osc_data.get('osc_1h_score', 0),
            'sentiment': sentiment_data.get('total_sentiment_score', 0)
        }
        
        # 集成 Prophet 预测得分
        if predict_result:
            # 将概率 (0~1) 映射到分数 (-100~+100)
            # 0.5 -> 0, 1.0 -> 100, 0.0 -> -100
            prob = predict_result.probability_up
            prophet_score = (prob - 0.5) * 200
            scores['prophet'] = prophet_score
        else:
            scores['prophet'] = 0.0
        
        # 计算动态 sentiment 权重 (有数据时使用配置权重，无数据时为 0)
        has_sentiment = scores.get('sentiment', 0) != 0
        w_sentiment = self.weights.sentiment if has_sentiment else 0.0
        w_others = 1.0 - w_sentiment

        # 2. 市场状态与位置分析
        regime = None
        position = None
        if market_data:
            df_5m = market_data.get('df_5m')
            curr_price = market_data.get('current_price')
            if df_5m is not None and curr_price is not None:
                regime = self.regime_detector.detect_regime(df_5m)
                position = self.position_analyzer.analyze_position(df_5m, curr_price)

        volume_ratio = self._get_volume_ratio(market_data.get('df_5m') if market_data else None)

        # 3. 加权计算（得分范围-100~+100）
        weighted_score = (
            (scores['trend_5m'] * self.weights.trend_5m +
             scores['trend_15m'] * self.weights.trend_15m +
             scores['trend_1h'] * self.weights.trend_1h +
             scores['oscillator_5m'] * self.weights.oscillator_5m +
             scores['oscillator_15m'] * self.weights.oscillator_15m +
             scores['oscillator_1h'] * self.weights.oscillator_1h +
             scores.get('prophet', 0) * self.weights.prophet) * w_others +
            (scores.get('sentiment', 0) * w_sentiment)
        )
        
        # 4. 计算各信号的实际贡献分 (用于 dashboard 显示)
        vote_details = {
            'trend_5m': scores['trend_5m'] * self.weights.trend_5m * w_others,
            'trend_15m': scores['trend_15m'] * self.weights.trend_15m * w_others,
            'trend_1h': scores['trend_1h'] * self.weights.trend_1h * w_others,
            'oscillator_5m': scores['oscillator_5m'] * self.weights.oscillator_5m * w_others,
            'oscillator_15m': scores['oscillator_15m'] * self.weights.oscillator_15m * w_others,
            'oscillator_1h': scores['oscillator_1h'] * self.weights.oscillator_1h * w_others,
            'prophet': scores.get('prophet', 0) * self.weights.prophet * w_others,
            'sentiment': scores.get('sentiment', 0) * w_sentiment
        }
        osc_bias = (scores['oscillator_5m'] + scores['oscillator_15m'] + scores['oscillator_1h']) / 3

        # 5. 提前过滤逻辑：震荡市+位置不佳（强信号可放行）
        if regime and position:
            if regime['regime'] == 'choppy' and position['location'] == 'middle' and abs(weighted_score) < 30:
                result = VoteResult(
                    action='hold',
                    confidence=10.0,
                    weighted_score=0,
                    vote_details=vote_details,
                    multi_period_aligned=False,
                    reason=f"对抗式过滤: 震荡市且价格处于区间中部({position['position_pct']:.1f}%)，禁止开仓",
                    regime=regime,
                    position=position
                )
                self.history.append(result)
                return result
        
        # 6. 多周期对齐检测
        
        # 6. 多周期对齐检测
        aligned, alignment_reason = self._check_multi_period_alignment(
            scores['trend_1h'],
            scores['trend_15m'],
            scores['trend_5m']
        )
        
        # 7. 初始决策映射（传入 regime 以使用动态阈值）
        action, base_confidence = self._score_to_action(weighted_score, aligned, regime)

        # ========== 对齐弱时收紧趋势强度 ==========
        if action in ['long', 'short', 'open_long', 'open_short'] and regime and not aligned:
            adx = regime.get('adx', 0)
            if adx < 25:
                log.warning(f"🚫 对齐弱且ADX不足: ADX {adx:.1f} < 25")
                action = 'hold'
                base_confidence = 0.1
                alignment_reason = f"对齐弱且ADX不足(ADX {adx:.1f} < 25)"

        # ========== 低量/弱趋势过滤 ==========
        if action in ['long', 'short', 'open_long', 'open_short'] and regime:
            adx = regime.get('adx', 0)
            if volume_ratio is not None and adx < 20 and volume_ratio < 0.8:
                if abs(weighted_score) < 35:
                    log.warning(f"🚫 低量/弱趋势过滤: ADX {adx:.1f}, RVOL {volume_ratio:.2f}")
                    action = 'hold'
                    base_confidence = 0.1
                    alignment_reason = f"低量/弱趋势过滤(ADX {adx:.1f}, RVOL {volume_ratio:.2f})"
                else:
                    # Strong signal but weak volume: reduce confidence
                    base_confidence *= 0.85
                    alignment_reason += f" | 低量降信心(ADX {adx:.1f}, RVOL {volume_ratio:.2f})"

        # ========== 交易防护拦截 ==========
        if action in ['long', 'short', 'open_long', 'open_short']:
            # 检查过度交易
            if not overtrade_allowed:
                log.warning(f"🚫 过度交易防护: {overtrade_reason}")
                action = 'hold'
                base_confidence = 0.1
                alignment_reason = overtrade_reason
        
        # 8. 综合信心度校准与对抗审计
        final_confidence = base_confidence * 100
        
        # --- 对抗式审计: 机构资金流背离检查 ---
        sent_details = quant_analysis.get('sentiment', {}).get('details', {})
        inst_nf_1h = sent_details.get('inst_netflow_1h', 0)
        
        if action == 'open_long' and inst_nf_1h < -1000000: # 1h 机构净流出超过 1M
            final_confidence *= 0.5
            alignment_reason += " | 对抗警告: 技术看多但机构资金大额流出 (背离)"
        elif action == 'open_short' and inst_nf_1h > 1000000: # 1h 机构净流入超过 1M
            final_confidence *= 0.5
            alignment_reason += " | 对抗警告: 技术看空但机构资金大额流入 (背离)"

        if regime and position:
            final_confidence = self._calculate_comprehensive_confidence(
                final_confidence, regime, position, aligned
            )
            # 位置约束：极端高位/低位仅允许强趋势信号
            regime_type = (regime.get('regime', '') or '').lower()
            adx = regime.get('adx', 0)
            position_pct = position.get('position_pct', 50.0)
            strong_long = (
                aligned and regime_type == 'trending_up' and adx >= 28 and weighted_score >= 35
            )
            strong_short = (
                aligned and regime_type == 'trending_down' and adx >= 28 and weighted_score <= -35
            )
            very_strong_long = aligned and weighted_score >= 45
            very_strong_short = aligned and weighted_score <= -45
            fade_long = scores['trend_5m'] < 0 or scores['trend_15m'] < 0
            fade_short = scores['trend_5m'] > 0 or scores['trend_15m'] > 0
            high_extreme = position_pct >= 90
            high_zone = position_pct >= 80
            low_extreme = position_pct <= 8
            low_zone = position_pct <= 20
            # Backtest hotspot: scope high-position guard to underperforming symbols.
            apply_position_penalty = symbol in {'LINKUSDT'}
            if apply_position_penalty:
                if action in ['open_long', 'long']:
                    if high_extreme:
                        if strong_long and osc_bias > -20:
                            final_confidence *= 0.9
                            alignment_reason += f" | 极高位做多降信心({position_pct:.1f}%)"
                        elif very_strong_long and osc_bias > -25:
                            final_confidence *= 0.8
                            alignment_reason += f" | 极高位强信号降信心({position_pct:.1f}%)"
                        else:
                            final_confidence *= 0.6
                            alignment_reason += f" | 极高位做多过滤({position_pct:.1f}%)"
                    elif high_zone and osc_bias <= -30 and (fade_long or not aligned):
                        final_confidence *= 0.8
                        alignment_reason += f" | 高位超买降信心({position_pct:.1f}%)"
                elif action in ['open_short', 'short']:
                    if low_extreme:
                        if strong_short and osc_bias < 20:
                            final_confidence *= 0.9
                            alignment_reason += f" | 极低位做空降信心({position_pct:.1f}%)"
                        elif very_strong_short and osc_bias < 25:
                            final_confidence *= 0.8
                            alignment_reason += f" | 极低位强信号降信心({position_pct:.1f}%)"
                        else:
                            final_confidence *= 0.6
                            alignment_reason += f" | 极低位做空过滤({position_pct:.1f}%)"
                    elif low_zone and osc_bias >= 30 and (fade_short or not aligned):
                        final_confidence *= 0.8
                        alignment_reason += f" | 低位超卖降信心({position_pct:.1f}%)"

        # 9. 生成决策原因
        reason = self._generate_reason(
            weighted_score, 
            aligned, 
            alignment_reason, 
            quant_analysis,
            prophet_score=scores.get('prophet', 0),
            regime=regime
        )
        
        # 10. 计算动态交易参数 (新增)
        trade_params = self._calculate_trade_params(regime, position, final_confidence, action)
        
        # 11. 构建结果
        result = VoteResult(
            action=action,
            confidence=final_confidence,
            weighted_score=weighted_score,
            vote_details=vote_details,
            multi_period_aligned=aligned,
            reason=reason,
            regime=regime,
            position=position,
            trade_params=trade_params
        )
        
        # 12. 记录历史
        self.history.append(result)
        
        return result

    def _get_volume_ratio(self, df: Optional[pd.DataFrame], window: int = 20) -> Optional[float]:
        """Return latest volume ratio (current / rolling mean)."""
        if df is None or df.empty or 'volume' not in df.columns:
            return None

        if 'volume_ratio' in df.columns:
            try:
                return float(df['volume_ratio'].iloc[-1])
            except Exception:
                pass

        if len(df) < window:
            return None

        series = df['volume'].iloc[-window:]
        avg = series.mean()
        if avg <= 0:
            return None

        return float(series.iloc[-1] / avg)

    async def vote(self, snapshot: Any, quant_analysis: Dict) -> VoteResult:
        """
        兼容性接口: 调用 make_decision
        """
        # 将 snapshot 转换为 market_data 格式供 make_decision 使用
        market_data = {
            'df_5m': snapshot.stable_5m if hasattr(snapshot, 'stable_5m') else None,
            'current_price': snapshot.live_5m.get('close', 0) if hasattr(snapshot, 'live_5m') else 0
        }
        return await self.make_decision(quant_analysis, market_data)

    def _calculate_comprehensive_confidence(self, 
                                          base_conf: float, 
                                          regime: Dict, 
                                          position: Dict, 
                                          aligned: bool) -> float:
        """计算综合信心度"""
        conf = base_conf
        
        # 加分项
        if aligned: conf += 15
        if regime['regime'] in ['trending_up', 'trending_down']: conf += 10
        if position['quality'] == 'excellent': conf += 15
        
        # 减分项
        if regime['regime'] == 'choppy': conf -= 25
        if position['location'] == 'middle': conf -= 30
        if regime['regime'] == 'volatile': conf -= 20
        
        return max(5.0, min(100.0, conf))
    
    def _calculate_trade_params(
        self, 
        regime: Optional[Dict], 
        position: Optional[Dict], 
        confidence: float,
        action: str
    ) -> Dict:
        """
        根据市场状态动态调整交易参数 (新增 2026-01-07)
        
        Args:
            regime: 市场状态信息
            position: 价格位置信息
            confidence: 决策置信度
            action: 交易动作
        
        Returns:
            动态交易参数字典
        """
        base_size = 100.0  # 基础仓位 USDT
        base_stop_loss = 1.0  # 基础止损百分比
        base_take_profit = 2.0  # 基础止盈百分比
        
        size_multiplier = 1.0
        sl_multiplier = 1.0
        tp_multiplier = 1.0
        
        # 根据市场状态调整
        if regime:
            regime_type = (regime.get('regime', '') or '').lower()
            
            if 'volatile' in regime_type:
                # 高波动市场：减少仓位，扩大止损
                size_multiplier *= 0.5
                sl_multiplier *= 1.5  # 止损放宽到1.5%
                tp_multiplier *= 1.5  # 止盈也放宽
            elif regime_type in ['trending_up', 'trending_down']:
                # 趋势市场：可以略增仓位，扩大止盈
                size_multiplier *= 1.2
                tp_multiplier *= 1.5  # 趋势中让利润奔跑
            elif regime_type == 'choppy':
                # 震荡市场：减少仓位，收紧参数
                size_multiplier *= 0.6
                sl_multiplier *= 0.8
                tp_multiplier *= 0.8
        
        # 根据价格位置调整
        if position:
            quality = position.get('quality', 'average')
            if quality == 'excellent':
                size_multiplier *= 1.3  # 优质位置可加仓
            elif quality == 'poor':
                size_multiplier *= 0.5  # 差位置减仓
        
        # 根据置信度调整
        if confidence > 70:
            size_multiplier *= min(confidence / 70, 1.5)  # 高置信度可加仓
        elif confidence < 50:
            size_multiplier *= 0.7  # 低置信度减仓
        
        # 如果是hold，仓位为0
        if action == 'hold':
            size_multiplier = 0
        
        return {
            'position_size': round(base_size * size_multiplier, 2),
            'stop_loss_pct': round(base_stop_loss * sl_multiplier, 2),
            'take_profit_pct': round(base_take_profit * tp_multiplier, 2),
            'leverage_suggested': 1 if size_multiplier < 0.8 else (2 if size_multiplier > 1.2 else 1),
            'reason': f"size_mult={size_multiplier:.2f}, sl_mult={sl_multiplier:.2f}, tp_mult={tp_multiplier:.2f}"
        }
    
    def _check_multi_period_alignment(
        self, 
        score_1h: float, 
        score_15m: float, 
        score_5m: float
    ) -> Tuple[bool, str]:
        """
        检测多周期趋势一致性 (优化版 2026-01-07)
        
        策略 (收紧条件，减少噪音交易):
        - 三个周期方向一致（同为正或同为负）-> 强对齐
        - 1h和15m一致（忽略5m噪音）-> 部分对齐
        - 其他情况 -> 不对齐（必须有1h方向确认）
        
        Returns:
            (是否对齐, 对齐原因)
        """
        # 提高阈值判断，减少噪音信号（回测放宽边界以避免无交易）
        signs = [
            1 if score_1h >= 18 else (-1 if score_1h <= -18 else 0),   # 1h 放宽至 >=18
            1 if score_15m >= 12 else (-1 if score_15m <= -12 else 0), # 15m 放宽至 >=12
            1 if score_5m >= 10 else (-1 if score_5m <= -10 else 0)    # 5m 保持
        ]
        
        # 三周期完全一致 - 最强信号
        if signs[0] == signs[1] == signs[2] and signs[0] != 0:
            return True, f"三周期强势{('多头' if signs[0] > 0 else '空头')}对齐"
        
        # 1h和15m一致（忽略5m噪音）- 可靠信号
        if signs[0] == signs[1] and signs[0] != 0:
            return True, f"中长周期{('多头' if signs[0] > 0 else '空头')}对齐(1h+15m)"
        
        # 移除：1h中性时的宽松条件
        # 原因：1h没有明确方向时不应轻易入场，减少噪音交易
        
        # 不对齐 - 需要等待更明确的信号
        return False, f"多周期分歧(1h:{signs[0]}, 15m:{signs[1]}, 5m:{signs[2]})，等待1h确认"
    
    def _score_to_action(
        self, 
        weighted_score: float, 
        aligned: bool,
        regime: Dict = None
    ) -> Tuple[str, float]:
        """
        将加权得分映射为交易动作
        
        策略 (优化后 2026-01-07):
        - 分离多空阈值，增加做空机会
        - 根据市场趋势动态调整阈值
        - 提高进场质量，减少噪音交易
        
        Returns:
            (action, confidence)
        """
        # 分离多空阈值 - 关键优化：启用双向交易
        long_threshold = 20   # 提高做多阈值
        short_threshold = 18  # 做空阈值略低，增加做空机会
        
        # 根据市场状态动态调整阈值
        if regime:
            regime_type = (regime.get('regime', '') or '').lower()
            if regime_type in ['trending_down']:
                # 下跌趋势：大幅降低做空阈值，提高做多阈值
                short_threshold = 12
                long_threshold = 25
            elif regime_type in ['trending_up']:
                # 上涨趋势：降低做多阈值，提高做空阈值
                long_threshold = 15
                short_threshold = 25
            elif regime_type in ['volatile_directionless', 'choppy']:
                # 震荡市：提高两边阈值，减少交易
                long_threshold = 25
                short_threshold = 25
            elif regime_type in ['volatile_trending']:
                # 波动趋势：中等阈值
                long_threshold = 18
                short_threshold = 18
        
        # 对齐时放宽阈值，提升中等信号的成交率
        if aligned:
            long_threshold = max(12, long_threshold - 2)
            short_threshold = max(12, short_threshold - 2)

        # 强信号阈值（需要多周期对齐）
        long_high_threshold = long_threshold + 15
        short_high_threshold = short_threshold + 15
        
        # 强信号：高阈值 + 多周期对齐
        if weighted_score > long_high_threshold and aligned:
            return 'long', 0.85
        if weighted_score < -short_high_threshold and aligned:
            return 'short', 0.85
        
        # 中等信号
        if weighted_score > long_threshold:
            confidence = 0.55 + (weighted_score - long_threshold) * 0.01
            return 'long', min(confidence, 0.75)
        if weighted_score < -short_threshold:
            confidence = 0.55 + (abs(weighted_score) - short_threshold) * 0.01
            return 'short', min(confidence, 0.75)
        
        # 弱信号或冲突 -> 观望
        return 'hold', abs(weighted_score) / 100
    
    def _generate_reason(
        self, 
        weighted_score: float,
        aligned: bool,
        alignment_reason: str,

        quant_analysis: Dict,
        prophet_score: float = 0.0,
        regime: Optional[Dict] = None
    ) -> str:
        """生成决策原因（可解释性）"""
        # 提取关键信息 (使用正确的key路径)
        trend_data = quant_analysis.get('trend', {})
        osc_data = quant_analysis.get('oscillator', {})
        sentiment_data = quant_analysis.get('sentiment', {})
        
        reasons = []
        
        # 1. 市场状态 (Regime)
        if regime:
            regime_name = regime.get('regime', 'unknown').upper()
            reasons.append(f"[{regime_name}]")
        
        # 2. 总体得分
        reasons.append(f"加权得分: {weighted_score:.1f}")
        
        # 3. 多周期对齐情况
        reasons.append(f"周期对齐: {alignment_reason}")
        
        # 4. 主要驱动因素（使用正确的granular scores）
        vote_details = {
            'trend_1h': trend_data.get('trend_1h_score', 0),
            'trend_15m': trend_data.get('trend_15m_score', 0),
            'oscillator_1h': osc_data.get('osc_1h_score', 0),
            'oscillator_15m': osc_data.get('osc_15m_score', 0),
            'sentiment': sentiment_data.get('total_sentiment_score', 0),
            'prophet': prophet_score
        }
        sorted_signals = sorted(
            vote_details.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:2]
        
        for sig_name, sig_score in sorted_signals:
            if abs(sig_score) > 20:
                reasons.append(f"{sig_name}: {sig_score:+.0f}")
        
        return " | ".join(reasons)
    
    def update_performance(self, signal_name: str, is_correct: bool):
        """
        更新信号历史表现（用于自适应权重调整）
        
        Args:
            signal_name: 信号名称（如'trend_5m'）
            is_correct: 该信号的预测是否准确
        """
        if signal_name in self.performance_tracker:
            self.performance_tracker[signal_name]['total'] += 1
            if is_correct:
                self.performance_tracker[signal_name]['correct'] += 1
    
    def adjust_weights_by_performance(self) -> SignalWeight:
        """
        根据历史表现自适应调整权重（高级功能）
        
        策略:
        - 计算各信号的胜率
        - 胜率高的信号增加权重，低的减少权重
        - 保证权重总和为1.0
        
        Returns:
            调整后的权重配置
        """
        # 计算各信号胜率
        win_rates = {}
        for sig_name, perf in self.performance_tracker.items():
            if perf['total'] > 0:
                win_rates[sig_name] = perf['correct'] / perf['total']
            else:
                win_rates[sig_name] = 0.5  # 默认50%
        
        # 归一化（总和=1.0）
        total_rate = sum(win_rates.values())
        if total_rate > 0:
            normalized_weights = {
                k: v / total_rate for k, v in win_rates.items()
            }
        else:
            return self.weights  # 无足够数据，保持原权重
        
        # 更新权重
        new_weights = SignalWeight(
            trend_5m=normalized_weights.get('trend_5m', self.weights.trend_5m),
            trend_15m=normalized_weights.get('trend_15m', self.weights.trend_15m),
            trend_1h=normalized_weights.get('trend_1h', self.weights.trend_1h),
            oscillator_5m=normalized_weights.get('oscillator_5m', self.weights.oscillator_5m),
            oscillator_15m=normalized_weights.get('oscillator_15m', self.weights.oscillator_15m),

            oscillator_1h=normalized_weights.get('oscillator_1h', self.weights.oscillator_1h),
            prophet=normalized_weights.get('prophet', self.weights.prophet),
        )
        
        return new_weights
    
    def to_llm_context(self, vote_result: VoteResult, quant_analysis: Dict) -> str:
        """
        将量化信号转换为LLM上下文（用于DeepSeek决策增强）
        
        Returns:
            格式化的文本上下文
        """
        context = f"""
### 量化信号汇总 (Decision Core Output)

**加权投票结果**:
- 综合得分: {vote_result.weighted_score:.1f} (-100~+100)
- 建议动作: {vote_result.action}
- 置信度: {vote_result.confidence:.1f}%
- 多周期对齐: {'✅ 是' if vote_result.multi_period_aligned else '❌ 否'}

**市场体制 (Regime Analysis)**:
- 状态: {vote_result.regime.get('regime', 'UNKNOWN').upper()}
- 信心度: {vote_result.regime.get('confidence', 0):.1f}%
- ADX: {vote_result.regime.get('adx', 0):.1f}
- 判定: {vote_result.regime.get('reason', 'N/A')}

**决策原因**: {vote_result.reason}

**各信号详情**:
"""
        # 添加各周期趋势分析
        for period in ['5m', '15m', '1h']:
            trend_key = f'trend_{period}'
            osc_key = f'oscillator_{period}'
            
            if trend_key in quant_analysis:
                trend = quant_analysis[trend_key]
                context += f"\n[{period}周期趋势] {trend.get('signal', 'N/A')} (得分:{trend.get('score', 0)})"
                context += f"\n  └ EMA状态: {trend.get('details', {}).get('ema_status', 'N/A')}"
            
            if osc_key in quant_analysis:
                osc = quant_analysis[osc_key]
                context += f"\n[{period}周期震荡] {osc.get('signal', 'N/A')} (得分:{osc.get('score', 0)})"
                rsi = osc.get('details', {}).get('rsi_value', 0)
                context += f"\n  └ RSI: {rsi:.1f}"
        
        context += f"\n\n**权重分配**: {json.dumps(vote_result.vote_details, indent=2)}"
        
        return context
    
    def get_statistics(self) -> Dict:
        """获取决策统计信息"""
        if not self.history:
            return {'total_decisions': 0}
        
        total = len(self.history)
        actions = [h.action for h in self.history]
        avg_confidence = sum(h.confidence for h in self.history) / total
        aligned_count = sum(1 for h in self.history if h.multi_period_aligned)
        
        return {
            'total_decisions': total,
            'action_distribution': {
                'long': actions.count('long'),
                'short': actions.count('short'),
                'hold': actions.count('hold'),
            },
            'avg_confidence': avg_confidence,
            'alignment_rate': aligned_count / total,
            'performance_tracker': self.performance_tracker,
        }


# ============================================
# 测试函数
# ============================================
async def test_decision_core():
    """测试决策中枢Agent"""
    print("\n" + "="*60)
    print("🧪 测试决策中枢Agent")
    print("="*60)
    
    # 模拟量化分析师的输出
    mock_quant_analysis = {
        'trend_5m': {
            'score': -15,
            'signal': 'weak_short',
            'details': {'ema_status': 'bearish_crossover'}
        },
        'trend_15m': {
            'score': 45,
            'signal': 'moderate_long',
            'details': {'ema_status': 'bullish'}
        },
        'trend_1h': {
            'score': 65,
            'signal': 'strong_long',
            'details': {'ema_status': 'strong_bullish'}
        },
        'oscillator_5m': {
            'score': -5,
            'signal': 'neutral',
            'details': {'rsi_value': 48.2}
        },
        'oscillator_15m': {
            'score': 20,
            'signal': 'moderate_long',
            'details': {'rsi_value': 62.5}
        },
        'oscillator_1h': {
            'score': 30,
            'signal': 'moderate_long',
            'details': {'rsi_value': 68.3}
        },
    }
    
    # 创建决策中枢
    decision_core = DecisionCoreAgent()
    
    # 执行决策
    print("\n1️⃣ 测试加权投票决策...")
    result = await decision_core.make_decision(mock_quant_analysis)
    
    print(f"  ✅ 决策动作: {result.action}")
    print(f"  ✅ 综合得分: {result.weighted_score:.2f}")
    print(f"  ✅ 置信度: {result.confidence:.1f}%")
    print(f"  ✅ 多周期对齐: {result.multi_period_aligned}")
    print(f"  ✅ 决策原因: {result.reason}")
    
    # 测试LLM上下文生成
    print("\n2️⃣ 测试LLM上下文生成...")
    llm_context = decision_core.to_llm_context(result, mock_quant_analysis)
    print(llm_context[:500] + "...")  # 只显示前500字符
    
    # 测试统计信息
    print("\n3️⃣ 测试统计信息...")
    # 再执行几次决策
    for _ in range(3):
        await decision_core.make_decision(mock_quant_analysis)
    
    stats = decision_core.get_statistics()
    print(f"  ✅ 总决策次数: {stats['total_decisions']}")
    print(f"  ✅ 平均置信度: {stats['avg_confidence']:.1f}%")
    print(f"  ✅ 对齐率: {stats['alignment_rate']:.2%}")
    
    print("\n✅ 决策中枢Agent测试通过!")
    return decision_core


if __name__ == '__main__':
    # 运行测试
    asyncio.run(test_decision_core())
