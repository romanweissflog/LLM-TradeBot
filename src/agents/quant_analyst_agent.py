"""
量化分析师 Agent (Quant Analyst Agent)

职责：
1. 趋势分析员：基于EMA/MACD计算趋势得分
2. 震荡分析员：基于RSI/BB计算反转得分
3. 实时价格修正：利用live_view更新指标

优化点：
- 得分制（-100~+100）替代布尔值
- 实时RSI计算（包含live K线）
- 多指标加权
"""

import pandas as pd
import numpy as np
from typing import Dict
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator

from src.agents.data_sync_agent import MarketSnapshot
from src.utils.logger import log


class TrendSubAgent:
    """
    趋势分析员（子Agent）
    
    职责：判断市场趋势方向和强度
    输出：trend_score (-100 到 +100)
    """
    
    def analyze(self, snapshot: MarketSnapshot) -> Dict:
        """
        计算趋势得分
        
        得分逻辑：
        - 1h EMA金叉 → +40分 (主趋势)
        - 15m MACD扩大 → +30分 (中期确认)
        - 5m 价格突破 → +30分 (短期动量)
        - live_view修正 → ±20分 (实时修正)
        
        Args:
            snapshot: 市场快照 (stable_xx DataFrames intended to be populated by MarketDataProcessor)
            
        Returns:
            分析结果字典
        """
        score = 0
        details = {}
        
        # 1. 1h 主趋势判断 (权重40%)
        stable_1h = snapshot.stable_1h
        if not stable_1h.empty and len(stable_1h) > 50:
            # 优先使用预计算指标
            if 'ema_12' in stable_1h.columns and 'ema_26' in stable_1h.columns:
                last_ema_12 = stable_1h['ema_12'].iloc[-1]
                last_ema_26 = stable_1h['ema_26'].iloc[-1]
            else:
                # 兼容模式：现场计算
                ema_12 = EMAIndicator(close=stable_1h['close'], window=12).ema_indicator()
                ema_26 = EMAIndicator(close=stable_1h['close'], window=26).ema_indicator()
                last_ema_12 = ema_12.iloc[-1]
                last_ema_26 = ema_26.iloc[-1]
            
            if last_ema_12 > last_ema_26:
                trend_1h_score = 40
                trend_1h_status = "上涨"
            else:
                trend_1h_score = -40
                trend_1h_status = "下跌"
            
            score += trend_1h_score
            details['1h_trend'] = trend_1h_status
            details['1h_ema12'] = float(last_ema_12)
            details['1h_ema26'] = float(last_ema_26)
        
        # 2. 实时修正 (权重±20%) - 核心创新
        live_1h = snapshot.live_1h
        if live_1h:
            # 计算当前K线的涨跌幅
            open_price = float(live_1h.get('open', 0))
            close_price = float(live_1h.get('close', 0))
            
            if open_price > 0:
                candle_change = (close_price - open_price) / open_price
                
                # 如果当前K线大跌1%，即使stable是上涨的，也要降低得分
                if candle_change < -0.01:
                    live_correction = -20
                    details['live_correction'] = "大跌1%，趋势可能反转"
                elif candle_change > 0.01:
                    live_correction = 20
                    details['live_correction'] = "大涨1%，趋势正在加速"
                else:
                    live_correction = 0
                    details['live_correction'] = "正常波动"
                
                score += live_correction
                details['live_candle_change'] = f"{candle_change*100:.2f}%"
        
        # 3. 15m 中期确认 (权重30%)
        stable_15m = snapshot.stable_15m
        if not stable_15m.empty and len(stable_15m) > 30:
            # 优先使用预计算指标
            if 'macd_diff' in stable_15m.columns:
                current_macd = stable_15m['macd_diff'].iloc[-1]
                prev_macd = stable_15m['macd_diff'].iloc[-2]
            else:
                macd_ind = MACD(close=stable_15m['close'])
                macd_diff = macd_ind.macd_diff()
                current_macd = macd_diff.iloc[-1]
                prev_macd = macd_diff.iloc[-2]
            
            # 检查MACD柱状图是否扩大
            if current_macd > prev_macd > 0:
                trend_15m_score = 30  # MACD金叉且扩大
                trend_15m_status = "上涨加速"
            elif current_macd < prev_macd < 0:
                trend_15m_score = -30  # MACD死叉且扩大
                trend_15m_status = "下跌加速"
            else:
                trend_15m_score = 0
                trend_15m_status = "震荡"
            
            score += trend_15m_score
            details['15m_trend'] = trend_15m_status
            details['15m_macd_diff'] = float(current_macd)
        
        # 限制得分范围
        score = max(-100, min(100, score))
        
        return {
            'score': score,
            'details': details,
            'confidence': abs(score)  # 得分越极端，置信度越高
        }


class OscillatorSubAgent:
    """
    震荡分析员（子Agent）
    
    职责：判断超买超卖和反转信号
    输出：reversion_score (-100 到 +100)
    """
    
    def analyze(self, snapshot: MarketSnapshot) -> Dict:
        """
        计算反转得分
        
        得分逻辑：
        - 1h RSI > 75 → -80 (超买严重，建议做空)
        - 5m RSI < 25 → +80 (超卖严重，建议做多)
        - live_view实时RSI → ±20分 (实时修正)
        
        Args:
            snapshot: 市场快照
            
        Returns:
            分析结果字典
        """
        score = 0
        details = {}
        
        # 1. 计算实时RSI (关键优化)
        stable_5m = snapshot.stable_5m
        live_5m = snapshot.live_5m
        
        if not stable_5m.empty and live_5m:
            # 将live_5m添加到stable_5m计算RSI
            df_with_live = stable_5m.copy()
            
            # 构造live K线的DataFrame行
            live_row = pd.DataFrame([{
                'open': float(live_5m.get('open', 0)),
                'high': float(live_5m.get('high', 0)),
                'low': float(live_5m.get('low', 0)),
                'close': float(live_5m.get('close', 0)),
                'volume': float(live_5m.get('volume', 0))
            }])
            
            # 添加到DataFrame
            df_with_live = pd.concat([df_with_live, live_row], ignore_index=True)
            
            # 计算RSI
            rsi_5m = RSIIndicator(close=df_with_live['close'], window=14).rsi()
            live_rsi = rsi_5m.iloc[-1] if len(rsi_5m) > 0 else 50
            
            # 基于RSI打分
            if live_rsi > 75:
                rsi_score = -80  # 强烈建议卖出/做空
                rsi_status = "超买严重"
            elif live_rsi < 25:
                rsi_score = +80  # 强烈建议买入/做多
                rsi_status = "超卖严重"
            elif live_rsi > 65:
                rsi_score = -40  # 轻度超买
                rsi_status = "轻度超买"
            elif live_rsi < 35:
                rsi_score = +40  # 轻度超卖
                rsi_status = "轻度超卖"
            else:
                rsi_score = 0
                rsi_status = "中性"
            
            score += rsi_score
            details['5m_rsi'] = float(live_rsi)
            details['5m_status'] = rsi_status
        
        # 2. 1h RSI确认
        stable_1h = snapshot.stable_1h
        if not stable_1h.empty:
            if 'rsi' in stable_1h.columns:
                last_rsi_1h = stable_1h['rsi'].iloc[-1]
            else:
                rsi_1h = RSIIndicator(close=stable_1h['close'], window=14).rsi()
                last_rsi_1h = rsi_1h.iloc[-1] if len(rsi_1h) > 0 else 50
            
            # 1h超买超卖的权重更高
            if last_rsi_1h > 80:
                score -= 20  # 额外扣分
                details['1h_warning'] = "1h级别超买"
            elif last_rsi_1h < 20:
                score += 20  # 额外加分
                details['1h_warning'] = "1h级别超卖"
            
            details['1h_rsi'] = float(last_rsi_1h)
        
        # 限制得分范围
        score = max(-100, min(100, score))
        
        return {
            'score': score,
            'details': details,
            'confidence': abs(score)
        }


class QuantAnalystAgent:
    """
    量化分析师（协调者）
    
    职责：协调趋势分析员和震荡分析员
    输出：综合分析报告
    """
    
    def __init__(self):
        self.trend_agent = TrendSubAgent()
        self.osc_agent = OscillatorSubAgent()
        log.info("👨‍🔬 量化分析师初始化完成")
    
    async def analyze_all_timeframes(self, snapshot: MarketSnapshot) -> Dict:
        """
        分析所有周期（异步版本，适配DecisionCoreAgent）
        
        Args:
            snapshot: 市场快照
            
        Returns:
            {
                'trend_5m': {...},
                'trend_15m': {...},
                'trend_1h': {...},
                'oscillator_5m': {...},
                'oscillator_15m': {...},
                'oscillator_1h': {...},
                'comprehensive': {...}
            }
        """
        # 调用原有的analyze方法
        analysis = self.analyze(snapshot)
        
        # 转换为DecisionCoreAgent期望的格式
        result = {
            # 趋势信号（从1h趋势得分推断）
            'trend_5m': {
                'score': analysis['trend_score'] * 0.3,  # 权重调整
                'signal': self._score_to_signal(analysis['trend_score'] * 0.3),
                'details': analysis['trend_details']
            },
            'trend_15m': {
                'score': analysis['trend_score'] * 0.6,
                'signal': self._score_to_signal(analysis['trend_score'] * 0.6),
                'details': analysis['trend_details']
            },
            'trend_1h': {
                'score': analysis['trend_score'],
                'signal': self._score_to_signal(analysis['trend_score']),
                'details': analysis['trend_details']
            },
            
            # 震荡信号
            'oscillator_5m': {
                'score': analysis['reversion_score'] * 0.3,
                'signal': self._score_to_signal(analysis['reversion_score'] * 0.3),
                'details': analysis['reversion_details']
            },
            'oscillator_15m': {
                'score': analysis['reversion_score'] * 0.6,
                'signal': self._score_to_signal(analysis['reversion_score'] * 0.6),
                'details': analysis['reversion_details']
            },
            'oscillator_1h': {
                'score': analysis['reversion_score'],
                'signal': self._score_to_signal(analysis['reversion_score']),
                'details': analysis['reversion_details']
            },
            
            # 综合信号
            'comprehensive': {
                'score': (analysis['trend_score'] + analysis['reversion_score']) / 2,
                'signal': self._score_to_signal((analysis['trend_score'] + analysis['reversion_score']) / 2),
                'details': {
                    'volatility': analysis['volatility'],
                    'trend_strength': 'strong' if abs(analysis['trend_score']) > 50 else 'moderate' if abs(analysis['trend_score']) > 20 else 'weak',
                    'alignment_ok': analysis['alignment_ok']
                }
            }
        }
        
        return result
    
    def _score_to_signal(self, score: float) -> str:
        """将得分转换为信号标签"""
        if score > 50:
            return 'strong_long'
        elif score > 20:
            return 'moderate_long'
        elif score > 0:
            return 'weak_long'
        elif score > -20:
            return 'weak_short'
        elif score > -50:
            return 'moderate_short'
        else:
            return 'strong_short'
    
    def analyze(self, snapshot: MarketSnapshot) -> Dict:
        """
        并行分析
        
        Args:
            snapshot: 市场快照
            
        Returns:
            综合分析结果
        """
        log.info("📊 开始量化分析...")
        
        # 1. 趋势分析
        trend_result = self.trend_agent.analyze(snapshot)
        log.info(f"  ├─ 趋势得分: {trend_result['score']}")
        
        # 2. 震荡分析
        osc_result = self.osc_agent.analyze(snapshot)
        log.info(f"  └─ 反转得分: {osc_result['score']}")
        
        # 3. 计算波动率（用于动态权重）
        volatility = self._calculate_volatility(snapshot)
        
        # 4. 综合报告
        analysis = {
            'trend_score': trend_result['score'],
            'trend_details': trend_result['details'],
            'trend_confidence': trend_result['confidence'],
            
            'reversion_score': osc_result['score'],
            'reversion_details': osc_result['details'],
            'reversion_confidence': osc_result['confidence'],
            
            'volatility': volatility,
            'timestamp': snapshot.timestamp.isoformat(),
            'alignment_ok': snapshot.alignment_ok
        }
        
        log.info(f"✅ 量化分析完成，波动率: {volatility:.2f}")
        
        return analysis
    
    def _calculate_volatility(self, snapshot: MarketSnapshot) -> float:
        """
        计算波动率
        
        使用ATR/价格作为波动率指标
        
        Args:
            snapshot: 市场快照
            
        Returns:
            波动率 (0-1)
        """
        stable_5m = snapshot.stable_5m
        
        if stable_5m.empty or len(stable_5m) < 20:
            return 0.5  # 默认中等波动
        
        # 计算True Range
        df = stable_5m.copy()
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # ATR (14周期)
        atr = df['true_range'].rolling(14).mean().iloc[-1]
        
        # 归一化 (ATR / 价格)
        current_price = df['close'].iloc[-1]
        volatility = atr / current_price if current_price > 0 else 0.5
        
        # 限制在0-1范围
        return max(0, min(1, volatility))


# 测试函数
def test_quant_analyst_agent():
    """测试量化分析师"""
    from src.agents.data_sync_agent import DataSyncAgent
    import asyncio
    
    async def run_test():
        print("\n" + "="*80)
        print("测试：量化分析师 (Quant Analyst Agent)")
        print("="*80)
        
        # 获取数据
        data_agent = DataSyncAgent()
        snapshot = await data_agent.fetch_all_timeframes("BTCUSDT")
        
        # 分析
        quant_agent = QuantAnalystAgent()
        analysis = quant_agent.analyze(snapshot)
        
        # 输出结果
        print("\n[分析结果]")
        print(f"  趋势得分: {analysis['trend_score']}")
        print(f"  趋势详情: {analysis['trend_details']}")
        print(f"\n  反转得分: {analysis['reversion_score']}")
        print(f"  反转详情: {analysis['reversion_details']}")
        print(f"\n  波动率: {analysis['volatility']:.4f}")
        
        print("\n" + "="*80)
        print("✅ 测试完成")
        print("="*80 + "\n")
    
    asyncio.run(run_test())


if __name__ == "__main__":
    test_quant_analyst_agent()
