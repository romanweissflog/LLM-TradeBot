"""
量化策略师 (The Strategist) Agent - 重构版

职责：
按时间周期组织技术分析，而非按指标类型
- 6小时分析：完整技术指标集
- 2小时分析：完整技术指标集
- 半小时分析：完整技术指标集

优化点：
- 时间周期为中心的组织方式
- 便于LLM理解每个时间周期的完整技术面
- 扩展指标集：EMA, MA, BOLL, RSI, MACD, KDJ, ATR, OBV
"""

import pandas as pd
from typing import Dict
from dataclasses import asdict

from src.agents.data_sync_agent import MarketSnapshot
from src.utils.logger import log
from src.agents.regime_detector import RegimeDetector


class QuantAnalystAgent:
    """
    量化策略师 (The Strategist)
    
    提供情绪分析和OI燃料验证
    技术指标分析现在直接在main.py中使用真实1h/15m/5m数据
    """
    
    def __init__(self):
        """初始化量化策略师"""
        self.regime_detector = RegimeDetector()
        log.info("👨‍🔬 The Strategist (QuantAnalyst Agent) initialized - Full Analysis Mode")
        
    @staticmethod
    def calculate_ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, m1: int = 3, m2: int = 3):
        low_min = low.rolling(window=n).min()
        high_max = high.rolling(window=n).max()
        rsv = 100 * (close - low_min) / (high_max - low_min)
        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()
        
    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Calculate trend score (-100 to +100)"""
        if df is None or len(df) < 60:
            return {'score': 0, 'signal': 'neutral', 'details': {}}
            
        close = df['close']
        ema20 = self.calculate_ema(close, 20)
        ema60 = self.calculate_ema(close, 60)
        
        curr_close = close.iloc[-1]
        curr_ema20 = ema20.iloc[-1]
        curr_ema60 = ema60.iloc[-1]
        
        score = 0
        details = {'ema_status': 'neutral'}
        
        if curr_close > curr_ema20 > curr_ema60:
            score = 60
            details['ema_status'] = 'bullish_alignment'
        elif curr_close < curr_ema20 < curr_ema60:
            score = -60
            details['ema_status'] = 'bearish_alignment'
        elif curr_close > curr_ema20 and curr_ema20 < curr_ema60:
             score = 20 # Potential reversal up
             details['ema_status'] = 'potential_reversal_up'
        elif curr_close < curr_ema20 and curr_ema20 > curr_ema60:
             score = -20 # Potential reversal down
             details['ema_status'] = 'potential_reversal_down'
             
        return {'score': score, 'signal': 'long' if score > 0 else 'short', 'details': details}

    def analyze_oscillator(self, df: pd.DataFrame) -> Dict:
        """Calculate oscillator score (-100 to +100)"""
        if df is None or len(df) < 30:
            return {'score': 0, 'signal': 'neutral', 'details': {}}
            
        close = df['close']
        high = df['high']
        low = df['low']
        
        rsi = self.calculate_rsi(close, 14)
        curr_rsi = rsi.iloc[-1]
        
        k, d, j = self.calculate_kdj(high, low, close)
        curr_j = j.iloc[-1]
        
        score = 0
        details = {'rsi_value': round(curr_rsi, 1), 'kdj_j': round(curr_j, 1)}
        
        if curr_rsi < 30:
            score += 40 # Oversold -> Bullish
        elif curr_rsi > 70:
            score -= 40 # Overbought -> Bearish
            
        if curr_j < 20:
             score += 30
        elif curr_j > 80:
             score -= 30
             
        return {'score': score, 'signal': 'long' if score > 0 else 'short', 'details': details}

    async def analyze_all_timeframes(self, snapshot: MarketSnapshot) -> Dict:
        """
        执行完整技术分析
        """
        # 1. 情绪分析
        sentiment = self._analyze_sentiment(snapshot)
        
        # 2. 趋势分析
        t_5m = self.analyze_trend(snapshot.stable_5m)
        t_15m = self.analyze_trend(snapshot.stable_15m)
        t_1h = self.analyze_trend(snapshot.stable_1h)
        
        # 3. 震荡分析
        o_5m = self.analyze_oscillator(snapshot.stable_5m)
        o_15m = self.analyze_oscillator(snapshot.stable_15m)
        o_1h = self.analyze_oscillator(snapshot.stable_1h)
        
        # 4. Volatility Analysis (ATR)
        volatility = {'atr_1h': 0.0, 'atr_15m': 0.0, 'atr_5m': 0.0}
        for p, df in [('1h', snapshot.stable_1h), ('15m', snapshot.stable_15m), ('5m', snapshot.stable_5m)]:
             if df is not None and len(df) > 20:
                 atr = self.calculate_atr(df['high'], df['low'], df['close']).iloc[-1]
                 volatility[f'atr_{p}'] = round(atr, 4)
        
        # 5. 计算综合得分
        total_trend_score = (t_5m['score'] + t_15m['score'] + t_1h['score']) / 3
        total_osc_score = (o_5m['score'] + o_15m['score'] + o_1h['score']) / 3
        
        # 6. 市场体制检测 (Using 1h for backbone regime)
        regime = self.regime_detector.detect_regime(snapshot.stable_1h) if snapshot.stable_1h is not None else {}
        
        result = {
            'symbol': snapshot.symbol,  # 🔧 FIX: Include symbol for DecisionCoreAgent's OvertradingGuard
            'sentiment': sentiment,
            'volatility': volatility,
            'regime': regime,
            # 保留空的占位符以兼容
            'timeframe_6h': {}, 
            'timeframe_2h': {},
            'timeframe_30m': {},
            
            'trend': {
                'trend_5m_score': t_5m['score'],
                'trend_15m_score': t_15m['score'],
                'trend_1h_score': t_1h['score'],
                'total_trend_score': total_trend_score,
                'trend_5m': t_5m,
                'trend_15m': t_15m,
                'trend_1h': t_1h
            },
            'oscillator': {
                'osc_5m_score': o_5m['score'],
                'osc_15m_score': o_15m['score'],
                'osc_1h_score': o_1h['score'],
                'total_osc_score': total_osc_score,
                'oscillator_5m': o_5m,
                'oscillator_15m': o_15m,
                'oscillator_1h': o_1h
            },
            'overall_score': (total_trend_score + total_osc_score) / 2
        }
        
        return result
    
    def analyze(self, snapshot: MarketSnapshot) -> Dict:
        return {}

    def _analyze_sentiment(self, snapshot: MarketSnapshot) -> Dict:
        details = {}
        b_funding = getattr(snapshot, 'binance_funding', {})
        has_data = False
        score = 0
        
        if b_funding and 'funding_rate' in b_funding:
            has_data = True
            funding_rate = float(b_funding['funding_rate']) * 100
            details['funding_rate'] = funding_rate
            if funding_rate > 0.05: score -= 30
            elif funding_rate > 0.01: score -= 15
            elif funding_rate < -0.05: score += 30
            elif funding_rate < -0.01: score += 15
        
        vol_change_pct = 0.0
        fuel_signal = "neutral"
        df_1h = snapshot.stable_1h
        if df_1h is not None and len(df_1h) >= 24:
            has_data = True
            current_vol = df_1h['volume'].iloc[-1]
            avg_vol = df_1h['volume'].iloc[-25:-1].mean()
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                vol_change_pct = max(min((vol_ratio - 1) * 100, 200), -100)
            
            details['oi_change_24h_pct'] = vol_change_pct
            if vol_change_pct > 50:
                score += 20
                fuel_signal = "strong"
            elif vol_change_pct > 20:
                score += 10
                fuel_signal = "moderate"
            elif vol_change_pct < -50:
                score -= 10
                fuel_signal = "weak"
        
        oi_fuel = {
            'oi_change_24h': vol_change_pct,
            'fuel_signal': fuel_signal,
            'fuel_score': min(100, max(-100, int(vol_change_pct))),
            'whale_trap_risk': False,
            'fuel_strength': fuel_signal, 
            'is_proxy': True
        }
        
        return {
            'score': score if has_data else 0,
            'details': details,
            'has_data': has_data,
            'total_sentiment_score': score if has_data else 0,
            'oi_change_24h_pct': vol_change_pct,
            'oi_fuel': oi_fuel, 
        }
