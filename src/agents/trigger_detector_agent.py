"""
5min Trigger Detector (Specification Module 4)

Purpose:
- Detect precise entry signals on 5min timeframe
- Pattern A: Bullish Engulfing (阳包阴)
- Pattern B: Volume Breakout (放量突破)

Specification:
- Data: Last 10 bars of 5min K-lines
- Volume MA3: Average of last 3 bars
- Engulfing: Previous bearish + Current bullish包住
- Breakout: Close > max(prev 3 highs) + Volume > 1.5 × MA3
"""

import pandas as pd
from typing import Dict
from src.utils.logger import log


class TriggerDetector:
    """
    5min Trigger Pattern Detector
    
    Implements specification Module 4: Trigger (扳机模块)
    """
    
    def __init__(self):
        """Initialize trigger detector"""
        pass
    
    def detect_engulfing(self, df_5m: pd.DataFrame, direction: str = 'long') -> Dict:
        """
        Detect Engulfing pattern (阳包阴 for long, 阴包阳 for short)
        
        Specification:
        - Previous candle: Bearish (Close < Open)
        - Current candle: Bullish (Close > Open) AND包住previous
        - Confirmation: Wait for candle close
        
        Args:
            df_5m: 5min K-line data
            direction: 'long' or 'short'
            
        Returns:
            {
                'detected': bool,
                'pattern': str,
                'prev_candle': dict,
                'curr_candle': dict
            }
        """
        if len(df_5m) < 2:
            return {'detected': False, 'pattern': None}
        
        prev = df_5m.iloc[-2]
        curr = df_5m.iloc[-1]
        
        if direction == 'long':
            # 前一根是阴线
            prev_bearish = prev['close'] < prev['open']
            # 当前是阳线
            curr_bullish = curr['close'] > curr['open']
            # 阳线包住阴线
            engulfing = curr['close'] > prev['open'] and curr['open'] < prev['close']
            
            detected = prev_bearish and curr_bullish and engulfing
            
        else:  # short
            # 前一根是阳线
            prev_bullish = prev['close'] > prev['open']
            # 当前是阴线
            curr_bearish = curr['close'] < curr['open']
            # 阴线包住阳线
            engulfing = curr['close'] < prev['open'] and curr['open'] > prev['close']
            
            detected = prev_bullish and curr_bearish and engulfing
        
        if detected:
            log.info(f"🎯 Engulfing pattern detected ({direction}): "
                    f"Prev [{prev['open']:.2f}->{prev['close']:.2f}], "
                    f"Curr [{curr['open']:.2f}->{curr['close']:.2f}]")
        
        return {
            'detected': detected,
            'pattern': 'engulfing',
            'prev_candle': {
                'open': prev['open'],
                'close': prev['close'],
                'high': prev['high'],
                'low': prev['low']
            },
            'curr_candle': {
                'open': curr['open'],
                'close': curr['close'],
                'high': curr['high'],
                'low': curr['low']
            }
        }
    
    def detect_breakout(self, df_5m: pd.DataFrame, direction: str = 'long') -> Dict:
        """
        Detect Volume Breakout (放量突破)
        
        Specification:
        - Price: Close > max(prev 3 highs) for long
        - Volume: Current > 1.5 × MA3
        - Timing: Can enter immediately, no need to wait for close
        
        Args:
            df_5m: 5min K-line data
            direction: 'long' or 'short'
            
        Returns:
            {
                'detected': bool,
                'pattern': str,
                'breakout_level': float,
                'volume_ratio': float
            }
        """
        if len(df_5m) < 4:
            return {'detected': False, 'pattern': None}
        
        curr = df_5m.iloc[-1]
        prev_3 = df_5m.iloc[-4:-1]
        
        # Calculate Volume MA3
        vol_ma3 = prev_3['volume'].mean()
        volume_ratio = curr['volume'] / vol_ma3 if vol_ma3 > 0 else 0
        
        if direction == 'long':
            # 突破前3根高点
            breakout_level = prev_3['high'].max()
            price_breakout = curr['close'] > breakout_level
        else:  # short
            # 跌破前3根低点
            breakout_level = prev_3['low'].min()
            price_breakout = curr['close'] < breakout_level
        
        # 成交量放大1.5倍
        volume_confirm = volume_ratio > 1.5
        
        detected = price_breakout and volume_confirm
        
        if detected:
            log.info(f"🚀 Breakout detected ({direction}): "
                    f"Price {curr['close']:.2f} {'>' if direction == 'long' else '<'} {breakout_level:.2f}, "
                    f"Volume ratio {volume_ratio:.2f}x")
        
        return {
            'detected': detected,
            'pattern': 'breakout',
            'breakout_level': breakout_level,
            'volume_ratio': volume_ratio,
            'current_price': curr['close'],
            'current_volume': curr['volume'],
            'vol_ma3': vol_ma3
        }
    
    def detect_trigger(self, df_5m: pd.DataFrame, direction: str = 'long') -> Dict:
        """
        Detect any trigger pattern (Engulfing OR Breakout)
        
        Args:
            df_5m: 5min K-line data
            direction: 'long' or 'short'
            
        Returns:
            {
                'triggered': bool,
                'pattern_type': 'engulfing' | 'breakout' | None,
                'details': dict,
                'rvol': float  # 🆕 Relative Volume
            }
        """
        # Check engulfing
        engulfing_result = self.detect_engulfing(df_5m, direction)
        
        # Check breakout
        breakout_result = self.detect_breakout(df_5m, direction)
        
        # 🆕 Calculate RVOL (Relative Volume vs 10-period average)
        rvol = self.calculate_rvol(df_5m)
        
        if engulfing_result['detected']:
            return {
                'triggered': True,
                'pattern_type': 'engulfing',
                'details': engulfing_result,
                'rvol': rvol
            }
        elif breakout_result['detected']:
            return {
                'triggered': True,
                'pattern_type': 'breakout',
                'details': breakout_result,
                'rvol': rvol
            }
        else:
            return {
                'triggered': False,
                'pattern_type': None,
                'details': {},
                'rvol': rvol
            }
    
    def calculate_rvol(self, df: pd.DataFrame, lookback: int = 10) -> float:
        """
        Calculate Relative Volume (RVOL)
        
        RVOL = Current Volume / Average Volume (last N bars)
        
        Interpretation:
        - RVOL > 1.5: High relative volume, significant interest
        - RVOL > 2.0: Very high volume, potential institutional activity
        - RVOL < 0.5: Low volume, weak conviction
        
        Args:
            df: K-line data with 'volume' column
            lookback: Number of bars for average calculation
            
        Returns:
            RVOL ratio (float)
        """
        if len(df) < lookback + 1 or 'volume' not in df.columns:
            return 1.0
        
        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].iloc[-lookback-1:-1].mean()
        
        if avg_vol > 0:
            return current_vol / avg_vol
        return 1.0
