"""
Backtest Agent Wrapper
======================

Adapts the production Multi-Agent system for Backtesting.

Key Components:
1. BacktestSignalCalculator: Re-implements technical analysis logic locally (since we don't have the external DataSyncAgent in backtest).
2. BacktestAgentRunner: Orchestrates the agents (Critic, etc.) using simulated data.

Author: AI Trader Team
Date: 2025-12-31
"""

import pandas as pd
import numpy as np
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.agents.decision_core_agent import DecisionCoreAgent
from src.utils.logger import log


class BacktestSignalCalculator:
    """
    Re-implements core signal logic for Backtesting.
    Generates numeric scores compatible with DecisionCoreAgent.
    """
    
    @staticmethod
    def calculate_ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, m1: int = 3, m2: int = 3):
        low_min = low.rolling(window=n).min()
        high_max = high.rolling(window=n).max()
        
        rsv = 100 * (close - low_min) / (high_max - low_min)
        
        # Use EWM for smooth K and D (classic KDJ uses SMA, but EWM is common in crypto)
        # Using com=m-1 which is equivalent to alpha=1/m
        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return k, d, j

    @staticmethod
    def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = (macd_line - signal_line) * 2
        return macd_line, signal_line, macd_hist

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
        
        # Basic EMA Alignment
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
        
        # RSI
        rsi = self.calculate_rsi(close, 14)
        curr_rsi = rsi.iloc[-1]
        
        # KDJ
        k, d, j = self.calculate_kdj(high, low, close)
        curr_j = j.iloc[-1]
        
        score = 0
        details = {'rsi_value': round(curr_rsi, 1), 'kdj_j': round(curr_j, 1)}
        
        # RSI Logic
        if curr_rsi < 30:
            score += 40 # Oversold -> Bullish
        elif curr_rsi > 70:
            score -= 40 # Overbought -> Bearish
            
        # KDJ Logic
        if curr_j < 20:
             score += 30
        elif curr_j > 80:
             score -= 30
             
        return {'score': score, 'signal': 'long' if score > 0 else 'short', 'details': details}

    def compute_all_signals(self, snapshot) -> Dict:
        """
        Compute full quant analysis structure compatible with DecisionCoreAgent
        """
        # Calculate Trend Scores
        t_5m = self.analyze_trend(snapshot.stable_5m)
        t_15m = self.analyze_trend(snapshot.stable_15m)
        t_1h = self.analyze_trend(snapshot.stable_1h)
        
        # Calculate Oscillator Scores
        o_5m = self.analyze_oscillator(snapshot.stable_5m)
        o_15m = self.analyze_oscillator(snapshot.stable_15m)
        o_1h = self.analyze_oscillator(snapshot.stable_1h)
        
        # Structure matches QuantAnalystAgent output
        return {
            'trend': {
                'trend_5m_score': t_5m['score'],
                'trend_15m_score': t_15m['score'],
                'trend_1h_score': t_1h['score'],
                'trend_5m': t_5m,
                'trend_15m': t_15m,
                'trend_1h': t_1h
            },
            'oscillator': {
                'osc_5m_score': o_5m['score'],
                'osc_15m_score': o_15m['score'],
                'osc_1h_score': o_1h['score'],
                'oscillator_5m': o_5m,
                'oscillator_15m': o_15m,
                'oscillator_1h': o_1h
            },
            'sentiment': {
                'total_sentiment_score': 0, # Placeholder, hard to simulate without global data
                'details': {'note': 'Sentiment neutral in backtest'}
            }
        }


class BacktestAgentRunner:
    """
    Wraps the Agent System for Backtesting.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        # We only need the DecisionCoreAgent (The Critic) to make final decisions
        # The other agents (Strategist, etc.) are simulated by BacktestSignalCalculator
        self.decision_core = DecisionCoreAgent()
        self.calculator = BacktestSignalCalculator()
        
        # Initialize LLM engine if enabled
        if self.config.get('use_llm', False):
            from src.strategy.llm_engine import StrategyEngine
            self.llm_engine = StrategyEngine()
            log.info("🤖 BacktestAgentRunner initialized with LLM Engine enabled")
        else:
            self.llm_engine = None
            log.info("🤖 BacktestAgentRunner initialized (LLM disabled)")

    async def step(self, snapshot) -> Dict:
        """
        Process one backtest step
        """
        try:
            # 1. Calculate Signals
            quant_analysis = self.calculator.compute_all_signals(snapshot)
            
            # 2. Make Decision via Critic
            # We mock the PredictResult as None (ML disabled for speed/simplicity in backtest)
            market_data = {
                 'df_5m': snapshot.stable_5m,
                 'current_price': snapshot.live_5m.get('close', 0)
            }
            
            vote_result = await self.decision_core.make_decision(
                quant_analysis=quant_analysis,
                predict_result=None,
                market_data=market_data
            )
            
            # 3. LLM Enhancement (if enabled)
            if self.llm_engine:
                # Throttle to avoid rate limits
                throttle_ms = self.config.get('llm_throttle_ms', 100)
                if throttle_ms > 0:
                    await asyncio.sleep(throttle_ms / 1000)
                
                # Call LLM with context
                llm_decision = await self._call_llm_with_context(
                    snapshot, quant_analysis, vote_result
                )
                
                # Merge LLM reasoning with quant signals
                final_decision = self._merge_decisions(vote_result, llm_decision)
            else:
                final_decision = vote_result
            
            # 4. Format result
            return {
                'action': final_decision.action,
                'confidence': final_decision.confidence,
                'reason': final_decision.reason,
                'vote_details': getattr(final_decision, 'vote_details', {}),
                'weighted_score': getattr(final_decision, 'weighted_score', 0),
                'llm_enhanced': self.llm_engine is not None
            }
            
        except Exception as e:
            log.error(f"Backtest agent step error: {e}")
            return {
                'action': 'hold',
                'confidence': 0,
                'reason': f"Error: {str(e)}"
            }
    
    async def _call_llm_with_context(self, snapshot, quant_analysis, vote_result):
        """Call LLM with full market context"""
        import json
        from datetime import datetime
        
        # Build context string
        context_text = self._build_llm_context(snapshot, quant_analysis, vote_result)
        
        # Build context data dict
        context_data = {
            'symbol': self.config.get('symbol', 'BTCUSDT'),
            'timestamp': datetime.now().isoformat(),
            'current_price': snapshot.live_5m.get('close', 0),
            'quant_analysis': quant_analysis,
            'vote_result': {
                'action': vote_result.action,
                'confidence': vote_result.confidence,
                'weighted_score': vote_result.weighted_score
            }
        }
        
        # Call LLM engine
        try:
            llm_result_dict = self.llm_engine.make_decision(
                market_context_text=context_text,
                market_context_data=context_data
            )
            
            # Convert dict to VoteResult-like object
            from src.agents.decision_core_agent import VoteResult
            llm_result = VoteResult(
                action=llm_result_dict.get('action', 'hold'),
                confidence=llm_result_dict.get('confidence', 0),
                reason=llm_result_dict.get('reasoning', 'LLM decision'),
                weighted_score=llm_result_dict.get('weighted_score', 0),
                vote_details={},
                multi_period_aligned=False
            )
            
            log.info(f"🤖 LLM Decision: {llm_result.action} (confidence: {llm_result.confidence}%)")
            return llm_result
        except Exception as e:
            log.warning(f"LLM call failed: {e}, falling back to quant decision")
            return vote_result
    
    def _build_llm_context(self, snapshot, quant_analysis, vote_result):
        """Build context string for LLM"""
        import json
        
        current_price = snapshot.live_5m.get('close', 0)
        
        return f"""Market Analysis Summary:
- Current Price: ${current_price:.2f}
- Quantitative Vote: {vote_result.action} (confidence: {vote_result.confidence:.1f}%)
- Weighted Score: {vote_result.weighted_score:.1f}
- Multi-Period Aligned: {vote_result.multi_period_aligned}

Technical Signals:
{json.dumps(quant_analysis, indent=2, default=str)}

Quantitative Reasoning: {vote_result.reason}

Please provide your trading decision based on this analysis."""
    
    def _merge_decisions(self, quant_vote, llm_decision):
        """Merge quantitative and LLM decisions"""
        from dataclasses import replace
        
        # Strategy: LLM can override if high confidence, otherwise enhance reasoning
        if llm_decision.confidence > 70:
            # LLM override with high confidence
            log.info(f"🎯 LLM override: {llm_decision.action} (confidence: {llm_decision.confidence}%)")
            return llm_decision
        else:
            # Enhance quant decision with LLM reasoning
            enhanced_reason = f"{quant_vote.reason} | LLM: {llm_decision.reason}"
            
            # Create a new VoteResult with enhanced reasoning
            try:
                enhanced_vote = replace(quant_vote, reason=enhanced_reason)
            except:
                # Fallback if replace doesn't work
                quant_vote.reason = enhanced_reason
                enhanced_vote = quant_vote
            
            log.info(f"✨ Enhanced decision: {enhanced_vote.action} with LLM reasoning")
            return enhanced_vote
