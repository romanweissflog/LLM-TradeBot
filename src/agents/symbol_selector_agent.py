"""
🔝 Symbol Selector Agent - Automated Top 2 Symbol Selection (AUTO2)
====================================================================

Responsibilities:
1. Get AI500 Top 5 symbols by volume
2. Backtest each symbol (24h lookback)
3. Rank by composite profitability score
4. Auto-select top 2 performers
5. 12-hour refresh cycle
6. Startup execution (mandatory)

Author: AI Trader Team
Date: 2026-01-07
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import threading
import time

from src.utils.logger import log
from src.backtest.engine import BacktestEngine, BacktestConfig


class SymbolSelectorAgent:
    """
    Automated symbol selection based on backtest performance (AUTO2)
    
    Workflow:
    1. Get AI500 Top 5 symbols by 24h volume
    2. Run backtests on each symbol
    3. Rank by composite score
    4. Select top 2 performers
    5. Cache results for 12 hours
    6. Auto-refresh every 12 hours
    """
    
    # AI500 Candidate Pool (30+ AI/Data/Compute coins)
    AI500_CANDIDATES = [
        "FETUSDT", "RENDERUSDT", "TAOUSDT", "NEARUSDT", "GRTUSDT", 
        "WLDUSDT", "ARKMUSDT", "LPTUSDT", "THETAUSDT", "ROSEUSDT",
        "PHBUSDT", "CTXCUSDT", "NMRUSDT", "RLCUSDT", "GLMUSDT",
        "IQUSDT", "MDTUSDT", "AIUSDT", "NFPUSDT", "XAIUSDT",
        "JASMYUSDT", "ICPUSDT", "FILUSDT", "VETUSDT", "LINKUSDT",
        "ACTUSDT", "GOATUSDT", "TURBOUSDT", "PNUTUSDT"
    ]
    
    FALLBACK_SYMBOLS = ["FETUSDT", "RENDERUSDT"]  # Top 2 fallback
    
    def __init__(
        self,
        candidate_symbols: Optional[List[str]] = None,
        cache_dir: str = "config",
        refresh_interval_hours: int = 12,
        lookback_hours: int = 24
    ):
        """
        Initialize Symbol Selector Agent
        
        Args:
            candidate_symbols: List of symbols to evaluate (default: 20 symbols)
            cache_dir: Directory for cache storage
            refresh_interval_hours: Auto-refresh interval (default: 12h)
            lookback_hours: Backtest lookback period (default: 24h)
        """
        self.ai500_candidates = self.AI500_CANDIDATES
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "auto_top2_cache.json"
        
        self.refresh_interval = refresh_interval_hours
        self.lookback_hours = lookback_hours
        
        # Background refresh thread
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop_refresh = threading.Event()
        
        log.info(f"🔝 SymbolSelectorAgent (AUTO2) initialized: AI500 Top5 -> Top2, {refresh_interval_hours}h refresh")
    
    async def select_top2(self, force_refresh: bool = False) -> List[str]:
        """
        Select top 2 symbols based on backtest performance from AI500 Top 5
        
        Args:
            force_refresh: Force re-run backtests even if cache valid
            
        Returns:
            List of 2 symbol names
        """
        # Check cache validity
        if not force_refresh and self._is_cache_valid():
            cached = self._load_cache()
            symbols = [item['symbol'] for item in cached['top2']]
            if symbols:
                log.info(f"🔝 Using cached AUTO2: {symbols} (age: {self._get_cache_age():.1f}h)")
                return symbols
            else:
                log.warning("⚠️ Cache has empty top2, forcing refresh...")
        
        # Step 1: Get AI500 Top 5 by volume
        log.info("🔄 Getting AI500 Top 5 by 24h volume...")
        ai500_top5 = await self._get_ai500_top5()
        log.info(f"📊 AI500 Top 5: {ai500_top5}")
        
        # Step 2: Backtest each symbol
        log.info(f"🔄 Running AUTO2 backtests on {len(ai500_top5)} symbols...")
        start_time = time.time()
        
        try:
            results = await self._run_backtests(ai500_top5)
            ranked = self._rank_symbols(results)
            
            # Get top 2
            top2_data = ranked[:2]
            top2_symbols = [item['symbol'] for item in top2_data]
            
            # Save to cache
            self._save_cache(top2_data, results)
            
            elapsed = time.time() - start_time
            log.info(f"✅ AUTO2 selected in {elapsed:.1f}s: {top2_symbols}")
            
            return top2_symbols
            
        except Exception as e:
            log.error(f"❌ AUTO2 selection failed: {e}", exc_info=True)
            log.warning(f"⚠️ Falling back to default symbols: {self.FALLBACK_SYMBOLS}")
            return self.FALLBACK_SYMBOLS
    
    async def _get_ai500_top5(self) -> List[str]:
        """Get AI500 Top 5 symbols by 24h volume"""
        try:
            from src.api.binance_client import BinanceClient
            
            client = BinanceClient()
            tickers = client.get_all_tickers()
            
            # Filter AI500 candidates and sort by volume
            ai_stats = []
            for t in tickers:
                if t['symbol'] in self.ai500_candidates:
                    try:
                        quote_vol = float(t.get('quoteVolume', 0))
                        ai_stats.append((t['symbol'], quote_vol))
                    except (ValueError, TypeError):
                        pass
            
            # Sort by volume descending
            ai_stats.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 5
            return [x[0] for x in ai_stats[:5]]
            
        except Exception as e:
            log.error(f"Failed to get AI500 Top5: {e}")
            # Fallback to first 5 candidates
            return self.ai500_candidates[:5]
    
    async def _run_backtests(self, symbols: List[str]) -> List[Dict]:
        """Run backtests on provided symbol list"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=self.lookback_hours)
        
        valid_results = []
        total = len(symbols)
        
        for i, symbol in enumerate(symbols):
            log.info(f"🔄 [{i+1}/{total}] Backtesting {symbol}...")
            try:
                result = await self._backtest_symbol(symbol, start_time, end_time)
                if result:
                    valid_results.append(result)
                    log.info(f"   ✅ {symbol}: Return {result['total_return']:+.2f}%, Trades {result['trades']}")
            except Exception as e:
                log.warning(f"   ⚠️ {symbol} failed: {e}")
        
        return valid_results
    
    async def _backtest_symbol(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[Dict]:
        """Run backtest for a single symbol using thread executor"""
        try:
            config = BacktestConfig(
                symbol=symbol,
                start_date=start_time.strftime('%Y-%m-%d %H:%M'),
                end_date=end_time.strftime('%Y-%m-%d %H:%M'),
                initial_capital=10000.0,
                strategy_mode="technical",  # Use simple mode for speed
                use_llm=False,
                step=12  # 1-hour intervals (faster than 5m)
            )
            
            # Run sync backtest in thread executor to avoid blocking
            loop = asyncio.get_event_loop()
            engine = BacktestEngine(config)
            result = await loop.run_in_executor(None, lambda: asyncio.run(engine.run(progress_callback=None)))
            
            # BacktestResult has .metrics attribute with MetricsResult
            metrics = result.metrics
            
            return {
                "symbol": symbol,
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "win_rate": metrics.win_rate,
                "max_drawdown": metrics.max_drawdown_pct,
                "trades": metrics.total_trades,
                "profit_factor": metrics.profit_factor
            }
            
        except Exception as e:
            log.error(f"Backtest error for {symbol}: {e}")
            return None
    
    def _rank_symbols(self, results: List[Dict]) -> List[Dict]:
        """
        Rank symbols by composite score
        
        Scoring Formula:
        - Total Return: 30%
        - Sharpe Ratio: 20%
        - Win Rate: 25%
        - Max Drawdown: 15% (inverted penalty)
        - Trade Count: 10% (prefer 3-5 trades)
        """
        scored = []
        
        for result in results:
            # Extract metrics
            ret = result["total_return"]
            sharpe = max(result["sharpe_ratio"], 0)  # No negative Sharpe
            win_rate = result["win_rate"]
            dd = result["max_drawdown"]
            trades = result["trades"]
            
            # Composite score (0-100)
            score = (
                ret * 30 +                           # Return weight
                sharpe * 20 +                        # Sharpe weight
                win_rate * 0.25 +                    # Win rate weight
                max(0, 100 + dd) * 0.15 +           # Drawdown penalty
                min(trades / 5 * 10, 10)            # Trade frequency
            )
            
            result["composite_score"] = round(score, 2)
            scored.append(result)
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        
        return scored
    
    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is still valid"""
        if not self.cache_file.exists():
            return False
        
        try:
            cache = self._load_cache()
            valid_until = datetime.fromisoformat(cache["valid_until"])
            return datetime.now() < valid_until
        except Exception:
            return False
    
    def _get_cache_age(self) -> float:
        """Get cache age in hours"""
        try:
            cache = self._load_cache()
            timestamp = datetime.fromisoformat(cache["timestamp"])
            return (datetime.now() - timestamp).total_seconds() / 3600
        except Exception:
            return 999
    
    def _load_cache(self) -> Dict:
        """Load cache from file"""
        with open(self.cache_file, 'r') as f:
            return json.load(f)
    
    def _save_cache(self, top2: List[Dict], all_results: List[Dict]):
        """Save results to cache"""
        now = datetime.now()
        cache_data = {
            "timestamp": now.isoformat(),
            "valid_until": (now + timedelta(hours=self.refresh_interval)).isoformat(),
            "lookback_hours": self.lookback_hours,
            "top2": top2,
            "all_results": all_results
        }
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        log.info(f"💾 AUTO2 cache saved: valid until {cache_data['valid_until']}")
    
    def start_auto_refresh(self):
        """Start background thread for auto-refresh every 12 hours"""
        if self._refresh_thread and self._refresh_thread.is_alive():
            log.warning("Auto-refresh thread already running")
            return
        
        def refresh_loop():
            """Background refresh loop"""
            while not self._stop_refresh.is_set():
                # Wait for refresh interval
                if self._stop_refresh.wait(timeout=self.refresh_interval * 3600):
                    break  # Stop signal received
                
                # Run refresh
                log.info(f"🔄 AUTO2 auto-refresh triggered ({self.refresh_interval}h interval)")
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.select_top2(force_refresh=True))
                    loop.close()
                except Exception as e:
                    log.error(f"❌ Auto-refresh failed: {e}", exc_info=True)
        
        self._refresh_thread = threading.Thread(target=refresh_loop, daemon=True, name="AUTO2-Refresh")
        self._refresh_thread.start()
        log.info(f"🔄 AUTO2 auto-refresh started ({self.refresh_interval}h interval)")
    
    def stop_auto_refresh(self):
        """Stop background refresh thread"""
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._stop_refresh.set()
            self._refresh_thread.join(timeout=5)
            log.info("🛑 AUTO2 auto-refresh stopped")


# Global instance
_selector_instance: Optional[SymbolSelectorAgent] = None

def get_selector() -> SymbolSelectorAgent:
    """Get global SymbolSelectorAgent instance (singleton)"""
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = SymbolSelectorAgent()
    return _selector_instance
