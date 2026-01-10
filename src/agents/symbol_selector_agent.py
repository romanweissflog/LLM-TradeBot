"""
🔝 Symbol Selector Agent - Automated Top 3 Symbol Selection (AUTO3)
====================================================================

Responsibilities:
1. Get AI500 Top 10 + Major coins by volume
2. Stage 1: Coarse filter (1h backtest) → Top 5
3. Stage 2: Fine filter (15m backtest) → Top 2
4. 6-hour refresh cycle
5. Startup execution (mandatory)

Author: AI Trader Team
Date: 2026-01-07
Updated: 2026-01-10 (Two-stage selection)
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
    Automated symbol selection based on backtest performance (AUTO3)
    
    Two-Stage Workflow:
    1. Get AI500 Top 10 + Major coins by 24h volume (~16 symbols)
    2. Stage 1: Coarse filter (1h backtest, step=12) → Top 5
    3. Stage 2: Fine filter (15m backtest, step=3) → Top 2
    4. Cache results for 6 hours
    5. Auto-refresh every 6 hours
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
    
    # Major coins to include alongside AI500
    MAJOR_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
    
    FALLBACK_SYMBOLS = ["BTCUSDT", "ETHUSDT"]  # Top 2 fallback
    
    def __init__(
        self,
        candidate_symbols: Optional[List[str]] = None,
        cache_dir: str = "config",
        refresh_interval_hours: int = 6,
        lookback_hours: int = 24
    ):
        """
        Initialize Symbol Selector Agent
        
        Args:
            candidate_symbols: List of symbols to evaluate (default: 20 symbols)
            cache_dir: Directory for cache storage
            refresh_interval_hours: Auto-refresh interval (default: 6h)
            lookback_hours: Backtest lookback period (default: 24h)
        """
        self.ai500_candidates = self.AI500_CANDIDATES
        self.major_coins = self.MAJOR_COINS
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "auto_top2_cache.json"
        
        self.refresh_interval = refresh_interval_hours
        self.lookback_hours = lookback_hours
        
        # Background refresh thread
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop_refresh = threading.Event()
        
        log.info(f"🔝 SymbolSelectorAgent (AUTO3) initialized: Two-stage selection, {refresh_interval_hours}h refresh")
    
    async def select_top2(self, force_refresh: bool = False) -> List[str]:
        """
        Select top 2 symbols using two-stage filtering
        
        Stage 1: Coarse filter (1h backtest) on ~16 symbols → Top 5
        Stage 2: Fine filter (15m backtest) on Top 5 → Top 2
        
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
                log.info(f"🔝 Using cached AUTO3: {symbols} (age: {self._get_cache_age():.1f}h)")
                return symbols
            else:
                log.warning("⚠️ Cache has empty top2, forcing refresh...")
        
        start_time = time.time()
        
        try:
            # ============================================================
            # STAGE 1: Coarse Filter (1h backtest) → Top 5
            # ============================================================
            log.info("=" * 60)
            log.info("🔄 STAGE 1: Coarse Filter (1h backtest)")
            log.info("=" * 60)
            
            # Get AI500 Top 10 + Major coins
            candidates = await self._get_expanded_candidates()
            log.info(f"📊 Candidates ({len(candidates)}): {candidates}")
            
            # Run 1h backtests (step=12, faster)
            stage1_results = await self._run_backtests_stage(
                symbols=candidates,
                step=12,  # 1-hour intervals
                stage_name="Stage 1"
            )
            
            # Rank and get Top 5
            ranked_stage1 = self._rank_symbols(stage1_results)
            top5_symbols = [item['symbol'] for item in ranked_stage1[:5]]
            
            log.info(f"✅ Stage 1 complete: Top 5 = {top5_symbols}")
            
            # ============================================================
            # STAGE 2: Fine Filter (15m backtest) → Top 2
            # ============================================================
            log.info("=" * 60)
            log.info("🔄 STAGE 2: Fine Filter (15m backtest)")
            log.info("=" * 60)
            
            # Run 15m backtests on Top 5 (step=3, more precise)
            stage2_results = await self._run_backtests_stage(
                symbols=top5_symbols,
                step=3,  # 15-minute intervals
                stage_name="Stage 2"
            )
            
            # Rank and get Top 2
            ranked_stage2 = self._rank_symbols(stage2_results)
            top2_data = ranked_stage2[:2]
            top2_symbols = [item['symbol'] for item in top2_data]
            
            # Save to cache (include both stages for reference)
            self._save_cache(top2_data, {
                "stage1_results": stage1_results,
                "stage2_results": stage2_results,
                "top5": top5_symbols
            })
            
            elapsed = time.time() - start_time
            log.info("=" * 60)
            log.info(f"✅ AUTO3 Two-Stage Selection Complete in {elapsed:.1f}s")
            log.info(f"   Stage 1: {len(candidates)} → 5 symbols (1h backtest)")
            log.info(f"   Stage 2: 5 → 2 symbols (15m backtest)")
            log.info(f"   🎯 Selected: {top2_symbols}")
            log.info("=" * 60)
            
            return top2_symbols
            
        except Exception as e:
            log.error(f"❌ AUTO3 selection failed: {e}", exc_info=True)
            log.warning(f"⚠️ Falling back to default symbols: {self.FALLBACK_SYMBOLS}")
            return self.FALLBACK_SYMBOLS
    
    async def _get_expanded_candidates(self) -> List[str]:
        """Get AI500 Top 10 + Major coins by 24h volume"""
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
            
            # Sort by volume descending and get Top 10
            ai_stats.sort(key=lambda x: x[1], reverse=True)
            ai500_top10 = [x[0] for x in ai_stats[:10]]
            
            # Combine with major coins (avoid duplicates)
            candidates = list(ai500_top10)
            for coin in self.major_coins:
                if coin not in candidates:
                    candidates.append(coin)
            
            log.info(f"📊 AI500 Top 10: {ai500_top10}")
            log.info(f"📊 + Major coins: {self.major_coins}")
            
            return candidates
            
        except Exception as e:
            log.error(f"Failed to get expanded candidates: {e}")
            # Fallback: first 10 AI500 + majors
            return self.ai500_candidates[:10] + self.major_coins
    
    async def _run_backtests_stage(
        self,
        symbols: List[str],
        step: int,
        stage_name: str
    ) -> List[Dict]:
        """Run backtests for a specific stage"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=self.lookback_hours)
        
        valid_results = []
        total = len(symbols)
        
        for i, symbol in enumerate(symbols):
            log.info(f"🔄 [{stage_name}] [{i+1}/{total}] Backtesting {symbol}...")
            try:
                result = await self._backtest_symbol(symbol, start_time, end_time, step)
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
        end_time: datetime,
        step: int = 12
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
                step=step
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
    
    def _save_cache(self, top2: List[Dict], all_results: Dict):
        """Save results to cache"""
        now = datetime.now()
        cache_data = {
            "timestamp": now.isoformat(),
            "valid_until": (now + timedelta(hours=self.refresh_interval)).isoformat(),
            "lookback_hours": self.lookback_hours,
            "selection_method": "two_stage",
            "top2": top2,
            "top5": all_results.get("top5", []),
            "stage1_results": all_results.get("stage1_results", []),
            "stage2_results": all_results.get("stage2_results", [])
        }
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        log.info(f"💾 AUTO3 cache saved: valid until {cache_data['valid_until']}")
    
    def start_auto_refresh(self):
        """Start background thread for auto-refresh every 6 hours"""
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
                log.info(f"🔄 AUTO3 auto-refresh triggered ({self.refresh_interval}h interval)")
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.select_top2(force_refresh=True))
                    loop.close()
                except Exception as e:
                    log.error(f"❌ Auto-refresh failed: {e}", exc_info=True)
        
        self._refresh_thread = threading.Thread(target=refresh_loop, daemon=True, name="AUTO3-Refresh")
        self._refresh_thread.start()
        log.info(f"🔄 AUTO3 auto-refresh started ({self.refresh_interval}h interval)")
    
    def stop_auto_refresh(self):
        """Stop background refresh thread"""
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._stop_refresh.set()
            self._refresh_thread.join(timeout=5)
            log.info("🛑 AUTO3 auto-refresh stopped")


# Global instance
_selector_instance: Optional[SymbolSelectorAgent] = None

def get_selector() -> SymbolSelectorAgent:
    """Get global SymbolSelectorAgent instance (singleton)"""
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = SymbolSelectorAgent()
    return _selector_instance
