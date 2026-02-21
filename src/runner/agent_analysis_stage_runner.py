from typing import Dict, Optional, Any, Tuple

from src.utils.data_saver import DataSaver

from src.agents.runtime_events import emit_global_runtime_event

from src.trading.symbol_manager import SymbolManager
from src.agents.predict_result import PredictResult  # ✅ PredictResult Import
from src.utils.logger import log
from src.server.state import global_state

from .runner_provider import RunnerProvider

class AgentAnalysisStageRunner:
    def __init__(
        self,
        symbol_manager: SymbolManager,
        runner_provider: RunnerProvider,
        saver: DataSaver
    ):
        self.symbol_manager = symbol_manager
        self.saver = saver
        self.runner_provider = runner_provider

    async def run(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        snapshot_id: str,
        market_snapshot: Any,
        processed_dfs: Dict[str, "pd.DataFrame"]
    ) -> Tuple[Dict[str, Any], PredictResult, Any, Optional[str]]:
        """Run Step 2 parallel analysts and persist analysis context."""
        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="analysis_stage",
            phase="start",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol
        )
        try:
            global_state.add_log(f"[📊 SYSTEM] Parallel analysis started for {self.symbol_manager.current_symbol}")

            quant_analysis, predict_result, reflection_result, reflection_text = await self.runner_provider.parallel_analysis_runner.run(
                run_id=run_id,
                cycle_id=cycle_id,
                snapshot_id=snapshot_id,
                market_snapshot=market_snapshot,
                processed_dfs=processed_dfs
            )

            try:
                df_15m = processed_dfs['15m']
                if 'macd_diff' in df_15m.columns:
                    macd_val = float(df_15m['macd_diff'].iloc[-1])
                    if 'trend' not in quant_analysis:
                        quant_analysis['trend'] = {}
                    if 'details' not in quant_analysis['trend']:
                        quant_analysis['trend']['details'] = {}
                    quant_analysis['trend']['details']['15m_macd_diff'] = macd_val
            except Exception as e:
                log.warning(f"Failed to inject MACD data: {e}")

            self.saver.save_context(quant_analysis, self.symbol_manager.current_symbol, 'analytics', snapshot_id, cycle_id=cycle_id)

            trend_score = quant_analysis.get('trend', {}).get('total_trend_score', 0)
            osc_score = quant_analysis.get('oscillator', {}).get('total_osc_score', 0)
            sent_score = quant_analysis.get('sentiment', {}).get('total_sentiment_score', 0)
            global_state.add_log(f"[👨‍🔬 STRATEGIST] Trend={trend_score:+.0f} | Osc={osc_score:+.0f} | Sent={sent_score:+.0f}")

            emit_global_runtime_event(
                run_id=run_id,
                stream="lifecycle",
                agent="analysis_stage",
                phase="end",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol
            )
            return quant_analysis, predict_result, reflection_result, reflection_text
        except Exception as e:
            emit_global_runtime_event(
                run_id=run_id,
                stream="error",
                agent="analysis_stage",
                phase="error",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"error": str(e)}
            )
            raise
