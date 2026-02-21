from typing import Dict, Optional, Any

from src.agents.runtime_events import emit_global_runtime_event
from src.agents.agent_provider import AgentProvider

from src.utils.logger import log
from src.server.state import global_state

from src.trading.symbol_manager import SymbolManager

class PostFilterStageRunner:
    def __init__(
        self,
        symbol_manager: SymbolManager,
        agent_provider: AgentProvider,
        runner_provider
    ):
        self.symbol_manager = symbol_manager
        self.agent_provider = agent_provider  
        self.runner_provider = runner_provider

    async def run(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        current_price: float,
        trend_1h: str,
        four_layer_result: Dict[str, Any],
        quant_analysis: Dict[str, Any],
        processed_dfs: Dict[str, "pd.DataFrame"]
    ) -> None:
        """Run semantic agents + multi-period parser after four-layer filtering."""
        emit_global_runtime_event(
            run_id=run_id,
            stream="lifecycle",
            agent="post_filter_stage",
            phase="start",
            cycle_id=cycle_id,
            symbol=self.symbol_manager.current_symbol
        )
        try:
            await self.runner_provider.semantic_analysis_runner.run(
                run_id=run_id,
                cycle_id=cycle_id,
                current_price=current_price,
                trend_1h=trend_1h,
                four_layer_result=four_layer_result,
                processed_dfs=processed_dfs
            )

            try:
                multi_period_result = self.agent_provider.multi_period_agent.analyze(
                    quant_analysis=quant_analysis,
                    four_layer_result=four_layer_result,
                    semantic_analyses=getattr(global_state, 'semantic_analyses', {}) or {}
                )
                global_state.multi_period_result = multi_period_result
                summary = multi_period_result.get('summary')
                if summary:
                    global_state.add_agent_message("multi_period_agent", summary, level="info")
            except Exception as e:
                log.error(f"❌ Multi-Period Parser Agent failed: {e}")
                global_state.multi_period_result = {}

            emit_global_runtime_event(
                run_id=run_id,
                stream="lifecycle",
                agent="post_filter_stage",
                phase="end",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol
            )
        except Exception as e:
            emit_global_runtime_event(
                run_id=run_id,
                stream="error",
                agent="post_filter_stage",
                phase="error",
                cycle_id=cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"error": str(e)}
            )
            raise

