
from typing import Dict, Any, TYPE_CHECKING

from src.agents.runtime_events import emit_global_runtime_event, emit_cycle_pipeline_end

from src.trading.cycle_context import CycleContext
from src.trading.symbol_manager import SymbolManager

if TYPE_CHECKING:
    from .runner_provider import RunnerProvider

class CyclePipelineRunner:
    def __init__(
        self,
        symbol_manager: SymbolManager,
        runner_provider: "RunnerProvider"
    ):
        self.symbol_manager = symbol_manager
        self.runner_provider = runner_provider
    
    async def run(self, *, context: CycleContext, analyze_only: bool) -> Dict[str, Any]:
        """Run the full trading pipeline using a prepared cycle context."""
        emit_global_runtime_event(
            run_id=context.run_id,
            stream="lifecycle",
            agent="cycle_pipeline",
            phase="start",
            cycle_id=context.cycle_id,
            symbol=self.symbol_manager.current_symbol,
            data={"symbol": context.symbol, "cycle": context.cycle_num}
        )
        try:
            oracle_result = await self.runner_provider.oracle_stage_runner.run(
                run_id=context.run_id,
                cycle_id=context.cycle_id,
                snapshot_id=context.snapshot_id
            )
            if oracle_result.early_result is not None:
                early = oracle_result.early_result
                emit_cycle_pipeline_end(context=context, result=early)
                return early

            market_snapshot = oracle_result.payload['market_snapshot']
            processed_dfs = oracle_result.payload['processed_dfs']
            current_price = oracle_result.payload['current_price']
            current_position_info = oracle_result.payload['current_position_info']

            quant_analysis, predict_result, _reflection_result, reflection_text = await self.runner_provider.agent_analysis_stage_runner.run(
                run_id=context.run_id,
                cycle_id=context.cycle_id,
                snapshot_id=context.snapshot_id,
                market_snapshot=market_snapshot,
                processed_dfs=processed_dfs
            )

            regime_result, four_layer_result, trend_1h = self.runner_provider.four_layer_filter_stage_runner.run(
                run_id=context.run_id,
                cycle_id=context.cycle_id,
                quant_analysis=quant_analysis,
                predict_result=predict_result,
                processed_dfs=processed_dfs,
                current_price=current_price
            )

            await self.runner_provider.post_filter_stage_runner.run(
                run_id=context.run_id,
                cycle_id=context.cycle_id,
                current_price=current_price,
                trend_1h=trend_1h,
                four_layer_result=four_layer_result,
                quant_analysis=quant_analysis,
                processed_dfs=processed_dfs
            )

            decision_result = await self.runner_provider.decision_pipeline_stage_runner.run(
                run_id=context.run_id,
                cycle_id=context.cycle_id,
                snapshot_id=context.snapshot_id,
                processed_dfs=processed_dfs,
                quant_analysis=quant_analysis,
                predict_result=predict_result,
                reflection_text=reflection_text,
                current_price=current_price,
                current_position_info=current_position_info,
                regime_result=regime_result
            )
            if decision_result.early_result is not None:
                early = decision_result.early_result
                emit_cycle_pipeline_end(context=context, result=early)
                return early
            vote_result = decision_result.payload['vote_result']

            result = await self.runner_provider.action_pipeline_stage_runner.run(
                run_id=context.run_id,
                cycle_id=context.cycle_id,
                snapshot_id=context.snapshot_id,
                analyze_only=analyze_only,
                vote_result=vote_result,
                quant_analysis=quant_analysis,
                predict_result=predict_result,
                processed_dfs=processed_dfs,
                current_price=current_price,
                current_position_info=current_position_info,
                regime_result=regime_result,
                market_snapshot=market_snapshot
            )
            emit_cycle_pipeline_end(context=context, result=result)
            return result
        except Exception as e:
            emit_global_runtime_event(
                run_id=context.run_id,
                stream="error",
                agent="cycle_pipeline",
                phase="error",
                cycle_id=context.cycle_id,
                symbol=self.symbol_manager.current_symbol,
                data={"error": str(e)}
            )
            raise
