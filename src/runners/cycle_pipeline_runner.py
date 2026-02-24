
from typing import Dict, Any, TYPE_CHECKING

from src.agents.runtime_events import emit_global_runtime_event, emit_cycle_pipeline_end

from src.trading import CycleContext

from .runner_decorators import log_run

if TYPE_CHECKING:
    from .runner_provider import RunnerProvider

class CyclePipelineRunner:
    def __init__(
        self,
        runner_provider: "RunnerProvider"
    ):
        self.runner_provider = runner_provider
    
    @log_run
    async def run(self, *, context: CycleContext) -> Dict[str, Any]:
        """Run the full trading pipeline using a prepared cycle context."""
        emit_global_runtime_event(
            context,
            stream="lifecycle",
            agent="cycle_pipeline",
            phase="start",
            data={"symbol": context.symbol, "cycle": context.cycle_num}
        )
        try:
            oracle_result = await self.runner_provider.oracle_stage_runner.run(context)
            if oracle_result.early_result is not None:
                early = oracle_result.early_result
                emit_cycle_pipeline_end(context=context, result=early)
                return early

            context.market_snapshot = oracle_result.payload['market_snapshot']
            context.processed_dfs = oracle_result.payload['processed_dfs']
            context.current_price = oracle_result.payload['current_price']
            context.current_position_info = oracle_result.payload['current_position_info']

            context.quant_analysis, context.predict_result, context.reflection_result, context.reflection_text = await self.runner_provider.agent_analysis_stage_runner.run(context)

            context.regime_result, context.four_layer_result, context.trend_1h = self.runner_provider.four_layer_filter_stage_runner.run(context)

            await self.runner_provider.post_filter_stage_runner.run(context)

            decision_result = await self.runner_provider.decision_pipeline_stage_runner.run(context)
            if decision_result.early_result is not None:
                early = decision_result.early_result
                emit_cycle_pipeline_end(context, result=early)
                return early
            context.vote_result = decision_result.payload['vote_result']

            result = await self.runner_provider.action_pipeline_stage_runner.run(context)
            emit_cycle_pipeline_end(context=context, result=result)
            return result
        except Exception as e:
            emit_global_runtime_event(
                context,
                stream="error",
                agent="cycle_pipeline",
                phase="error",
                data={"error": str(e)}
            )
            raise
