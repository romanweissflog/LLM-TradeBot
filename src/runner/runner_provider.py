from src.agents.agent_config import AgentConfig
from src.config import Config

from src.strategy.llm_engine import StrategyEngine
from src.api.binance_client import BinanceClient
from src.utils.data_saver import DataSaver
from src.agents.agent_provider import AgentProvider

from src.trading.symbol_manager import SymbolManager

from src.runner import (
    ActionPipelineStageRunner,
    AgentAnalysisStageRunner,
    CyclePipelineRunner,
    DecisionStageRunner,
    ExecutionStageRunner,
    OracleStageRunner,
    ParallelAnalysisRunner,
    PostFilterStageRunner,
    RiskAuditStageRunner,
    SemanticAnalysisRunner,
    FourLayerFilterStageRunner
)

class RunnerProvider:
    def __init__(
        self,
        config: Config,
        agent_config: AgentConfig,
        client: BinanceClient,
        symbol_manager: SymbolManager,
        agent_provider: AgentProvider,
        strategy_engine: StrategyEngine,
        saver: DataSaver,
        max_position_size: float,
        leverage: int,
        stop_loss_pct: float,
        take_profit_pct: float,
        kline_limit: int,
        test_mode: bool = False
    ):
        self.action_pipeline_stage_runner = ActionPipelineStageRunner(
            symbol_manager,
            self
        )

        self.agent_analysis_stage_runner = AgentAnalysisStageRunner(
            symbol_manager,
            self,
            saver
        )

        self.cycle_pipeline_runner = CyclePipelineRunner(
            symbol_manager,
            self
        )

        self.decision_stage_runner = DecisionStageRunner(
            agent_config,
            symbol_manager,
            strategy_engine,
            max_position_size
        )

        self.execution_stage_runner = ExecutionStageRunner(
            symbol_manager,
            saver,
            test_mode
        )

        self.four_layer_filter_stage_runner = FourLayerFilterStageRunner(
            config,
            agent_config,
            symbol_manager,
            agent_provider
        )

        self.oracle_stage_runner = OracleStageRunner(
            config,
            agent_config,
            client,
            symbol_manager,
            saver,
            kline_limit,
            test_mode
        )
        
        self.parallel_analysis_runner = ParallelAnalysisRunner(
            config,
            agent_config,
            symbol_manager,
            agent_provider,
            saver
        )

        self.post_filter_stage_runner = PostFilterStageRunner(
            config,
            agent_config,
            symbol_manager,
            agent_provider
        )

        self.risk_audit_stage_runner = RiskAuditStageRunner(
            config,
            agent_config,
            client,
            symbol_manager,
            agent_provider,
            leverage,
            stop_loss_pct,
            take_profit_pct,
            test_mode
        )

        self.semantic_analysis_runner = SemanticAnalysisRunner(
            config,
            agent_config,
            symbol_manager,
            agent_provider
        )