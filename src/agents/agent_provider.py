
from src.config import Config
from .agent_config import AgentConfig
from .predict_agents_provider import PredictAgentsProvider
from src.api.binance_client import BinanceClient
from src.trading.symbol_manager import SymbolManager

from src.agents import (
    RiskAuditAgent,
    QuantAnalystAgent,
    RegimeDetector,
    DataSyncAgent
)

from src.agents.reflection import (
    ReflectionAgentLLM,
    ReflectionAgentNoLLM
)

from src.agents.trend import (
    TrendAgentLLM,
    TrendAgentNoLLM
)

from src.agents.setup import (
    SetupAgentLLM,
    SetupAgentNoLLM
)

from src.agents.trigger import (
    TriggerAgentLLM,
    TriggerAgentNoLLM
)

class AgentProvider:
    def __init__(
        self,
        config: Config,
        agent_config: AgentConfig,
        client: BinanceClient,
        symbol_manager: SymbolManager,
    ):
        self.config = config
        self.agent_config = agent_config
        self.client = client
        self.symbol_manager = symbol_manager
        self._set_predict_agents_provider()
        self._set_agents()
        
        self.data_sync_agent = DataSyncAgent(client)

        self.risk_audit_agent = RiskAuditAgent(
            max_leverage=10.0,
            max_position_pct=0.3,
            min_stop_loss_pct=0.005,
            max_stop_loss_pct=0.05
        )
        self.quant_analyst_agent = QuantAnalystAgent()
          
        print("  ✅ DataSyncAgent ready")
        print("  ✅ QuantAnalystAgent ready")
        print("  ✅ RiskAuditAgent ready")

    def reload(self):
        self._set_agents()

        if self.agent_config.predict_agent:
            self.predict_agents_provider.reload()
        else:
            self.predict_agents_provider = None

    def _set_agents(self):
        self._set_reflection_agent()
        self._set_trend_agent()
        self._set_setup_agent()
        self._set_trigger_agent()
        self._set_regime_director_agent()

    def _set_predict_agents_provider(self):
        # 🆕 Optional Agent: PredictAgent (per symbol)
        if self.agent_config.predict_agent:
            print("[DEBUG] Creating PredictAgents...")
            self.predict_agents_provider = PredictAgentsProvider(self.client, self.symbol_manager, self.agent_config)
            print(f"  ✅ PredictAgent ready ({len(self.symbol_manager.symbols)} symbols)")
        else:
            print("  ⏭️ PredictAgent disabled")
            self.predict_agents_provider = None

    def _set_reflection_agent(self):
        # Optional Agent: ReflectionAgent
        if self.agent_config.reflection_agent_llm or self.agent_config.reflection_agent_local:
            if self.agent_config.reflection_agent_llm:
                if not hasattr(self, 'reflection_agent') or not isinstance(self.reflection_agent, ReflectionAgentLLM):
                    self.reflection_agent = ReflectionAgentLLM(self.config)
            else:
                if not hasattr(self, 'reflection_agent') or not isinstance(self.reflection_agent, ReflectionAgentNoLLM):
                    self.reflection_agent = ReflectionAgentNoLLM()
        else:
            print("  ⏭️ ReflectionAgent disabled")
            self.reflection_agent = None

    def _set_trend_agent(self):
        # 🆕 Optional Agent: TrendAgent
        if self.agent_config.trend_agent_llm or self.agent_config.trend_agent_local:       
            if self.agent_config.trend_agent_llm:
                if not hasattr(self, 'trend_agent') or not isinstance(self.trend_agent, TrendAgentLLM):
                    self.trend_agent = TrendAgentLLM(self.config)
            else:
                if not hasattr(self, 'trend_agent') or not isinstance(self.trend_agent, TrendAgentNoLLM):
                    self.trend_agent = TrendAgentNoLLM()
        else:
            print("  ⏭️ TrendAgent disabled")
            self.trend_agent = None

    def _set_setup_agent(self):
        # 🆕 Optional Agent: SetupAgent
        if self.agent_config.setup_agent_llm or self.agent_config.setup_agent_local:       
            if self.agent_config.setup_agent_llm:
                if not hasattr(self, 'setup_agent') or not isinstance(self.setup_agent, SetupAgentLLM):
                    self.setup_agent = SetupAgentLLM(self.config)
            else:
                if not hasattr(self, 'setup_agent') or not isinstance(self.setup_agent, SetupAgentNoLLM):
                    self.setup_agent = SetupAgentNoLLM()
        else:
            print("  ⏭️ SetupAgent disabled")
            self.setup_agent = None
    
    def _set_trigger_agent(self):
        # 🆕 Optional Agent: TriggerAgent
        if self.agent_config.trigger_agent_llm or self.agent_config.trigger_agent_local:       
            if self.agent_config.trigger_agent_llm:
                if not hasattr(self, 'trigger_agent') or not isinstance(self.trigger_agent, TriggerAgentLLM):
                    self.trigger_agent = TriggerAgentLLM(self.config)
            else:
                if not hasattr(self, 'trigger_agent') or not isinstance(self.trigger_agent, TriggerAgentNoLLM):
                    self.trigger_agent = TriggerAgentNoLLM()
        else:
            print("  ⏭️ TriggerAgent disabled")
            self.trigger_agent = None

    def _set_regime_director_agent(self):
        # 🆕 Optional Agent: RegimeDirectorAgent
        if self.agent_config.regime_director_agent:
            if not hasattr(self, 'regime_director_agent'):
                self.regime_director_agent = RegimeDetector()
        else:
            print("  ⏭️ RegimeDirectorAgent disabled")
            self.regime_director_agent = None