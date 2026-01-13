"""
Agent Configuration Module
===========================

Provides centralized configuration for optional agents.
Core agents (DataSyncAgent, QuantAnalystAgent, RiskAuditAgent) are always enabled.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class AgentConfig:
    """
    Configuration for optional agents.
    
    Core agents are always enabled and not configurable:
    - DataSyncAgent: Market data fetching
    - QuantAnalystAgent: Technical analysis
    - RiskAuditAgent: Risk control
    
    Optional agents can be enabled/disabled via config.
    """
    
    # ML/AI Prediction Layer
    predict_agent: bool = True              # PredictAgent: ML probability prediction
    ai_prediction_filter_agent: bool = True  # AIPredictionFilterAgent: AI veto mechanism
    
    # Market Analysis
    regime_detector_agent: bool = True       # RegimeDetectorAgent: Market state detection
    position_analyzer_agent: bool = False    # PositionAnalyzerAgent: Price position analysis
    
    # Trigger Detection
    trigger_detector_agent: bool = True      # TriggerDetectorAgent: 5m pattern detection
    
    # LLM Semantic Analysis (expensive, disabled by default)
    trend_agent: bool = False                # TrendAgent: 1h trend LLM analysis
    trigger_agent: bool = False              # TriggerAgent: 5m trigger LLM analysis
    
    # Trading Retrospection
    reflection_agent: bool = True            # ReflectionAgent: Trade reflection
    
    # Symbol Selection
    symbol_selector_agent: bool = False      # SymbolSelectorAgent: AUTO3 selection
    
    def __post_init__(self):
        """Validate dependencies between agents"""
        # AIPredictionFilterAgent requires PredictAgent
        if self.ai_prediction_filter_agent and not self.predict_agent:
            self.ai_prediction_filter_agent = False
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'AgentConfig':
        """
        Create AgentConfig from a dictionary (e.g., from config.yaml)
        
        Environment variables take priority over config values.
        Use AGENT_<NAME>=true/false to override, e.g., AGENT_PREDICT_AGENT=false
        
        Args:
            config: Dictionary with agent enable/disable settings
            
        Returns:
            AgentConfig instance
        """
        import os
        agents_config = config.get('agents', {})
        
        def get_value(key: str, default: bool) -> bool:
            """Get value from env var (priority) or config or default"""
            env_key = f"AGENT_{key.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                return env_val.lower() in ('true', '1', 'yes', 'on')
            return agents_config.get(key, default)
        
        # Map config keys to dataclass fields
        return cls(
            predict_agent=get_value('predict_agent', True),
            ai_prediction_filter_agent=get_value('ai_prediction_filter_agent', True),
            regime_detector_agent=get_value('regime_detector_agent', True),
            position_analyzer_agent=get_value('position_analyzer_agent', False),
            trigger_detector_agent=get_value('trigger_detector_agent', True),
            trend_agent=get_value('trend_agent', False),
            trigger_agent=get_value('trigger_agent', False),
            reflection_agent=get_value('reflection_agent', True),
            symbol_selector_agent=get_value('symbol_selector_agent', False),
        )
    
    def is_enabled(self, agent_name: str) -> bool:
        """
        Check if an agent is enabled by name.
        
        Args:
            agent_name: Agent name (e.g., 'predict_agent', 'regime_detector_agent')
            
        Returns:
            True if enabled, False otherwise
        """
        # Convert CamelCase to snake_case if needed
        if agent_name.endswith('Agent') and not agent_name.endswith('_agent'):
            # Convert e.g., "PredictAgent" -> "predict_agent"
            name = ''.join(['_' + c.lower() if c.isupper() else c for c in agent_name]).lstrip('_')
        else:
            name = agent_name
            
        return getattr(self, name, False)
    
    def get_enabled_agents(self) -> Dict[str, bool]:
        """Get dictionary of all agent enabled states"""
        return {
            'predict_agent': self.predict_agent,
            'ai_prediction_filter_agent': self.ai_prediction_filter_agent,
            'regime_detector_agent': self.regime_detector_agent,
            'position_analyzer_agent': self.position_analyzer_agent,
            'trigger_detector_agent': self.trigger_detector_agent,
            'trend_agent': self.trend_agent,
            'trigger_agent': self.trigger_agent,
            'reflection_agent': self.reflection_agent,
            'symbol_selector_agent': self.symbol_selector_agent,
        }
    
    def __str__(self) -> str:
        enabled = [k for k, v in self.get_enabled_agents().items() if v]
        disabled = [k for k, v in self.get_enabled_agents().items() if not v]
        return f"AgentConfig(enabled={enabled}, disabled={disabled})"
