"""
Multi-Agent Trading System

基于异步并发的多Agent交易架构

Core Agents (always enabled):
- DataSyncAgent: Market data fetching
- QuantAnalystAgent: Technical analysis
- RiskAuditAgent: Risk control

Optional Agents (configurable via AgentConfig):
- PredictAgent, ReflectionAgent, RegimeDetectorAgent, etc.
"""

# Agent Framework
from .agent_config import AgentConfig
from .base_agent import BaseAgent, AgentResult
from .agent_registry import AgentRegistry

# Core Agents
from .data_sync_agent import DataSyncAgent, MarketSnapshot
from .quant_analyst_agent import QuantAnalystAgent
from .decision_core_agent import DecisionCoreAgent, VoteResult, SignalWeight
from .risk_audit_agent import RiskAuditAgent, RiskCheckResult, PositionInfo, RiskLevel

# Optional Agents
from .predict_result import PredictResult
from .predict_agent import PredictAgent
from .multi_period_agent import MultiPeriodParserAgent

__all__ = [
    # Framework
    'AgentConfig',
    'BaseAgent',
    'AgentResult',
    'AgentRegistry',
    # Core Agents
    'DataSyncAgent',
    'MarketSnapshot',
    'QuantAnalystAgent',
    'DecisionCoreAgent',
    'VoteResult',
    'SignalWeight',
    'RiskAuditAgent',
    'RiskCheckResult',
    'PositionInfo',
    'RiskLevel',
    # Optional Agents
    'PredictAgent',
    'PredictResult',
    'MultiPeriodParserAgent',
]
