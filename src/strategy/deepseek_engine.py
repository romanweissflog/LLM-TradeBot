"""
LLM 策略推理引擎 (Multi-Provider Support)
=========================================

支持多种 LLM 提供商: OpenAI, DeepSeek, Claude, Qwen, Gemini
"""
import json
from typing import Dict, Optional
from src.config import config
from src.utils.logger import log
from src.strategy.llm_parser import LLMOutputParser
from src.strategy.decision_validator import DecisionValidator
from src.llm import create_client, LLMConfig


class StrategyEngine:
    """多 LLM 提供商策略决策引擎"""
    
    def __init__(self):
        # 获取 LLM 配置
        llm_config = config.llm
        provider = llm_config.get('provider', 'deepseek')
        
        # 获取对应提供商的 API Key
        api_keys = llm_config.get('api_keys', {})
        api_key = api_keys.get(provider)
        
        # 向后兼容: 如果没有新配置，使用旧的 deepseek 配置
        if not api_key and provider == 'deepseek':
            api_key = config.deepseek.get('api_key')
        
        if not api_key:
            raise ValueError(f"No API key found for provider: {provider}")
        
        # LLM 参数
        self.provider = provider
        self.model = llm_config.get('model') or config.deepseek.get('model', 'deepseek-chat')
        self.temperature = llm_config.get('temperature', config.deepseek.get('temperature', 0.3))
        self.max_tokens = llm_config.get('max_tokens', config.deepseek.get('max_tokens', 2000))
        
        # 创建 LLM 客户端
        llm_cfg = LLMConfig(
            api_key=api_key,
            base_url=llm_config.get('base_url'),
            model=self.model,
            timeout=llm_config.get('timeout', 120),
            max_retries=llm_config.get('max_retries', 3),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        self.client = create_client(provider, llm_cfg)
        
        # 初始化解析器和验证器
        self.parser = LLMOutputParser()
        self.validator = DecisionValidator({
            'max_leverage': config.risk.get('max_leverage', 5),
            'max_position_pct': config.risk.get('max_total_position_pct', 30.0),
            'min_risk_reward_ratio': 2.0
        })
        
        log.info(f"🤖 策略引擎初始化完成 (Provider: {provider}, Model: {self.model})")
    
    def make_decision(self, market_context_text: str, market_context_data: Dict) -> Dict:
        """
        基于市场上下文做出交易决策
        
        Args:
            market_context_text: 格式化的市场上下文文本
            market_context_data: 原始市场数据
            
        Returns:
            决策结果字典
        """
        
        # 🐂🐻 Get adversarial perspectives first
        log.info("🐂🐻 Gathering Bull/Bear perspectives...")
        bull_perspective = self.get_bull_perspective(market_context_text)
        bear_perspective = self.get_bear_perspective(market_context_text)
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(market_context_text, bull_perspective, bear_perspective)
        
        # 记录 LLM 输入
        log.llm_input(f"正在发送市场数据到 {self.provider}...", market_context_text)

        
        try:
            response = self.client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # 获取原始响应
            content = response.content
            
            # 使用新解析器解析结构化输出
            parsed = self.parser.parse(content)
            decision = parsed['decision']
            reasoning = parsed['reasoning']
            
            # 标准化 action 字段
            if 'action' in decision:
                decision['action'] = self.parser.normalize_action(decision['action'])
            
            # 验证决策
            is_valid, errors = self.validator.validate(decision)
            if not is_valid:
                log.warning(f"LLM 决策验证失败: {errors}")
                log.warning(f"原始决策: {decision}")
                return self._get_fallback_decision(market_context_data)
            
            # 记录 LLM 输出
            log.llm_output(f"{self.provider} 返回决策结果", decision)
            if reasoning:
                log.info(f"推理过程:\n{reasoning}")
            
            # 记录决策
            log.llm_decision(
                action=decision.get('action', 'hold'),
                confidence=decision.get('confidence', 0),
                reasoning=decision.get('reasoning', reasoning)
            )
            
            # 添加元数据
            decision['timestamp'] = market_context_data['timestamp']
            decision['symbol'] = market_context_data['symbol']
            decision['model'] = self.model
            decision['raw_response'] = content
            decision['reasoning_detail'] = reasoning
            decision['validation_passed'] = True
            
            # ✅ Return full prompt for logging
            decision['system_prompt'] = system_prompt
            decision['user_prompt'] = user_prompt
            
            # 🐂🐻 Add Bull/Bear perspectives for dashboard
            decision['bull_perspective'] = bull_perspective
            decision['bear_perspective'] = bear_perspective
            
            return decision
            
        except Exception as e:
            log.error(f"LLM decision failed: {e}")
            # 返回保守决策
            return self._get_fallback_decision(market_context_data)
    
    def get_bull_perspective(self, market_context_text: str) -> Dict:
        """
        🐂 Bull Agent: Analyze market from bullish perspective
        
        Args:
            market_context_text: Formatted market context
            
        Returns:
            Dict with bullish_reasons and bull_confidence
        """
        bull_prompt = """You are a BULLISH market analyst. Your job is to find reasons WHY the market could go UP.

Analyze the provided market data and identify:
1. Bullish technical signals (support levels, RSI oversold, MACD crossovers)
2. Positive trend indicators
3. Entry opportunities for LONG positions

Output your analysis in this EXACT JSON format:
```json
{
  "stance": "STRONGLY_BULLISH",
  "bullish_reasons": "Your 3-5 key bullish observations, separated by semicolons",
  "bull_confidence": 75
}
```

stance must be one of: STRONGLY_BULLISH, SLIGHTLY_BULLISH, NEUTRAL, UNCERTAIN
bull_confidence should be 0-100 based on how strong the bullish case is.
Focus ONLY on bullish factors. Ignore bearish signals."""

        try:
            response = self.client.chat(
                system_prompt=bull_prompt,
                user_prompt=market_context_text,
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.content
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                stance = result.get('stance', 'UNKNOWN')
                log.info(f"🐂 Bull Agent: [{stance}] {result.get('bullish_reasons', '')[:40]}... (Conf: {result.get('bull_confidence', 0)}%)")
                return result
            
            return {"bullish_reasons": "Unable to analyze", "bull_confidence": 50}
            
        except Exception as e:
            log.warning(f"Bull Agent failed: {e}")
            return {"bullish_reasons": "Analysis unavailable", "bull_confidence": 50}
    
    def get_bear_perspective(self, market_context_text: str) -> Dict:
        """
        🐻 Bear Agent: Analyze market from bearish perspective
        
        Args:
            market_context_text: Formatted market context
            
        Returns:
            Dict with bearish_reasons and bear_confidence
        """
        bear_prompt = """You are a BEARISH market analyst. Your job is to find reasons WHY the market could go DOWN.

Analyze the provided market data and identify:
1. Bearish technical signals (resistance levels, RSI overbought, bearish divergence)
2. Negative trend indicators
3. Entry opportunities for SHORT positions or exit warnings for LONG

Output your analysis in this EXACT JSON format:
```json
{
  "stance": "STRONGLY_BEARISH",
  "bearish_reasons": "Your 3-5 key bearish observations, separated by semicolons",
  "bear_confidence": 60
}
```

stance must be one of: STRONGLY_BEARISH, SLIGHTLY_BEARISH, NEUTRAL, UNCERTAIN
bear_confidence should be 0-100 based on how strong the bearish case is.
Focus ONLY on bearish factors. Ignore bullish signals."""


        try:
            response = self.client.chat(
                system_prompt=bear_prompt,
                user_prompt=market_context_text,
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.content
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                stance = result.get('stance', 'UNKNOWN')
                log.info(f"🐻 Bear Agent: [{stance}] {result.get('bearish_reasons', '')[:40]}... (Conf: {result.get('bear_confidence', 0)}%)")
                return result
            
            return {"bearish_reasons": "Unable to analyze", "bear_confidence": 50}
            
        except Exception as e:
            log.warning(f"Bear Agent failed: {e}")
            return {"bearish_reasons": "Analysis unavailable", "bear_confidence": 50}
    
    def _build_system_prompt(self) -> str:
        """Build System Prompt (English Version) or Load Custom"""
        import os
        
        # Check for custom prompt
        # Assuming src/strategy/deepseek_engine.py, so config is ../../config/custom_prompt.md
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        custom_prompt_path = os.path.join(base_dir, 'config', 'custom_prompt.md')
        
        if os.path.exists(custom_prompt_path):
            try:
                with open(custom_prompt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        log.info("📝 Loading Custom System Prompt from file")
                        return content
            except Exception as e:
                log.error(f"Failed to load custom prompt: {e}")
        
        # Load default from template
        try:
            from src.config.default_prompt_template import DEFAULT_SYSTEM_PROMPT
            return DEFAULT_SYSTEM_PROMPT
        except ImportError:
            log.error("Failed to import DEFAULT_SYSTEM_PROMPT")
            return "Error: Default prompt missing"
    
    def _build_user_prompt(self, market_context: str, bull_perspective: Dict = None, bear_perspective: Dict = None) -> str:
        """Build User Prompt (English Version) with Bull/Bear perspectives"""
        
        # Build adversarial analysis section
        adversarial_section = ""
        if bull_perspective or bear_perspective:
            bull_reasons = bull_perspective.get('bullish_reasons', 'N/A') if bull_perspective else 'N/A'
            bull_conf = bull_perspective.get('bull_confidence', 50) if bull_perspective else 50
            bull_stance = bull_perspective.get('stance', 'UNKNOWN') if bull_perspective else 'UNKNOWN'
            bear_reasons = bear_perspective.get('bearish_reasons', 'N/A') if bear_perspective else 'N/A'
            bear_conf = bear_perspective.get('bear_confidence', 50) if bear_perspective else 50
            bear_stance = bear_perspective.get('stance', 'UNKNOWN') if bear_perspective else 'UNKNOWN'
            
            adversarial_section = f"""

## 🐂🐻 Adversarial Analysis (Consider BOTH perspectives)

### 🐂 Bull Agent [{bull_stance}] (Confidence: {bull_conf}%)
{bull_reasons}

### 🐻 Bear Agent [{bear_stance}] (Confidence: {bear_conf}%)
{bear_reasons}

> **IMPORTANT**: Weigh both perspectives. If one side has significantly higher confidence (>20% difference), lean towards that direction. If similar, prefer `wait`.

---
"""
        
        return f"""# 📊 Real-Time Market Data (Technical Analysis Completed)

The system has prepared the following complete market status for **5m/15m/1h** timeframes:

{market_context}
{adversarial_section}
---

## 🎯 Your Task

Please follow this flow for analysis and decision-making:

### 1️⃣ Multi-Timeframe Trend Judgment (Mandatory)
- Analyze **1h** main trend direction (SMA/MACD)
- Check **15m** for confluence with 1h
- Observe **5m** for short-term momentum

### 2️⃣ Key Indicator Confirmation (Mandatory)
- Is RSI in reasonable range (30-70)?
- Is MACD histogram expanding (momentum up) or contracting?
- Does Volume support the trend?
- Is ATR showing abnormal volatility?

### 3️⃣ Risk Assessment (Mandatory)
- Are there extreme indicators (RSI>80 or <20)?
- Are timeframes contradicting?
- Is liquidity (Volume) sufficient?

### 4️⃣ Entry Timing (If Opening)
- Where is price relative to Support/Resistance?
- Is there a clear entry signal (Breakout/Pullback/Cross)?
- Is Risk-Reward Ratio ≥ 2?

### 5️⃣ Stop Loss / Take Profit (If Opening)
- Calculate logical SL distance using ATR
- **Verify SL Direction**:
  - Long: stop_loss < entry_price
  - Short: stop_loss > entry_price
- TP must be at least 2x risk

---

## ⚡ Output Format Requirements (Mandatory)

1. **Use <reasoning> and <decision> XML tags**
2. **JSON must be wrapped in ```json code block**
3. **JSON must be an array format `[{{...}}]`**, starting with `[{{`
4. **reasoning field is required**: One sentence summary in English (under 50 words)
5. **Prohibited**: Range symbols `~`, thousand separators `,`, Chinese comments

---

## 🚨 Format Example

<reasoning>
1h: [trend analysis]
15m: [confluence check]
5m: [entry timing]
Risk: [assessment]
</reasoning>

<decision>
```json
[
  {{
    "symbol": "BTCUSDT",
    "action": "wait",
    "confidence": 45,
    "reasoning": "Weak signals, await clearer entry"
  }}
]
```
</decision>


Please start your analysis and output the decision in JSON Array format `[{{...}}]`.
"""
    
    def _get_fallback_decision(self, context: Dict) -> Dict:
        """
        获取兜底决策（当LLM失败时）
        
        返回保守的hold决策
        """
        return {
            'action': 'wait',
            'symbol': context.get('symbol', 'BTCUSDT'),
            'confidence': 0,
            'leverage': 1,
            'position_size_pct': 0,
            'stop_loss_pct': 1.0,
            'take_profit_pct': 2.0,
            'reasoning': 'LLM decision failed, using conservative fallback strategy',
            'timestamp': context.get('timestamp'),
            'is_fallback': True
        }
    
    def validate_decision(self, decision: Dict) -> bool:
        """
        验证决策格式是否正确
        
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'action', 'symbol', 'confidence', 'leverage',
            'position_size_pct', 'stop_loss_pct', 'take_profit_pct', 'reasoning'
        ]
        
        # 检查必需字段
        for field in required_fields:
            if field not in decision:
                log.error(f"决策缺少必需字段: {field}")
                return False
        
        # 检查action合法性
        valid_actions = [
            'open_long', 'open_short', 'close_position',
            'add_position', 'reduce_position', 'hold'
        ]
        if decision['action'] not in valid_actions:
            log.error(f"无效的action: {decision['action']}")
            return False
        
        # 检查数值范围
        if not (0 <= decision['confidence'] <= 100):
            log.error(f"confidence超出范围: {decision['confidence']}")
            return False
        
        # STRICT ENFORCEMENT: Open trades must have confidence >= 80
        action = decision['action']
        confidence = decision['confidence']
        if action in ['open_long', 'open_short'] and confidence < 80:
            log.warning(f"🚫 Confidence < 80 ({confidence}%) for {action}, converting to 'wait'")
            decision['action'] = 'wait'
            decision['reasoning'] = f"Low confidence ({confidence}% < 80%), wait for better setup"
        
        if not (1 <= decision['leverage'] <= config.risk.get('max_leverage', 5)):
            log.error(f"leverage超出范围: {decision['leverage']}")
            return False
        
        if not (0 <= decision['position_size_pct'] <= 100):
            log.error(f"position_size_pct超出范围: {decision['position_size_pct']}")
            return False
        
        return True
