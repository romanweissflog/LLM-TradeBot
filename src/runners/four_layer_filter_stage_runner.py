from typing import Dict, Optional, Any, Tuple

from src.config import Config
from src.agents.agent_config import AgentConfig
from src.agents.ai_prediction_filter_agent import AIPredictionFilter
from src.agents.runtime_events import emit_global_runtime_event
from src.agents.agent_provider import AgentProvider

from src.utils.logger import log
from src.server.state import global_state

from src.trading import CycleContext

from .runner_decorators import log_run

class FourLayerFilterStageRunner:
    def __init__(
        self,
        config: Config,
        agent_config: AgentConfig,
        agent_provider: AgentProvider
    ):
        self.config = config
        self.agent_config = agent_config
        self.agent_provider = agent_provider

    @log_run
    def run(
        self,
        context: CycleContext
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """Run four-layer strategy filtering and return regime/four-layer outputs."""
        emit_global_runtime_event(
            context,
            stream="lifecycle",
            agent="four_layer_filter",
            phase="start"
        )

        sentiment = context.quant_analysis.get('sentiment', {})
        oi_fuel = sentiment.get('oi_fuel', {})

        funding_rate = sentiment.get('details', {}).get('funding_rate', 0)
        if funding_rate is None:
            funding_rate = 0

        df_1h = context.processed_dfs['1h']
        if self.agent_provider.regime_detector_agent and len(df_1h) >= 20:
            regime_result = self.agent_provider.regime_detector_agent.detect_regime(df_1h)
        else:
            regime_result = {'adx': 20, 'regime': 'unknown', 'confidence': 0}
        adx_value = regime_result.get('adx', 20)

        four_layer_result = {
            'layer1_pass': False,
            'layer2_pass': False,
            'layer3_pass': False,
            'layer4_pass': False,
            'final_action': 'wait',
            'blocking_reason': None,
            'confidence_boost': 0,
            'tp_multiplier': 1.0,
            'sl_multiplier': 1.0,
            'adx': adx_value,
            'funding_rate': funding_rate,
            'regime': regime_result.get('regime', 'unknown')
        }

        df_1h = context.processed_dfs['1h']
        if len(df_1h) >= 20:
            close_1h = df_1h['close'].iloc[-1]
            ema20_1h = df_1h['ema_20'].iloc[-1] if 'ema_20' in df_1h.columns else close_1h
            ema60_1h = df_1h['ema_60'].iloc[-1] if 'ema_60' in df_1h.columns else close_1h
            four_layer_result['close_1h'] = close_1h
            four_layer_result['ema20_1h'] = ema20_1h
            four_layer_result['ema60_1h'] = ema60_1h
        else:
            close_1h = context.current_price
            ema20_1h = context.current_price
            ema60_1h = context.current_price
            four_layer_result['close_1h'] = close_1h
            four_layer_result['ema20_1h'] = ema20_1h
            four_layer_result['ema60_1h'] = ema60_1h

        oi_change = oi_fuel.get('oi_change_24h', 0)
        if oi_change is None:
            oi_change = 0
        four_layer_result['oi_change'] = oi_change

        data_anomalies = []
        if abs(oi_change) > 200:
            data_anomalies.append(f"OI_ANOMALY: {oi_change:.1f}% (>200% likely data error)")
            log.warning(f"⚠️ DATA ANOMALY: OI Change {oi_change:.1f}% is abnormally high")
            oi_change = max(min(oi_change, 100), -100)
            four_layer_result['oi_change'] = oi_change
            four_layer_result['oi_change_raw'] = oi_fuel.get('oi_change_24h', 0)

        if adx_value < 5:
            data_anomalies.append(f"ADX_ANOMALY: {adx_value:.0f} (<5 may be unreliable)")
            log.warning(f"⚠️ DATA ANOMALY: ADX {adx_value:.0f} is abnormally low")

        if abs(funding_rate) > 1.0:
            data_anomalies.append(f"FUNDING_ANOMALY: {funding_rate:.3f}% (extreme)")
            log.warning(f"⚠️ DATA ANOMALY: Funding Rate {funding_rate:.3f}% is extreme")

        regime = regime_result.get('regime', 'unknown')
        if adx_value < 15 and regime in ['trending_up', 'trending_down']:
            data_anomalies.append(f"LOGIC_PARADOX: ADX={adx_value:.0f} (no trend) + Regime={regime} (trending)")
            log.warning(f"⚠️ LOGIC PARADOX: ADX={adx_value:.0f} indicates NO trend, but Regime={regime}. Forcing to choppy.")
            four_layer_result['regime'] = 'choppy'
            four_layer_result['regime_override'] = True

        four_layer_result['data_anomalies'] = data_anomalies if data_anomalies else None

        if len(df_1h) < 60:
            log.warning(f"⚠️ Insufficient 1h data: {len(df_1h)} bars (need 60+)")
            four_layer_result['blocking_reason'] = 'Insufficient 1h data'
            trend_1h = 'neutral'
        else:
            if ema20_1h > ema60_1h:
                trend_1h = 'long'
                if close_1h > ema20_1h:
                    four_layer_result['trend_confirmation'] = 'strong'
                else:
                    four_layer_result['trend_confirmation'] = 'moderate'
            elif ema20_1h < ema60_1h:
                trend_1h = 'short'
                if close_1h < ema20_1h:
                    four_layer_result['trend_confirmation'] = 'strong'
                else:
                    four_layer_result['trend_confirmation'] = 'moderate'
            else:
                trend_1h = 'neutral'

            log.info(f"📊 1h EMA: Close=${close_1h:.2f}, EMA20=${ema20_1h:.2f}, EMA60=${ema60_1h:.2f} => {trend_1h.upper()}")

        oi_divergence_warning = None
        oi_divergence_warn = 15.0
        oi_divergence_block = 60.0

        if trend_1h == 'neutral':
            four_layer_result['blocking_reason'] = 'No clear 1h trend (EMA 20/60)'
            log.info("❌ Layer 1 FAIL: No clear trend")
        elif adx_value < 15:
            four_layer_result['blocking_reason'] = f"Weak Trend Strength (ADX {adx_value:.0f} < 15)"
            log.info(f"❌ Layer 1 FAIL: ADX={adx_value:.0f} < 15, trend not strong enough")
        elif trend_1h == 'long' and oi_change < -oi_divergence_block:
            four_layer_result['blocking_reason'] = f"OI Divergence: Trend UP but OI {oi_change:.1f}%"
            log.warning(f"🚨 Layer 1 FAIL: OI Divergence - Price up but OI {oi_change:.1f}%")
        elif trend_1h == 'short' and oi_change > oi_divergence_block:
            four_layer_result['blocking_reason'] = f"OI Divergence: Trend DOWN but OI +{oi_change:.1f}%"
            log.warning(f"🚨 Layer 1 FAIL: OI Divergence - Price down but OI +{oi_change:.1f}%")
        elif trend_1h == 'long' and oi_change < -oi_divergence_warn:
            oi_divergence_warning = f"OI Divergence: Trend UP but OI {oi_change:.1f}%"
            log.warning(f"⚠️ Layer 1 WARNING: OI Divergence - Price up but OI {oi_change:.1f}%")
        elif trend_1h == 'short' and oi_change > oi_divergence_warn:
            oi_divergence_warning = f"OI Divergence: Trend DOWN but OI +{oi_change:.1f}%"
            log.warning(f"⚠️ Layer 1 WARNING: OI Divergence - Price down but OI +{oi_change:.1f}%")
        elif trend_1h == 'long' and oi_fuel.get('whale_trap_risk', False):
            four_layer_result['blocking_reason'] = f"Whale trap detected (OI {oi_change:.1f}%)"
            log.warning("🐋 Layer 1 FAIL: Whale exit trap")

        if not four_layer_result.get('blocking_reason'):
            four_layer_result['layer1_pass'] = True

            if abs(oi_change) < 1.0:
                four_layer_result['fuel_warning'] = f"Weak Fuel (OI {oi_change:.1f}%)"
                four_layer_result['confidence_penalty'] = -10
                log.warning(f"⚠️ Layer 1 WARNING: Weak fuel - OI {oi_change:.1f}% (proceed with caution)")
                fuel_strength = 'Weak'
            else:
                fuel_strength = 'Strong' if abs(oi_change) > 3.0 else 'Moderate'
            if oi_divergence_warning:
                existing_warning = four_layer_result.get('fuel_warning')
                if existing_warning:
                    four_layer_result['fuel_warning'] = f"{existing_warning} | {oi_divergence_warning}"
                else:
                    four_layer_result['fuel_warning'] = oi_divergence_warning
                current_penalty = four_layer_result.get('confidence_penalty', 0)
                four_layer_result['confidence_penalty'] = min(current_penalty, -10) if current_penalty else -10
            log.info(f"✅ Layer 1 PASS: {trend_1h.upper()} trend + {fuel_strength} Fuel (OI {oi_change:+.1f}%)")

            if self.agent_config.ai_prediction_filter_agent and self.agent_config.predict_agent:
                ai_filter = AIPredictionFilter()
                ai_check = ai_filter.check_divergence(trend_1h, context.predict_result)

                four_layer_result['ai_check'] = ai_check
                if adx_value < 5:
                    ai_check['ai_invalidated'] = True
                    ai_check['original_signal'] = ai_check.get('ai_signal', 'unknown')
                    ai_check['ai_signal'] = 'INVALID (ADX<5)'
                    four_layer_result['ai_prediction_note'] = (
                        f"AI prediction invalidated: ADX={adx_value:.0f} (<5), "
                        "directional signals are statistically meaningless"
                    )
                    log.warning(f"⚠️ AI prediction invalidated: ADX={adx_value:.0f} is too low for any directional signal to be reliable")

                if ai_check['ai_veto']:
                    four_layer_result['blocking_reason'] = ai_check['reason']
                    log.warning(f"🚫 Layer 2 VETO: {ai_check['reason']}")
                else:
                    four_layer_result['layer2_pass'] = True
                    four_layer_result['confidence_boost'] = ai_check['confidence_boost']
                    log.info(f"✅ Layer 2 PASS: AI {ai_check['ai_signal']} (boost: {ai_check['confidence_boost']:+d}%)")
            else:
                four_layer_result['layer2_pass'] = True
                four_layer_result['ai_check'] = {
                    'allow_trade': True,
                    'reason': 'AIPredictionFilter disabled',
                    'confidence_boost': 0,
                    'ai_veto': False,
                    'ai_signal': 'disabled',
                    'ai_confidence': 0
                }
                log.info("⏭️ Layer 2 SKIP: AIPredictionFilterAgent disabled")

            if four_layer_result['layer2_pass']:
                df_15m = context.processed_dfs['15m']
                if len(df_15m) < 20:
                    log.warning(f"⚠️ Insufficient 15m data: {len(df_15m)} bars")
                    four_layer_result['blocking_reason'] = 'Insufficient 15m data'
                    setup_ready = False
                else:
                    close_15m = df_15m['close'].iloc[-1]
                    bb_middle = df_15m['bb_middle'].iloc[-1]
                    bb_upper = df_15m['bb_upper'].iloc[-1]
                    bb_lower = df_15m['bb_lower'].iloc[-1]
                    kdj_j = df_15m['kdj_j'].iloc[-1]
                    kdj_k = df_15m['kdj_k'].iloc[-1]

                    log.info(f"📊 15m Setup: Close=${close_15m:.2f}, BB[{bb_lower:.2f}/{bb_middle:.2f}/{bb_upper:.2f}], KDJ_J={kdj_j:.1f}")
                    four_layer_result['setup_note'] = f"KDJ_J={kdj_j:.0f}, Close={'>' if close_15m > bb_middle else '<'}BB_mid"
                    four_layer_result['kdj_j'] = kdj_j
                    four_layer_result['bb_position'] = 'upper' if close_15m > bb_upper else 'lower' if close_15m < bb_lower else 'middle'

                    if kdj_j > 80 or close_15m > bb_upper:
                        four_layer_result['kdj_zone'] = 'overbought'
                    elif kdj_j < 20 or close_15m < bb_lower:
                        four_layer_result['kdj_zone'] = 'oversold'
                    else:
                        four_layer_result['kdj_zone'] = 'neutral'

                    if trend_1h == 'long':
                        if close_15m > bb_upper or kdj_j > 80:
                            setup_ready = False
                            four_layer_result['blocking_reason'] = f"15m overbought (J={kdj_j:.0f}) - wait for pullback"
                            log.info("⏳ Layer 3 WAIT: Overbought - waiting for pullback")
                        elif close_15m < bb_middle or kdj_j < 50:
                            setup_ready = True
                            four_layer_result['setup_quality'] = 'IDEAL'
                            log.info(f"✅ Layer 3 READY: IDEAL PULLBACK - J={kdj_j:.0f} < 50 or Close < BB_middle")
                        else:
                            setup_ready = True
                            four_layer_result['setup_quality'] = 'ACCEPTABLE'
                            log.info(f"✅ Layer 3 READY: Acceptable mid-range entry (J={kdj_j:.0f})")
                    elif trend_1h == 'short':
                        if close_15m < bb_lower or kdj_j < 20:
                            setup_ready = False
                            four_layer_result['blocking_reason'] = f"15m oversold (J={kdj_j:.0f}) - wait for rally"
                            log.info("⏳ Layer 3 WAIT: Oversold - waiting for rally")
                        elif close_15m > bb_middle or kdj_j > 50:
                            setup_ready = True
                            four_layer_result['setup_quality'] = 'IDEAL'
                            log.info(f"✅ Layer 3 READY: IDEAL RALLY - J={kdj_j:.0f} > 60 or Close > BB_middle")
                        else:
                            setup_ready = True
                            four_layer_result['setup_quality'] = 'ACCEPTABLE'
                            log.info(f"✅ Layer 3 READY: Acceptable mid-range entry (J={kdj_j:.0f})")
                    else:
                        setup_ready = False

                if not setup_ready:
                    four_layer_result['blocking_reason'] = "15m setup not ready"
                    log.info("⏳ Layer 3 WAIT: 15m setup not ready")
                else:
                    four_layer_result['layer3_pass'] = True
                    log.info("✅ Layer 3 PASS: 15m setup ready")

                    if self.agent_config.trigger_detector_agent:
                        df_5m = context.processed_dfs['5m']
                        trigger_result = self.agent_provider.trigger_detector_agent.detect_trigger(df_5m, direction=trend_1h)
                        four_layer_result['trigger_pattern'] = trigger_result.get('pattern_type') or 'None'
                        rvol = trigger_result.get('rvol', 1.0)
                        four_layer_result['trigger_rvol'] = rvol

                        if rvol < 0.5:
                            log.warning(f"⚠️ Low Volume Warning (RVOL {rvol:.1f}x < 0.5) - Trend validation may be unreliable")
                            if not four_layer_result.get('data_anomalies'):
                                four_layer_result['data_anomalies'] = []
                            four_layer_result['data_anomalies'].append(f"Low Volume (RVOL {rvol:.1f}x)")

                        if not trigger_result['triggered']:
                            four_layer_result['blocking_reason'] = f"5min trigger not confirmed (RVOL={trigger_result.get('rvol', 1.0):.1f}x)"
                            log.info(f"⏳ Layer 4 WAIT: No engulfing or breakout pattern (RVOL={trigger_result.get('rvol', 1.0):.1f}x)")
                        else:
                            log.info(f"🎯 Layer 4 TRIGGER: {trigger_result['pattern_type']} detected")
                            sentiment_score = sentiment.get('total_sentiment_score', 0)

                            if sentiment_score > 80:
                                four_layer_result['tp_multiplier'] = 0.5
                                four_layer_result['sl_multiplier'] = 1.0
                                log.warning(f"🔴 Extreme Greed ({sentiment_score:.0f}): TP target halved")
                            elif sentiment_score < -80:
                                four_layer_result['tp_multiplier'] = 1.5
                                four_layer_result['sl_multiplier'] = 0.8
                                log.info(f"🟢 Extreme Fear ({sentiment_score:.0f}): Be greedy when others are fearful")
                            else:
                                four_layer_result['tp_multiplier'] = 1.0
                                four_layer_result['sl_multiplier'] = 1.0

                            if trend_1h == 'long' and funding_rate > 0.05:
                                four_layer_result['tp_multiplier'] *= 0.7
                                log.warning(f"💰 High Funding Rate ({funding_rate:.3f}%): Longs crowded, TP reduced")
                            elif trend_1h == 'short' and funding_rate < -0.05:
                                four_layer_result['tp_multiplier'] *= 0.7
                                log.warning(f"💰 Negative Funding Rate ({funding_rate:.3f}%): Shorts crowded, TP reduced")

                            four_layer_result['layer4_pass'] = True
                            four_layer_result['final_action'] = trend_1h
                            four_layer_result['trigger_pattern'] = trigger_result['pattern_type']
                            log.info(f"✅ Layer 4 PASS: Sentiment {sentiment_score:.0f}, Trigger={trigger_result['pattern_type']}")
                            log.info(f"🎯 ALL LAYERS PASSED: {trend_1h.upper()} with {70 + four_layer_result['confidence_boost']}% confidence")
                    else:
                        four_layer_result['trigger_pattern'] = 'disabled'
                        four_layer_result['trigger_rvol'] = None
                        four_layer_result['layer4_pass'] = True
                        four_layer_result['final_action'] = trend_1h
                        log.info("⏭️ Layer 4 SKIP: TriggerDetectorAgent disabled")

        global_state.four_layer_result = four_layer_result
        emit_global_runtime_event(
            context,
            stream="lifecycle",
            agent="four_layer_filter",
            phase="end",
            data={
                "final_action": four_layer_result.get('final_action', 'wait'),
                "layer4_pass": bool(four_layer_result.get('layer4_pass'))
            }
        )
        return regime_result, four_layer_result, trend_1h