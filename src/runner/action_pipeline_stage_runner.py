from typing import Dict, Optional, Any, TYPE_CHECKING

from src.utils.logger import log
from src.server.state import global_state

from src.trading.symbol_manager import SymbolManager
from src.trading.result_builder import ResultBuilder

from src.utils.data_saver import DataSaver

if TYPE_CHECKING:
    from .runner_provider import RunnerProvider

class ActionPipelineStageRunner:
    def __init__(
        self,
        symbol_manager: SymbolManager,
        runner_provider: "RunnerProvider",
        saver: DataSaver
    ):
        self.symbol_manager = symbol_manager
        self.runner_provider = runner_provider
        self.result_builder = ResultBuilder(
            symbol_manager,
            saver
        )
    
    async def run(
        self,
        *,
        run_id: str,
        cycle_id: Optional[str],
        snapshot_id: str,
        analyze_only: bool,
        vote_result: Any,
        quant_analysis: Dict[str, Any],
        predict_result: Any,
        processed_dfs: Dict[str, "pd.DataFrame"],
        current_price: float,
        current_position_info: Optional[Dict[str, Any]],
        regime_result: Optional[Dict[str, Any]],
        market_snapshot: Any
    ) -> Dict[str, Any]:
        """Run risk-audit -> analyze-only gate -> execution as one stage."""
        order_params, audit_result, account_balance, current_position = await self.runner_provider.risk_audit_stage_runner.run(
            run_id=run_id,
            cycle_id=cycle_id,
            vote_result=vote_result,
            quant_analysis=quant_analysis,
            processed_dfs=processed_dfs,
            current_price=current_price,
            current_position_info=current_position_info,
            regime_result=regime_result
        )

        blocked_result = self._finalize_risk_audit_decision(
            vote_result=vote_result,
            quant_analysis=quant_analysis,
            predict_result=predict_result,
            audit_result=audit_result,
            order_params=order_params,
            snapshot_id=snapshot_id,
            cycle_id=cycle_id,
            current_price=current_price
        )
        if blocked_result is not None:
            return blocked_result

        analyze_only_result = self._build_analyze_only_suggestion(
            analyze_only=analyze_only,
            vote_result=vote_result,
            order_params=order_params,
            current_price=current_price
        )
        if analyze_only_result is not None:
            return analyze_only_result

        return await self.runner_provider.execution_stage_runner.run(
            run_id=run_id,
            cycle_id=cycle_id,
            vote_result=vote_result,
            order_params=order_params,
            current_price=current_price,
            current_position_info=current_position_info,
            current_position=current_position,
            account_balance=account_balance,
            market_snapshot=market_snapshot
        )

    
    def _finalize_risk_audit_decision(
        self,
        *,
        vote_result: Any,
        quant_analysis: Dict[str, Any],
        predict_result: Any,
        audit_result: Any,
        order_params: Dict[str, Any],
        snapshot_id: str,
        cycle_id: Optional[str],
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Persist audit outcome, apply corrections, update state, and return blocked result if needed."""
        decision_dict = self.result_builder.build_decision_snapshot(
            vote_result=vote_result,
            quant_analysis=quant_analysis,
            predict_result=predict_result,
            risk_level=audit_result.risk_level.value,
            guardian_passed=audit_result.passed,
            guardian_reason=audit_result.blocked_reason
        )

        self.saver.save_risk_audit(
            audit_result={
                'passed': audit_result.passed,
                'risk_level': audit_result.risk_level.value,
                'blocked_reason': audit_result.blocked_reason,
                'corrections': audit_result.corrections,
                'warnings': audit_result.warnings,
                'order_params': order_params,
                'cycle_id': cycle_id
            },
            symbol=self.symbol_manager.current_symbol,
            snapshot_id=snapshot_id,
            cycle_id=cycle_id
        )

        print(f"  ✅ 审计结果: {'✅ 通过' if audit_result.passed else '❌ 拦截'}")
        print(f"  ✅ 风险等级: {audit_result.risk_level.value}")

        if audit_result.corrections:
            print("  ⚠️  自动修正:")
            for key, value in audit_result.corrections.items():
                print(f"     {key}: {order_params[key]} -> {value}")
                order_params[key] = value

        if audit_result.warnings:
            print("  ⚠️  警告信息:")
            for warning in audit_result.warnings:
                print(f"     {warning}")

        decision_dict['order_params'] = order_params
        global_state.update_decision(decision_dict)

        if not audit_result.passed:
            print(f"\n❌ 决策被风控拦截: {audit_result.blocked_reason}")
            return {
                'status': 'blocked',
                'action': vote_result.action,
                'details': {
                    'reason': audit_result.blocked_reason,
                    'risk_level': audit_result.risk_level.value
                },
                'current_price': current_price
            }

        return None

    def _build_analyze_only_suggestion(
        self,
        *,
        analyze_only: bool,
        vote_result: Any,
        order_params: Dict[str, Any],
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Return suggested result for open actions in analyze_only mode."""
        if not (analyze_only and vote_result.action in ('open_long', 'open_short')):
            return None

        log.info(f"🔍 [Analyze Only] Strategy suggests {vote_result.action.upper()} for {self.symbol_manager.current_symbol}, skipping execution for selector")
        return {
            'status': 'suggested',
            'action': vote_result.action,
            'confidence': vote_result.confidence,
            'order_params': order_params,
            'vote_result': vote_result,
            'current_price': current_price
        }
