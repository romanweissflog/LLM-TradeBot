"""
多Agent架构主循环 (Multi-Agent Trading Loop)
===========================================

集成:
1. 🕵️ DataSyncAgent - 异步并发数据采集
2. 👨‍🔬 QuantAnalystAgent - 量化信号分析
3. ⚖️ DecisionCoreAgent - 加权投票决策
4. 👮 RiskAuditAgent - 风控审计拦截

优化:
- 异步并发执行（减少60%等待时间）
- 双视图数据结构（stable + live）
- 分层信号分析（趋势 + 震荡）
- 多周期对齐决策
- 止损方向自动修正
- 一票否决风控

Author: AI Trader Team
Date: 2025-12-19
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from typing import Dict, Optional
from datetime import datetime
import json
import time

from src.api.binance_client import BinanceClient
from src.data.processor import MarketDataProcessor  # ✅ Import Processor
from src.execution.engine import ExecutionEngine
from src.risk.manager import RiskManager
from src.config import Config
from src.utils.logger import log
from src.utils.trade_logger import trade_logger
from src.utils.data_saver import DataSaver
from dataclasses import asdict

# 导入多Agent
from src.agents import (
    DataSyncAgent,
    QuantAnalystAgent,
    DecisionCoreAgent,
    RiskAuditAgent,
    PositionInfo,
    SignalWeight
)

class MultiAgentTradingBot:
    """
    多Agent交易机器人（重构版）
    
    工作流程:
    1. DataSyncAgent: 异步采集5m/15m/1h数据
    2. QuantAnalystAgent: 生成量化信号（趋势+震荡）
    3. DecisionCoreAgent: 加权投票决策
    4. RiskAuditAgent: 风控审计拦截
    5. ExecutionEngine: 执行交易
    """
    
    def __init__(
        self,
        max_position_size: float = 100.0,
        leverage: int = 1,
        stop_loss_pct: float = 1.0,
        take_profit_pct: float = 2.0,
        test_mode: bool = False
    ):
        """
        初始化多Agent交易机器人
        
        Args:
            max_position_size: 最大单笔金额（USDT）
            leverage: 杠杆倍数
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
            test_mode: 测试模式（不执行真实交易）
        """
        print("\n" + "="*80)
        print("🤖 AI Trader - 多Agent架构版本")
        print("="*80)
        
        self.config = Config()
        self.symbol = self.config.get('trading.symbol', 'BTCUSDT')
        self.test_mode = test_mode
        
        # 交易参数
        self.max_position_size = max_position_size
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # 初始化客户端
        self.client = BinanceClient()
        self.risk_manager = RiskManager()
        self.execution_engine = ExecutionEngine(self.client, self.risk_manager)
        self.saver = DataSaver()  # ✅ 初始化数据保存器
        self.processor = MarketDataProcessor() # ✅ 初始化数据处理器 (for 7-step pipeline support)
        
        # 初始化4大Agent
        print("\n🚀 初始化Agent...")
        self.data_sync_agent = DataSyncAgent(self.client)
        self.quant_analyst = QuantAnalystAgent()
        self.decision_core = DecisionCoreAgent()
        self.risk_audit = RiskAuditAgent(
            max_leverage=10.0,
            max_position_pct=0.3,
            min_stop_loss_pct=0.005,
            max_stop_loss_pct=0.05
        )
        
        print("  ✅ DataSyncAgent 已就绪")
        print("  ✅ QuantAnalystAgent 已就绪")
        print("  ✅ DecisionCoreAgent 已就绪")
        print("  ✅ RiskAuditAgent 已就绪")
        
        print(f"\n⚙️  交易配置:")
        print(f"  - 交易对: {self.symbol}")
        print(f"  - 最大单笔: ${self.max_position_size:.2f} USDT")
        print(f"  - 杠杆倍数: {self.leverage}x")
        print(f"  - 止损: {self.stop_loss_pct}%")
        print(f"  - 止盈: {self.take_profit_pct}%")
        print(f"  - 测试模式: {'✅ 是' if self.test_mode else '❌ 否'}")
    
    async def run_trading_cycle(self) -> Dict:
        """
        执行完整的交易循环（异步版本）
        
        Returns:
            {
                'status': 'success/failed/hold/blocked',
                'action': 'long/short/hold',
                'details': {...}
            }
        """
        print(f"\n{'='*80}")
        print(f"🔄 交易循环 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        try:
            # Step 1: 异步数据采集 (Archived by DataSyncAgent or explicitly here)
            print("\n[Step 1/5] 🕵️ DataSyncAgent - 异步数据采集...")
            market_snapshot = await self.data_sync_agent.fetch_all_timeframes(self.symbol)
            
            # ✅ Save Step 1: Raw Data (Already updated before, ensuring it stays)
            # Note: We can rely on DataSyncAgent but for explicit pipeline control we save here
            self.saver.save_step1_klines(market_snapshot.raw_5m, self.symbol, '5m')
            self.saver.save_step1_klines(market_snapshot.raw_15m, self.symbol, '15m')
            self.saver.save_step1_klines(market_snapshot.raw_1h, self.symbol, '1h')
            
            # 🔴 Step 2: Data Processing (Indicator Calculation)
            # This integrates the "7-Step Pipeline" Step 2 into the Multi-Agent flow
            print("\n[Step 2/5] ⚙️ MarketDataProcessor - 计算技术指标 (Pipeline Step 2)...")
            
            # Process and Archive Step 2 (Indicators)
            df_5m = self.processor.process_klines(market_snapshot.raw_5m, self.symbol, '5m')
            df_15m = self.processor.process_klines(market_snapshot.raw_15m, self.symbol, '15m')
            df_1h = self.processor.process_klines(market_snapshot.raw_1h, self.symbol, '1h')
            
            # Update snapshot with processed data (so Agents use standard indicators)
            market_snapshot.stable_5m = df_5m
            market_snapshot.stable_15m = df_15m
            market_snapshot.stable_1h = df_1h
            
            current_price = market_snapshot.live_5m.get('close')
            print(f"  ✅ 当前价格: ${current_price:,.2f}")
            print(f"  ✅ 数据时间: {market_snapshot.timestamp}")
            
            # Step 3 (was 2): 量化分析
            print("\n[Step 3/5] 👨‍🔬 QuantAnalystAgent - 量化分析...")
            quant_analysis = await self.quant_analyst.analyze_all_timeframes(market_snapshot)
            
            # ✅ Save Step 4: Context (Quant Analysis) as we skip Step 2/3 DFs
            snapshot_id = f"multi_{int(time.time())}"
            self.saver.save_step4_context(quant_analysis, self.symbol, 'mixed', snapshot_id)
            
            comprehensive = quant_analysis.get('comprehensive', {})
            print(f"  ✅ 综合信号: {comprehensive.get('signal', 'N/A')}")
            print(f"  ✅ 综合得分: {comprehensive.get('score', 0)}")
            print(f"  ✅ 趋势强度: {comprehensive.get('details', {}).get('trend_strength', 'N/A')}")
            
            # Step 3: 决策中枢
            print("\n[Step 3/5] ⚖️ DecisionCoreAgent - 加权投票决策...")
            vote_result = await self.decision_core.make_decision(quant_analysis)
            
            # ✅ Save Step 5: LLM Context (Generated but maybe not used by LLM yet)
            llm_ctx = self.decision_core.to_llm_context(vote_result, quant_analysis)
            self.saver.save_step5_markdown(llm_ctx, self.symbol, 'mixed', snapshot_id)
            
            # ✅ Save Step 6: Decision
            self.saver.save_step6_decision(asdict(vote_result), self.symbol, 'mixed', snapshot_id)
            
            print(f"  ✅ 决策动作: {vote_result.action}")
            print(f"  ✅ 置信度: {vote_result.confidence:.2%}")
            print(f"  ✅ 加权得分: {vote_result.weighted_score:.1f}")
            print(f"  ✅ 周期对齐: {'是' if vote_result.multi_period_aligned else '否'}")
            print(f"  ✅ 决策原因: {vote_result.reason}")
            
            # 如果是观望，直接返回
            if vote_result.action == 'hold':
                print("\n✅ 决策: 观望")
                return {
                    'status': 'hold',
                    'action': 'hold',
                    'details': {
                        'reason': vote_result.reason,
                        'confidence': vote_result.confidence
                    }
                }
            
            # Step 4: 构建订单
            print(f"\n[Step 4/5] 📝 构建订单参数...")
            order_params = self._build_order_params(
                action=vote_result.action,
                current_price=current_price,
                confidence=vote_result.confidence
            )
            
            print(f"  ✅ 动作: {order_params['action']}")
            print(f"  ✅ 入场价: ${order_params['entry_price']:,.2f}")
            print(f"  ✅ 止损价: ${order_params['stop_loss']:,.2f}")
            print(f"  ✅ 止盈价: ${order_params['take_profit']:,.2f}")
            print(f"  ✅ 数量: {order_params['quantity']:.4f} {self.symbol.replace('USDT', '')}")
            print(f"  ✅ 杠杆: {order_params['leverage']}x")
            
            # Step 5: 风控审计
            print(f"\n[Step 5/5] 👮 RiskAuditAgent - 风控审计...")
            
            # 获取账户信息
            account_balance = self._get_account_balance()
            current_position = self._get_current_position()
            
            # 执行审计
            audit_result = await self.risk_audit.audit_decision(
                decision=order_params,
                current_position=current_position,
                account_balance=account_balance,
                current_price=current_price
            )
            
            print(f"  ✅ 审计结果: {'✅ 通过' if audit_result.passed else '❌ 拦截'}")
            print(f"  ✅ 风险等级: {audit_result.risk_level.value}")
            
            # 如果有修正
            if audit_result.corrections:
                print(f"  ⚠️  自动修正:")
                for key, value in audit_result.corrections.items():
                    print(f"     {key}: {order_params[key]} -> {value}")
                    order_params[key] = value  # 应用修正
            
            # 如果有警告
            if audit_result.warnings:
                print(f"  ⚠️  警告信息:")
                for warning in audit_result.warnings:
                    print(f"     {warning}")
            
            # 如果被拦截
            if not audit_result.passed:
                print(f"\n❌ 决策被风控拦截: {audit_result.blocked_reason}")
                return {
                    'status': 'blocked',
                    'action': vote_result.action,
                    'details': {
                        'reason': audit_result.blocked_reason,
                        'risk_level': audit_result.risk_level.value
                    }
                }
            
            # Step 6: 执行交易
            print(f"\n[Step 6/6] 🎯 执行交易...")
            
            if self.test_mode:
                print("  ⚠️  测试模式: 不执行真实交易")
                return {
                    'status': 'test',
                    'action': vote_result.action,
                    'details': order_params
                }
            
            # 真实执行
            executed = self._execute_order(order_params)
            
            # ✅ Save Step 7: Execution
            self.saver.save_step7_execution({
                'success': executed,
                'params': order_params,
                'timestamp': datetime.now().isoformat()
            }, self.symbol, 'mixed')
            
            if executed:
                print("  ✅ 订单执行成功!")
                
                # 记录交易日志
                trade_logger.log_trade(
                    symbol=self.symbol,
                    action=order_params['action'],
                    entry_price=order_params['entry_price'],
                    quantity=order_params['quantity'],
                    stop_loss=order_params['stop_loss'],
                    take_profit=order_params['take_profit'],
                    leverage=order_params['leverage'],
                    reason=vote_result.reason
                )
                
                return {
                    'status': 'success',
                    'action': vote_result.action,
                    'details': order_params
                }
            else:
                print("  ❌ 订单执行失败")
                return {
                    'status': 'failed',
                    'action': vote_result.action,
                    'details': {'error': 'execution_failed'}
                }
        
        except Exception as e:
            log.error(f"交易循环异常: {e}", exc_info=True)
            return {
                'status': 'error',
                'details': {'error': str(e)}
            }
        
        except Exception as e:
            log.error(f"交易循环异常: {e}", exc_info=True)
            return {
                'status': 'error',
                'details': {'error': str(e)}
            }
    
    def _build_order_params(
        self, 
        action: str, 
        current_price: float,
        confidence: float
    ) -> Dict:
        """
        构建订单参数
        
        Args:
            action: 'long' or 'short'
            current_price: 当前价格
            confidence: 决策置信度
        
        Returns:
            订单参数字典
        """
        # 计算仓位大小（根据置信度调整）
        position_multiplier = min(confidence * 1.2, 1.0)  # 最高100%仓位
        adjusted_position = self.max_position_size * position_multiplier
        
        # 计算数量
        quantity = adjusted_position / current_price
        
        # 计算止损止盈
        if action == 'long':
            stop_loss = current_price * (1 - self.stop_loss_pct / 100)
            take_profit = current_price * (1 + self.take_profit_pct / 100)
        else:  # short
            stop_loss = current_price * (1 + self.stop_loss_pct / 100)
            take_profit = current_price * (1 - self.take_profit_pct / 100)
        
        return {
            'action': action,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'quantity': quantity,
            'leverage': self.leverage,
            'confidence': confidence
        }
    
    def _get_account_balance(self) -> float:
        """获取账户可用余额"""
        try:
            balance_info = self.client.get_futures_balance()
            usdt_balance = next(
                (b for b in balance_info if b['asset'] == 'USDT'),
                None
            )
            if usdt_balance:
                return float(usdt_balance['availableBalance'])
            return 0.0
        except Exception as e:
            log.error(f"获取余额失败: {e}")
            return 0.0
    
    def _get_current_position(self) -> Optional[PositionInfo]:
        """获取当前持仓"""
        try:
            positions = self.client.get_futures_positions()
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    amt = float(pos['positionAmt'])
                    if abs(amt) > 0:
                        return PositionInfo(
                            symbol=self.symbol,
                            side='long' if amt > 0 else 'short',
                            entry_price=float(pos['entryPrice']),
                            quantity=abs(amt),
                            unrealized_pnl=float(pos['unRealizedProfit'])
                        )
            return None
        except Exception as e:
            log.error(f"获取持仓失败: {e}")
            return None
    
    def _execute_order(self, order_params: Dict) -> bool:
        """
        执行订单
        
        Args:
            order_params: 订单参数
        
        Returns:
            是否成功
        """
        try:
            # 设置杠杆
            self.client.set_leverage(
                symbol=self.symbol,
                leverage=order_params['leverage']
            )
            
            # 市价开仓
            side = 'BUY' if order_params['action'] == 'long' else 'SELL'
            order = self.client.place_futures_market_order(
                symbol=self.symbol,
                side=side,
                quantity=order_params['quantity']
            )
            
            if not order:
                return False
            
            # 设置止损止盈
            self.execution_engine.set_stop_loss_take_profit(
                symbol=self.symbol,
                position_side='LONG' if order_params['action'] == 'long' else 'SHORT',
                stop_loss=order_params['stop_loss'],
                take_profit=order_params['take_profit']
            )
            
            return True
            
        except Exception as e:
            log.error(f"订单执行失败: {e}", exc_info=True)
            return False
    
    def run_once(self) -> Dict:
        """运行一次交易循环（同步包装）"""
        return asyncio.run(self.run_trading_cycle())
    
    def run_continuous(self, interval_minutes: int = 5):
        """
        持续运行交易机器人
        
        Args:
            interval_minutes: 检查间隔（分钟）
        """
        print(f"\n🔄 开始持续运行模式，间隔 {interval_minutes} 分钟...")
        
        try:
            while True:
                result = self.run_once()
                
                print(f"\n循环结果: {result['status']}")
                
                # 等待下一次检查
                print(f"\n⏳ 等待 {interval_minutes} 分钟...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print(f"\n\n⚠️  收到停止信号，退出...")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'decision_core': self.decision_core.get_statistics(),
            'risk_audit': self.risk_audit.get_audit_report(),
        }


# ============================================
# 主入口
# ============================================
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多Agent交易机器人')
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--max-position', type=float, default=100.0, help='最大单笔金额')
    parser.add_argument('--leverage', type=int, default=1, help='杠杆倍数')
    parser.add_argument('--stop-loss', type=float, default=1.0, help='止损百分比')
    parser.add_argument('--take-profit', type=float, default=2.0, help='止盈百分比')
    parser.add_argument('--mode', choices=['once', 'continuous'], default='once', help='运行模式')
    parser.add_argument('--interval', type=int, default=5, help='持续运行间隔（分钟）')
    
    args = parser.parse_args()
    
    # 创建机器人
    bot = MultiAgentTradingBot(
        max_position_size=args.max_position,
        leverage=args.leverage,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        test_mode=args.test
    )
    
    # 运行
    if args.mode == 'once':
        result = bot.run_once()
        print(f"\n最终结果: {json.dumps(result, indent=2)}")
        
        # 显示统计
        stats = bot.get_statistics()
        print(f"\n统计信息:")
        print(json.dumps(stats, indent=2))
    else:
        bot.run_continuous(interval_minutes=args.interval)


if __name__ == '__main__':
    main()
