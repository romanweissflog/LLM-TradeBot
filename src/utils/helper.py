
from typing import Optional

from src.agents.risk_audit_agent import PositionInfo

from src.api.binance_client import BinanceClient

from src.utils.logger import log
from src.server.state import global_state

def get_current_position(
    client: BinanceClient,
    symbol: str,
    test_mode: bool,
) -> Optional[PositionInfo]:
    """获取当前持仓 (支持实盘 + Test Mode)"""
    try:
        # 1. Test Mode Support
        if test_mode:
            if symbol in global_state.virtual_positions:
                v_pos = global_state.virtual_positions[symbol]
                return PositionInfo(
                    symbol=symbol,
                    side=v_pos['side'].lower(), # ensure lowercase 'long'/'short'
                    entry_price=v_pos['entry_price'],
                    quantity=v_pos['quantity'],
                    unrealized_pnl=v_pos.get('unrealized_pnl', 0)
                )
            return None

        # 2. Live Mode Support
        pos = client.get_futures_position(symbol)
        if pos and abs(pos['position_amt']) > 0:
            return PositionInfo(
                symbol=symbol,
                side='long' if pos['position_amt'] > 0 else 'short',
                entry_price=pos['entry_price'],
                quantity=abs(pos['position_amt']),
                unrealized_pnl=pos['unrealized_profit']
            )
        return None
    except Exception as e:
        log.error(f"Failed to get positions: {e}")
        return None