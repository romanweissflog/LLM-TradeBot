from dataclasses import dataclass
from typing import Optional

@dataclass
class CycleContext:
    """Per-cycle immutable context used across trading stages."""
    run_id: str
    cycle_id: Optional[str]
    snapshot_id: str
    cycle_num: int
    symbol: str