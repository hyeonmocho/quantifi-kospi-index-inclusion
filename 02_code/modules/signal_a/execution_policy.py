"""User-specified current entry policy; immutable historical comparison retained."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionPolicy:
    name: str
    daily_adv_fraction: float
    max_entry_sessions: int
    adv_sessions: int = 20

    def __post_init__(self):
        if not 0 < self.daily_adv_fraction <= 1:
            raise ValueError('daily_adv_fraction must be in (0, 1]')
        if self.max_entry_sessions < 1 or self.adv_sessions != 20:
            raise ValueError('Positive entry window and exactly 20 ADV sessions required')


CURRENT_POLICY = ExecutionPolicy('current_10pct_10d', .10, 10)
PREVIOUS_POLICY = ExecutionPolicy('previous_5pct_1d', .05, 1)
CAPITALS_KRW = (1e8, 1e9, 1e10)
