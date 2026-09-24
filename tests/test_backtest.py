"""Offline regression tests for backtest bracket exits."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Code'))

from ml.backtest import _simulate_trade


@pytest.mark.parametrize(
    'highs,lows,reason,price,exit_bar',
    [
        ([100, 108, 102, 103], [100, 95, 100, 101], 'sl', 96.5, 1),
        ([100, 102, 108, 103], [100, 99, 95, 101], 'sl', 96.5, 2),
        ([100, 108, 102, 103], [100, 99, 95, 101], 'tp', 107, 1),
        ([100, 102, 108, 103], [100, 95, 99, 101], 'sl', 96.5, 1),
        ([100, 102, 102, 103], [100, 99, 100, 101], 'timeout', 102, 3),
    ],
    ids=['same-bar-first', 'same-bar-later', 'tp-first', 'sl-first', 'no-hit'],
)
def test_bracket_exit(highs, lows, reason, price, exit_bar):
    bars = pd.DataFrame(
        {'close': [100., 100., 101., 102.], 'high': highs, 'low': lows},
        index=pd.date_range('2026-01-01', periods=4, freq='30min'),
    )

    trade = _simulate_trade(
        bars, entry_idx=0, equity=10_000,
        tp_pct=0.07, sl_pct=0.035, timeout_bars=3,
    )

    assert trade.exit_reason == reason
    assert trade.exit_time == bars.index[exit_bar]
    assert trade.exit_price == pytest.approx(price)
    assert trade.pnl_pct == pytest.approx(price / 100 - 1)
    assert trade.shares > 0
    assert trade.pnl == pytest.approx(trade.shares * (price - 100))
