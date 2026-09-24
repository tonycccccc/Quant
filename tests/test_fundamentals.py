"""Offline unit tests for the fundamentals valuation paths."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'Code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ml import fundamentals


def test_segment_sum_parts_bridges_debt_and_cash_once():
    row = pd.Series({
        'ticker': 'TEST', 'current_price': 100.0, 'shares_outstanding': 10.0,
        'total_revenue': 1000.0, 'total_debt': 100.0, 'total_cash': 50.0,
        'segments': json.dumps([
            {'name': 'chips', 'revenue': 600, 'revenue_growth': .12,
             'fcff': 120, 'wacc': .10, 'terminal_growth': .025},
            {'name': 'software', 'revenue': 400, 'revenue_growth': .08,
             'ebit': 100, 'tax_rate': .20, 'da': 10, 'capex': 20,
             'change_working_capital': 5, 'wacc': .09, 'terminal_growth': .025},
        ]),
    })

    result = fundamentals.compute_fair_value(row)

    assert result['method'] == 'sum_of_parts'
    assert result['segments_valued'] == 2
    assert result['segment_revenue_coverage'] == pytest.approx(1.0)
    assert result['equity_value'] == pytest.approx(result['enterprise_value'] - 50)
    assert result['fair_value'] == pytest.approx(result['equity_value'] / 10)
    assert result['buy_price'] == pytest.approx(result['fair_value'] * .75)


def test_incomplete_segment_does_not_emit_a_stock_price():
    row = pd.Series({
        'ticker': 'TEST', 'current_price': 100, 'shares_outstanding': 10,
        'segments': [{'name': 'software', 'revenue': 100, 'revenue_growth': .1}],
    })

    result = fundamentals.compute_fair_value(row)

    assert 'fair_value' not in result
    assert result['data_quality'] == 'incomplete segment inputs: software'


def test_invalid_discount_rate_is_rejected():
    row = pd.Series({'ticker': 'TEST', 'current_price': 100})
    with pytest.raises(ValueError):
        fundamentals.compute_fair_value(row, margin_of_safety=1.0)


def test_quarterly_cash_flow_is_summed_over_latest_four_quarters():
    from types import SimpleNamespace

    dates = pd.to_datetime(['2026-06-30', '2026-03-31', '2025-12-31',
                            '2025-09-30', '2025-06-30'])
    frame = pd.DataFrame([[20, 15, 10, 25, 30]],
                         index=['Free Cash Flow'], columns=dates)

    assert fundamentals._quarterly_fcf_ttm(SimpleNamespace(quarterly_cashflow=frame)) == 70


def test_microsoft_without_segments_emits_no_price():
    row = pd.Series({'ticker': 'MSFT', 'industry': 'Software - Infrastructure',
                     'current_price': 500, 'forward_pe': 30, 'revenue_growth': .15})

    result = fundamentals.compute_fair_value(row)

    assert result['method'] == 'sum_of_parts'
    assert 'fair_value' not in result
    assert 'Intelligent Cloud' in result['data_quality']


def test_microsoft_with_segments_uses_sum_of_parts():
    row = pd.Series({
        'ticker': 'MSFT', 'current_price': 100.0, 'shares_outstanding': 10.0,
        'total_revenue': 300.0,
        'segments': json.dumps([
            {'name': 'pbp', 'revenue': 120, 'revenue_growth': .10, 'fcff': 40},
            {'name': 'cloud', 'revenue': 110, 'revenue_growth': .20, 'fcff': 25},
            {'name': 'mpc', 'revenue': 70, 'revenue_growth': .03, 'fcff': 10},
        ]),
    })

    result = fundamentals.compute_fair_value(row)

    assert result['method'] == 'sum_of_parts'
    assert result['segments_valued'] == 3
