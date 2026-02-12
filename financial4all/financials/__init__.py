# financial4all/financials/__init__.py
"""
Financial statement extraction and standardization from XBRL.

Exposes IncomeStatement, BalanceSheet, CashFlowStatement (built from
FactSet/company facts), and FinancialRatios (derived from those statements).
All statement classes expose to_dataframe() for period-indexed DataFrames
with standardized metric names.
"""

from financial4all.financials.income_statement import IncomeStatement
from financial4all.financials.balance_sheet import BalanceSheet
from financial4all.financials.cash_flow import CashFlowStatement
from financial4all.financials.ratios import FinancialRatios

__all__ = [
    "IncomeStatement",
    "BalanceSheet",
    "CashFlowStatement",
    "FinancialRatios",
]
