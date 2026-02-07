# financial4all/__init__.py
"""
Financial4All - A Python library for accessing and analyzing SEC EDGAR financial data.

This package provides tools for fetching, parsing, and analyzing XBRL financial statements
from the SEC EDGAR database.

Example:
    >>> from financial4all import Company, set_identity
    >>> set_identity("your.email@example.com")
    >>> company = Company("AAPL")
    >>> financials = company.get_financials()
    >>> income_statement = financials["income_statement"]
    >>> df = income_statement.to_dataframe()
"""

from financial4all.sec.company import Company
from financial4all.config import set_identity, get_config
from financial4all.financials import IncomeStatement, BalanceSheet, CashFlowStatement, FinancialRatios
from financial4all.xbrl.facts import FactSet, Fact
from financial4all.xbrl.periods import Period, PeriodType
from financial4all.exceptions import (
    Financial4AllError,
    SECAPIError,
    CompanyNotFoundError,
    InvalidTickerError,
    InvalidCIKError,
    XBRLError,
    NoDataError,
    CalculationError,
)

__version__ = "0.1.0"
__all__ = [
    "Company",
    "set_identity",
    "get_config",
    "IncomeStatement",
    "BalanceSheet",
    "CashFlowStatement",
    "FinancialRatios",
    "FactSet",
    "Fact",
    "Period",
    "PeriodType",
    "Financial4AllError",
    "SECAPIError",
    "CompanyNotFoundError",
    "InvalidTickerError",
    "InvalidCIKError",
    "XBRLError",
    "NoDataError",
    "CalculationError",
]
