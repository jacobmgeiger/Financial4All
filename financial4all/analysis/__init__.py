# financial4all/analysis/__init__.py
"""
Financial analysis and reporting on standardized statements.

Provides TrendAnalyzer (period-over-period), CommonSizeGenerator (vertical/horizontal),
ProfitabilityAnalyzer (margins, returns), and FinancialAnalysisReport (combined
report plus Excel export with formulas). All operate on DataFrames from the
financials module.
"""

from financial4all.analysis.report_generator import FinancialAnalysisReport
from financial4all.analysis.trend_analyzer import TrendAnalyzer
from financial4all.analysis.common_size import CommonSizeGenerator
from financial4all.analysis.profitability_analyzer import ProfitabilityAnalyzer

__all__ = [
    "FinancialAnalysisReport",
    "TrendAnalyzer",
    "CommonSizeGenerator",
    "ProfitabilityAnalyzer",
]
