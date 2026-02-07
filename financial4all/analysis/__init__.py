# financial4all/analysis/__init__.py
"""
Financial analysis and reporting module.

This module provides functionality for generating comprehensive financial
analysis reports including multi-year comparisons, ratios, trends, and Excel exports.
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
