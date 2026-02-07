# financial4all/analysis/report_generator.py
"""
Financial analysis report generator.

This module provides functionality for generating comprehensive financial
analysis reports including multi-year comparisons, ratios, trends, and more.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime

from financial4all.sec.company import Company
from financial4all.financials import IncomeStatement, BalanceSheet, CashFlowStatement, FinancialRatios
from financial4all.analysis.trend_analyzer import TrendAnalyzer
from financial4all.analysis.common_size import CommonSizeGenerator
from financial4all.core import log


class FinancialAnalysisReport:
    """
    Comprehensive financial analysis report generator.
    
    This class generates multi-year comparisons, ratios, trends, and
    common-size statements for a company.
    """
    
    def __init__(self, company: Company):
        """
        Initialize financial analysis report.
        
        Args:
            company: Company object to analyze
        """
        self.company = company
        self._financials: Optional[Dict[str, Any]] = None
        self._income_statement: Optional[IncomeStatement] = None
        self._balance_sheet: Optional[BalanceSheet] = None
        self._cash_flow: Optional[CashFlowStatement] = None
        self._ratios: Optional[FinancialRatios] = None
    
    def _load_financials(self) -> None:
        """Load financial statements if not already loaded."""
        if self._financials is None:
            self._financials = self.company.get_financials()
            self._income_statement = self._financials["income_statement"]
            self._balance_sheet = self._financials["balance_sheet"]
            self._cash_flow = self._financials["cash_flow"]
            self._ratios = FinancialRatios(
                self._income_statement,
                self._balance_sheet,
                self._cash_flow
            )
    
    def get_multi_year_comparison(self) -> Dict[str, pd.DataFrame]:
        """
        Get multi-year comparison tables for all statements.
        
        Returns:
            Dictionary with 'income_statement', 'balance_sheet', 'cash_flow' DataFrames
        """
        self._load_financials()
        
        result = {}
        
        # Income Statement
        is_df = self._income_statement.to_dataframe()
        if not is_df.empty:
            # Sort by date descending (most recent first)
            is_df = is_df.sort_index(ascending=False)
            result["income_statement"] = is_df
        
        # Balance Sheet
        if self._balance_sheet:
            bs_df = self._balance_sheet.to_dataframe()
            if not bs_df.empty:
                bs_df = bs_df.sort_index(ascending=False)
                result["balance_sheet"] = bs_df
        
        # Cash Flow
        if self._cash_flow:
            cf_df = self._cash_flow.to_dataframe()
            if not cf_df.empty:
                cf_df = cf_df.sort_index(ascending=False)
                result["cash_flow"] = cf_df
        
        return result
    
    def get_ratios_analysis(self) -> pd.DataFrame:
        """
        Get comprehensive ratios analysis.
        
        Returns:
            DataFrame with all calculated ratios
        """
        self._load_financials()
        return self._ratios.calculate_all_ratios()
    
    def get_trend_analysis(self) -> pd.DataFrame:
        """
        Get trend analysis with growth rates.
        
        Returns:
            DataFrame with growth rates and trend information
        """
        self._load_financials()
        
        is_df = self._income_statement.to_dataframe()
        if is_df.empty:
            return pd.DataFrame()
        
        analyzer = TrendAnalyzer(is_df)
        return analyzer.calculate_all_growth_rates()
    
    def get_trend_summary(self) -> pd.DataFrame:
        """
        Get trend summary for key metrics.
        
        Returns:
            DataFrame with trend information
        """
        self._load_financials()
        
        is_df = self._income_statement.to_dataframe()
        if is_df.empty:
            return pd.DataFrame()
        
        analyzer = TrendAnalyzer(is_df)
        return analyzer.get_trend_summary()
    
    def get_common_size_statements(self) -> Dict[str, pd.DataFrame]:
        """
        Get common-size financial statements.
        
        Returns:
            Dictionary with common-size statements
        """
        self._load_financials()
        
        result = {}
        
        # Common-size Income Statement
        is_df = self._income_statement.to_dataframe()
        if not is_df.empty:
            result["income_statement"] = CommonSizeGenerator.income_statement_common_size(is_df)
        
        # Common-size Balance Sheet
        if self._balance_sheet:
            bs_df = self._balance_sheet.to_dataframe()
            if not bs_df.empty:
                result["balance_sheet"] = CommonSizeGenerator.balance_sheet_common_size(bs_df)
        
        # Common-size Cash Flow
        if self._cash_flow:
            cf_df = self._cash_flow.to_dataframe()
            is_df = self._income_statement.to_dataframe()
            revenue = is_df["Revenue"] if "Revenue" in is_df.columns else None
            if not cf_df.empty:
                result["cash_flow"] = CommonSizeGenerator.cash_flow_common_size(cf_df, revenue)
        
        return result
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Get summary key metrics.
        
        Returns:
            Dictionary with key summary metrics
        """
        self._load_financials()
        
        is_df = self._income_statement.to_dataframe()
        bs_df = self._balance_sheet.to_dataframe() if self._balance_sheet else pd.DataFrame()
        cf_df = self._cash_flow.to_dataframe() if self._cash_flow else pd.DataFrame()
        
        summary = {
            "company_name": self.company.company_info["title"],
            "ticker": self.company.ticker,
            "cik": self.company.cik,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        }
        
        # Get most recent values
        if not is_df.empty:
            latest_is = is_df.iloc[0]  # Most recent (first row after descending sort)
            summary["revenue"] = latest_is.get("Revenue", np.nan)
            summary["net_income"] = latest_is.get("Net Income", np.nan)
            summary["gross_profit"] = latest_is.get("Gross Profit", np.nan)
            summary["operating_income"] = latest_is.get("Operating Income", np.nan)
        
        if not bs_df.empty:
            latest_bs = bs_df.iloc[0]
            summary["total_assets"] = latest_bs.get("Total Assets", np.nan)
            summary["total_liabilities"] = latest_bs.get("Total Liabilities", np.nan)
            summary["equity"] = latest_bs.get("Stockholders Equity", np.nan)
        
        if not cf_df.empty:
            latest_cf = cf_df.iloc[0]
            summary["operating_cash_flow"] = latest_cf.get("Operating Cash Flow", np.nan)
        
        # Calculate key ratios
        ratios_df = self.get_ratios_analysis()
        if not ratios_df.empty:
            latest_ratios = ratios_df.iloc[0]
            summary["gross_margin"] = latest_ratios.get("Gross Profit Margin", np.nan)
            summary["net_margin"] = latest_ratios.get("Net Profit Margin", np.nan)
            summary["roa"] = latest_ratios.get("Return on Assets (ROA)", np.nan)
            summary["roe"] = latest_ratios.get("Return on Equity (ROE)", np.nan)
        
        return summary
    
    def generate_report(self) -> Dict[str, pd.DataFrame]:
        """
        Generate complete analysis report.
        
        Returns:
            Dictionary with all analysis components
        """
        return {
            "multi_year_comparison": self.get_multi_year_comparison(),
            "ratios": self.get_ratios_analysis(),
            "trends": self.get_trend_analysis(),
            "trend_summary": self.get_trend_summary(),
            "common_size": self.get_common_size_statements(),
            "summary": pd.DataFrame([self.get_summary_metrics()])
        }
    
    def export_to_excel(self, file_path_or_buffer) -> None:
        """
        Export complete analysis to Excel file.
        
        Args:
            file_path_or_buffer: File path (string) or file-like object (BytesIO)
        """
        from financial4all.analysis.excel_exporter import ExcelExporter
        
        exporter = ExcelExporter()
        exporter.export_analysis(self, file_path_or_buffer, include_charts=False)
