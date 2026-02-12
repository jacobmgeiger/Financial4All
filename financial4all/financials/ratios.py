# financial4all/financials/ratios.py
"""
Financial ratio calculations from standardized statements.

FinancialRatios takes IncomeStatement and optionally BalanceSheet and
CashFlowStatement, and produces DataFrames of profitability (margins),
liquidity (e.g. current ratio), efficiency, leverage, return, and cash ratios.
Formulas align with standard practice and EdgarTools where applicable:
  - Current Ratio = Current Assets / Current Liabilities (EdgarTools)
  - Debt-to-Assets = Total Liabilities / Total Assets (EdgarTools)
  - Free Cash Flow = Operating Cash Flow - |CapEx| (EdgarTools)
"""

import pandas as pd
import numpy as np
from typing import Optional

from financial4all.financials.income_statement import IncomeStatement
from financial4all.financials.balance_sheet import BalanceSheet
from financial4all.financials.cash_flow import CashFlowStatement


class FinancialRatios:
    """
    Computes financial ratios from standardized income statement, balance sheet, and cash flow.

    All methods return a DataFrame indexed by period (same as the source statement).
    Ratios are computed only where required columns exist; missing metrics yield NaN.
    """

    def __init__(
        self,
        income_statement: IncomeStatement,
        balance_sheet: Optional[BalanceSheet] = None,
        cash_flow: Optional[CashFlowStatement] = None,
    ) -> None:
        """
        Initialize the ratios calculator with at least an income statement.

        Args:
            income_statement: Required; used for profitability ratios.
            balance_sheet: Optional; used for liquidity (and related) ratios.
            cash_flow: Optional; used for cash-based ratios if needed.
        """
        self.income_statement = income_statement
        self.balance_sheet = balance_sheet
        self.cash_flow = cash_flow
    
    def calculate_profitability_ratios(self) -> pd.DataFrame:
        """
        Calculate profitability ratios (margins as % of revenue).

        Formulas:
          - Gross Profit Margin = Gross Profit / Revenue × 100
          - Operating Profit Margin = Operating Income / Revenue × 100
          - Net Profit Margin = Net Income / Revenue × 100

        Returns:
            DataFrame with profitability ratios (percentages 0–100)
        """
        is_df = self.income_statement.to_dataframe()
        
        if is_df.empty:
            return pd.DataFrame()
        
        ratios = pd.DataFrame(index=is_df.index)
        
        # Gross Profit Margin
        if "Revenue" in is_df.columns and "Gross Profit" in is_df.columns:
            revenue = is_df["Revenue"]
            gross_profit = is_df["Gross Profit"]
            ratios["Gross Profit Margin"] = (
                gross_profit.divide(revenue).replace([np.inf, -np.inf], np.nan) * 100
            )
        
        # Operating Profit Margin
        if "Revenue" in is_df.columns and "Operating Income" in is_df.columns:
            revenue = is_df["Revenue"]
            operating_income = is_df["Operating Income"]
            ratios["Operating Profit Margin"] = (
                operating_income.divide(revenue).replace([np.inf, -np.inf], np.nan) * 100
            )
        
        # Net Profit Margin
        if "Revenue" in is_df.columns and "Net Income" in is_df.columns:
            revenue = is_df["Revenue"]
            net_income = is_df["Net Income"]
            ratios["Net Profit Margin"] = (
                net_income.divide(revenue).replace([np.inf, -np.inf], np.nan) * 100
            )
        
        return ratios.dropna(how='all')
    
    def calculate_liquidity_ratios(self) -> pd.DataFrame:
        """
        Calculate liquidity ratios.

        Current Ratio = Current Assets / Current Liabilities
        (Matches EdgarTools get_financial_metrics current_ratio.)

        Returns:
            DataFrame with liquidity ratios
        """
        if self.balance_sheet is None:
            return pd.DataFrame()
        
        bs_df = self.balance_sheet.to_dataframe()
        
        if bs_df.empty:
            return pd.DataFrame()
        
        ratios = pd.DataFrame(index=bs_df.index)
        
        # Current Ratio
        if "Current Assets" in bs_df.columns and "Current Liabilities" in bs_df.columns:
            current_assets = bs_df["Current Assets"]
            current_liabilities = bs_df["Current Liabilities"]
            ratios["Current Ratio"] = (
                current_assets.divide(current_liabilities).replace([np.inf, -np.inf], np.nan)
            )
        
        return ratios.dropna(how='all')
    
    def calculate_efficiency_ratios(self) -> pd.DataFrame:
        """
        Calculate efficiency ratios (period-aligned with income statement).

        Asset Turnover = Revenue / Total Assets
        Uses common_index between income statement and balance sheet periods.

        Returns:
            DataFrame with efficiency ratios
        """
        ratios = pd.DataFrame()
        
        is_df = self.income_statement.to_dataframe()
        bs_df = self.balance_sheet.to_dataframe() if self.balance_sheet else pd.DataFrame()
        
        if is_df.empty or bs_df.empty:
            return ratios
        
        # Asset Turnover
        if "Revenue" in is_df.columns and "Total Assets" in bs_df.columns:
            # Align indices
            common_index = is_df.index.intersection(bs_df.index)
            if len(common_index) > 0:
                revenue = is_df.loc[common_index, "Revenue"]
                total_assets = bs_df.loc[common_index, "Total Assets"]
                ratios = pd.DataFrame(index=common_index)
                ratios["Asset Turnover"] = (
                    revenue.divide(total_assets).replace([np.inf, -np.inf], np.nan)
                )
        
        return ratios.dropna(how='all')
    
    def calculate_leverage_ratios(self) -> pd.DataFrame:
        """
        Calculate leverage ratios.

        Formulas:
          - Debt-to-Equity = Total Liabilities / Stockholders Equity
          - Debt-to-Assets = Total Liabilities / Total Assets (matches EdgarTools)

        Returns:
            DataFrame with leverage ratios
        """
        if self.balance_sheet is None:
            return pd.DataFrame()
        
        bs_df = self.balance_sheet.to_dataframe()
        
        if bs_df.empty:
            return pd.DataFrame()
        
        ratios = pd.DataFrame(index=bs_df.index)
        
        # Debt-to-Equity Ratio
        if "Total Liabilities" in bs_df.columns and "Stockholders Equity" in bs_df.columns:
            total_liabilities = bs_df["Total Liabilities"]
            equity = bs_df["Stockholders Equity"]
            ratios["Debt-to-Equity Ratio"] = (
                total_liabilities.divide(equity).replace([np.inf, -np.inf], np.nan)
            )
        
        # Debt-to-Assets Ratio
        if "Total Liabilities" in bs_df.columns and "Total Assets" in bs_df.columns:
            total_liabilities = bs_df["Total Liabilities"]
            total_assets = bs_df["Total Assets"]
            ratios["Debt-to-Assets Ratio"] = (
                total_liabilities.divide(total_assets).replace([np.inf, -np.inf], np.nan)
            )
        
        return ratios.dropna(how='all')
    
    def calculate_return_ratios(self) -> pd.DataFrame:
        """
        Calculate return ratios (ROA, ROE as % of base).

        Formulas (end-of-period balance sheet, period-aligned):
          - ROA = Net Income / Total Assets × 100
          - ROE = Net Income / Stockholders Equity × 100

        Returns:
            DataFrame with return ratios (percentages)
        """
        ratios = pd.DataFrame()
        
        is_df = self.income_statement.to_dataframe()
        bs_df = self.balance_sheet.to_dataframe() if self.balance_sheet else pd.DataFrame()
        
        if is_df.empty or bs_df.empty:
            return ratios
        
        # Align indices
        common_index = is_df.index.intersection(bs_df.index)
        if len(common_index) == 0:
            return ratios
        
        ratios = pd.DataFrame(index=common_index)
        
        # Return on Assets (ROA)
        if "Net Income" in is_df.columns and "Total Assets" in bs_df.columns:
            net_income = is_df.loc[common_index, "Net Income"]
            total_assets = bs_df.loc[common_index, "Total Assets"]
            ratios["Return on Assets (ROA)"] = (
                net_income.divide(total_assets).replace([np.inf, -np.inf], np.nan) * 100
            )
        
        # Return on Equity (ROE)
        if "Net Income" in is_df.columns and "Stockholders Equity" in bs_df.columns:
            net_income = is_df.loc[common_index, "Net Income"]
            equity = bs_df.loc[common_index, "Stockholders Equity"]
            ratios["Return on Equity (ROE)"] = (
                net_income.divide(equity).replace([np.inf, -np.inf], np.nan) * 100
            )
        
        return ratios.dropna(how='all')
    
    def calculate_cash_ratios(self) -> pd.DataFrame:
        """
        Calculate cash-based metrics.

        Free Cash Flow = Operating Cash Flow - |CapEx|
        (Aligns with EdgarTools. CapEx is typically negative in XBRL; abs() ensures
        correct subtraction regardless of sign convention.)

        Returns:
            DataFrame with Free Cash Flow and other cash metrics indexed by period
        """
        if self.cash_flow is None:
            return pd.DataFrame()
        
        cf_df = self.cash_flow.to_dataframe()
        if cf_df.empty:
            return pd.DataFrame()
        
        ratios = pd.DataFrame(index=cf_df.index)
        
        # Free Cash Flow = OCF - |CapEx| (EdgarTools convention)
        if "Operating Cash Flow" in cf_df.columns and "CapEx" in cf_df.columns:
            ocf = cf_df["Operating Cash Flow"]
            capex = cf_df["CapEx"]
            # CapEx is usually negative (outflow); abs() ensures correct subtraction
            fcf = ocf - np.abs(capex)
            ratios["Free Cash Flow"] = (
                fcf.replace([np.inf, -np.inf], np.nan)
            )
        
        return ratios.dropna(how='all')
    
    def calculate_all_ratios(self) -> pd.DataFrame:
        """
        Calculate all available ratios.
        
        Returns:
            DataFrame with all calculated ratios
        """
        profitability = self.calculate_profitability_ratios()
        liquidity = self.calculate_liquidity_ratios()
        efficiency = self.calculate_efficiency_ratios()
        leverage = self.calculate_leverage_ratios()
        returns = self.calculate_return_ratios()
        cash = self.calculate_cash_ratios()
        
        # Combine all ratios
        all_indices = set()
        for df in [profitability, liquidity, efficiency, leverage, returns, cash]:
            all_indices.update(df.index)
        
        all_ratios = pd.DataFrame(index=sorted(all_indices))
        
        for df in [profitability, liquidity, efficiency, leverage, returns, cash]:
            for col in df.columns:
                all_ratios[col] = df[col]
        
        return all_ratios.dropna(how='all')
