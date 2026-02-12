# financial4all/financials/ratios.py
"""
Financial ratio calculations from standardized statements.

FinancialRatios takes IncomeStatement and optionally BalanceSheet and
CashFlowStatement, and produces DataFrames of profitability (margins),
liquidity (e.g. current ratio), and efficiency ratios. Period index
aligns with the underlying statement DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from financial4all.financials.income_statement import IncomeStatement
from financial4all.financials.balance_sheet import BalanceSheet
from financial4all.financials.cash_flow import CashFlowStatement
from financial4all.core import log


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
        Calculate profitability ratios.
        
        Returns:
            DataFrame with profitability ratios
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
        Calculate efficiency ratios.
        
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
        Calculate return ratios (ROA, ROE, ROIC).
        
        Returns:
            DataFrame with return ratios
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
        
        # Combine all ratios
        all_indices = set()
        for df in [profitability, liquidity, efficiency, leverage, returns]:
            all_indices.update(df.index)
        
        all_ratios = pd.DataFrame(index=sorted(all_indices))
        
        for df in [profitability, liquidity, efficiency, leverage, returns]:
            for col in df.columns:
                all_ratios[col] = df[col]
        
        return all_ratios.dropna(how='all')
