# financial4all/analysis/profitability_analyzer.py
"""
Profitability ratio analyzer for income statements.

This module provides functionality for calculating profitability ratios
including Y/Y revenue growth, expenses as percentage of revenue, operating margin,
and tax rate.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union

from financial4all.core import log


class ProfitabilityAnalyzer:
    """
    Analyzer for calculating profitability ratios from income statements.
    
    Calculates year-over-year revenue growth, expenses as percentage of revenue,
    operating margin, and tax rate.
    """
    
    @staticmethod
    def calculate_ratios(is_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate profitability ratios from an income statement DataFrame.
        
        Args:
            is_df: Income statement DataFrame with periods as index and metrics as columns.
                  This is the format returned by IncomeStatement.to_dataframe().
        
        Returns:
            DataFrame in transposed format (Metric column + date columns) ready for display.
            Values are returned as decimals (0.50 for 50%).
        """
        if is_df.empty:
            return pd.DataFrame()
        
        # Get date columns (periods) from index
        date_columns = is_df.index.tolist()
        
        # Initialize result dictionary
        result_data = {}
        
        # 1. Y/Y Revenue Growth
        if "Revenue" in is_df.columns:
            revenue_growth = ProfitabilityAnalyzer._calculate_yoy_growth(
                is_df["Revenue"], date_columns
            )
            result_data["Revenue"] = revenue_growth
        
        # 2. Expenses as % of Revenue section
        # Add section header row (empty values)
        result_data["Expenses as % of Revenue"] = {col: np.nan for col in date_columns}
        
        # Calculate each expense as % of Revenue
        expense_metrics = [
            ("Gross Margin", "Gross Profit"),
            ("Research and development", "R&D Expenses"),
            ("Sales, general and administrative", "SG&A Expenses"),
            ("Restructuring and other charges", "Restructuring and other charges"),
            ("Acquisition termination cost", "Acquisition termination cost"),
            ("Interest income", "Interest Income"),
            ("Interest expense", "Interest Expense"),
            ("Other, net", "Other, net"),
        ]
        
        for display_name, metric_name in expense_metrics:
            if metric_name in is_df.columns and "Revenue" in is_df.columns:
                expense_pct = ProfitabilityAnalyzer._calculate_percentage_of_revenue(
                    is_df[metric_name], is_df["Revenue"], date_columns
                )
                result_data[display_name] = expense_pct
        
        # 3. Operating Margin = Operating Income / Revenue
        if "Operating Income" in is_df.columns and "Revenue" in is_df.columns:
            operating_margin = ProfitabilityAnalyzer._calculate_percentage_of_revenue(
                is_df["Operating Income"], is_df["Revenue"], date_columns
            )
            result_data["Operating Margin"] = operating_margin
        
        # 4. Tax Rate = Income Tax Expense (benefit) / Profit Before Taxes
        # Note: "Taxes" is the standard metric name, "Income Before Taxes" is the standard name
        if "Taxes" in is_df.columns and "Income Before Taxes" in is_df.columns:
            tax_rate = ProfitabilityAnalyzer._calculate_percentage_of_revenue(
                is_df["Taxes"], is_df["Income Before Taxes"], date_columns
            )
            result_data["Tax rate"] = tax_rate
        
        # Convert to DataFrame with Metric as index, then transpose to get Metric column
        if not result_data:
            return pd.DataFrame()
        
        # Build DataFrame: rows are metrics, columns are dates
        result_df = pd.DataFrame(result_data, index=date_columns)
        result_df = result_df.T  # Transpose: rows become metrics, columns become dates
        
        # Reset index and rename to "Metric"
        result_df = result_df.reset_index().rename(columns={"index": "Metric"})
        
        return result_df
    
    @staticmethod
    def _parse_numeric_value(value: Union[str, float, int, None]) -> Optional[float]:
        """
        Parse a numeric value, handling parentheses notation for negatives.
        
        Args:
            value: Value that may be a number, string with parentheses, or None
        
        Returns:
            Float value (negative if parentheses notation was used), or None if invalid
        """
        if value is None or pd.isna(value):
            return None
        
        # If already a numeric type, return as float
        if isinstance(value, (int, float)):
            return float(value)
        
        # If string, check for parentheses notation
        if isinstance(value, str):
            value = value.strip()
            # Check if value is wrapped in parentheses (accounting notation for negative)
            if value.startswith('(') and value.endswith(')'):
                # Remove parentheses and parse as negative
                inner_value = value[1:-1].strip()
                try:
                    return -float(inner_value.replace(',', ''))
                except (ValueError, TypeError):
                    return None
            else:
                # Regular parsing
                try:
                    return float(value.replace(',', ''))
                except (ValueError, TypeError):
                    return None
        
        # Try direct conversion
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _calculate_yoy_growth(series: pd.Series, date_columns: list) -> Dict[str, float]:
        """
        Calculate year-over-year growth for a metric.
        
        With periods ordered newest-first, compares each period to the next (older) period.
        For example, with [2024, 2023, 2022]:
        - 2024 growth compares 2024 to 2023
        - 2023 growth compares 2023 to 2022
        - 2022 growth is NaN (no older period)
        
        Args:
            series: Series with values indexed by period
            date_columns: List of date column names (periods), ordered newest-first
        
        Returns:
            Dictionary mapping date columns to growth rates (as decimals)
        """
        growth_data = {}
        
        # Iterate through periods (newest to oldest)
        # Compare each period to the next period (which is older)
        for i, date_col in enumerate(date_columns):
            current_value = series.get(date_col)
            current_float = ProfitabilityAnalyzer._parse_numeric_value(current_value)
            
            # Check if there's a next (older) period to compare to
            if i + 1 < len(date_columns):
                next_date_col = date_columns[i + 1]
                next_value = series.get(next_date_col)
                next_float = ProfitabilityAnalyzer._parse_numeric_value(next_value)
                
                if current_float is not None and next_float is not None:
                    try:
                        if next_float != 0:
                            # Growth = (current - previous) / previous
                            # where "previous" is the next (older) period
                            growth = (current_float - next_float) / next_float
                            growth_data[date_col] = growth
                        else:
                            growth_data[date_col] = np.nan
                    except (ValueError, TypeError, ZeroDivisionError):
                        growth_data[date_col] = np.nan
                else:
                    growth_data[date_col] = np.nan
            else:
                # No older period to compare to (oldest period)
                growth_data[date_col] = np.nan
        
        return growth_data
    
    @staticmethod
    def _calculate_percentage_of_revenue(
        numerator_series: pd.Series,
        denominator_series: pd.Series,
        date_columns: list
    ) -> Dict[str, float]:
        """
        Calculate numerator as percentage of denominator for each period.
        
        Args:
            numerator_series: Series with numerator values indexed by period
            denominator_series: Series with denominator values indexed by period
            date_columns: List of date column names (periods)
        
        Returns:
            Dictionary mapping date columns to percentages (as decimals, e.g., 0.50 for 50%)
            Negative values are preserved as negative (e.g., -0.50 for -50%)
        """
        pct_data = {}
        
        for date_col in date_columns:
            numerator_value = numerator_series.get(date_col)
            denominator_value = denominator_series.get(date_col)
            
            # Parse values, handling parentheses notation for negatives
            num_float = ProfitabilityAnalyzer._parse_numeric_value(numerator_value)
            denom_float = ProfitabilityAnalyzer._parse_numeric_value(denominator_value)
            
            if num_float is not None and denom_float is not None:
                try:
                    if denom_float != 0:
                        pct = num_float / denom_float
                        pct_data[date_col] = pct
                    else:
                        pct_data[date_col] = np.nan
                except (ValueError, TypeError, ZeroDivisionError):
                    pct_data[date_col] = np.nan
            else:
                pct_data[date_col] = np.nan
        
        return pct_data
