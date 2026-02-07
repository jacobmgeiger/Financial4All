# financial4all/analysis/common_size.py
"""
Common-size financial statement generator.

This module provides functionality for generating common-size financial
statements where line items are expressed as percentages of a base figure.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from financial4all.core import log


class CommonSizeGenerator:
    """
    Generates common-size financial statements.
    
    Common-size statements express each line item as a percentage of
    a base figure (revenue for income statement, total assets for balance sheet).
    """
    
    @staticmethod
    def income_statement_common_size(is_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert income statement to common-size format (% of revenue).
        
        Args:
            is_df: Income statement DataFrame
            
        Returns:
            Common-size income statement DataFrame
        """
        if is_df.empty:
            return pd.DataFrame()
        
        if "Revenue" not in is_df.columns:
            log.warning("Revenue column not found for common-size income statement")
            return pd.DataFrame()
        
        common_size = is_df.copy()
        revenue = is_df["Revenue"]
        
        # Convert all columns to percentages of revenue
        for col in common_size.columns:
            if col != "Revenue":
                common_size[col] = (
                    common_size[col].divide(revenue).replace([np.inf, -np.inf], np.nan) * 100
                )
            else:
                # Revenue is always 100%
                common_size[col] = 100.0
        
        return common_size
    
    @staticmethod
    def balance_sheet_common_size(bs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert balance sheet to common-size format (% of total assets).
        
        Args:
            bs_df: Balance sheet DataFrame
            
        Returns:
            Common-size balance sheet DataFrame
        """
        if bs_df.empty:
            return pd.DataFrame()
        
        if "Total Assets" not in bs_df.columns:
            log.warning("Total Assets column not found for common-size balance sheet")
            return pd.DataFrame()
        
        common_size = bs_df.copy()
        total_assets = bs_df["Total Assets"]
        
        # Convert all columns to percentages of total assets
        for col in common_size.columns:
            if col != "Total Assets":
                common_size[col] = (
                    common_size[col].divide(total_assets).replace([np.inf, -np.inf], np.nan) * 100
                )
            else:
                # Total Assets is always 100%
                common_size[col] = 100.0
        
        return common_size
    
    @staticmethod
    def cash_flow_common_size(cf_df: pd.DataFrame, revenue: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Convert cash flow statement to common-size format (% of revenue).
        
        Args:
            cf_df: Cash flow statement DataFrame
            revenue: Optional revenue series (if not provided, uses first column as base)
            
        Returns:
            Common-size cash flow DataFrame
        """
        if cf_df.empty:
            return pd.DataFrame()
        
        common_size = cf_df.copy()
        
        # Use revenue if provided, otherwise use first column
        if revenue is not None:
            base = revenue
        elif len(cf_df.columns) > 0:
            base = cf_df.iloc[:, 0]
        else:
            return pd.DataFrame()
        
        # Convert all columns to percentages
        for col in common_size.columns:
            common_size[col] = (
                common_size[col].divide(base).replace([np.inf, -np.inf], np.nan) * 100
            )
        
        return common_size
