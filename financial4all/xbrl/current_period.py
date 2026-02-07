# financial4all/xbrl/current_period.py
"""
Current period view for easy access to current reporting period data.

This module provides a simplified API for accessing financial statements
for the current reporting period without needing to specify periods explicitly.
"""

from typing import Any, Dict, List, Optional

from financial4all.xbrl.period_selector import select_periods
from financial4all.xbrl.rendering import render_statement


class CurrentPeriodView:
    """
    View focused on current reporting period.
    
    Provides simplified access to financial statements for the current
    reporting period with automatic period selection.
    """
    
    def __init__(self, xbrl):
        """
        Initialize current period view.
        
        Args:
            xbrl: XBRL instance
        """
        self.xbrl = xbrl
    
    def balance_sheet(self, **kwargs) -> Any:
        """
        Get current period balance sheet.
        
        Args:
            **kwargs: Additional arguments passed to get_statement
            
        Returns:
            Balance sheet statement data
        """
        # Get current period for balance sheet
        periods = select_periods(self.xbrl, 'BalanceSheet')
        if periods:
            period_filter = periods[0][0]  # Use first (current) period
            return self.xbrl.get_statement('BalanceSheet', period_filter=period_filter, **kwargs)
        return []
    
    def income_statement(self, raw_concepts: bool = False, **kwargs) -> Any:
        """
        Get current period income statement.
        
        Args:
            raw_concepts: If True, return raw concept names without standardization
            **kwargs: Additional arguments passed to get_statement
            
        Returns:
            Income statement data
        """
        # Get current period for income statement
        periods = select_periods(self.xbrl, 'IncomeStatement')
        if periods:
            period_filter = periods[0][0]  # Use first (current) period
            return self.xbrl.get_statement('IncomeStatement', period_filter=period_filter, **kwargs)
        return []
    
    def cashflow_statement(self, **kwargs) -> Any:
        """
        Get current period cash flow statement.
        
        Args:
            **kwargs: Additional arguments passed to get_statement
            
        Returns:
            Cash flow statement data
        """
        # Get current period for cash flow statement
        periods = select_periods(self.xbrl, 'CashFlowStatement')
        if periods:
            period_filter = periods[0][0]  # Use first (current) period
            return self.xbrl.get_statement('CashFlowStatement', period_filter=period_filter, **kwargs)
        return []
    
    def statement_of_equity(self, **kwargs) -> Any:
        """
        Get current period statement of equity.
        
        Args:
            **kwargs: Additional arguments passed to get_statement
            
        Returns:
            Statement of equity data
        """
        # Get current period for statement of equity
        periods = select_periods(self.xbrl, 'StatementOfEquity')
        if periods:
            period_filter = periods[0][0]  # Use first (current) period
            return self.xbrl.get_statement('StatementOfEquity', period_filter=period_filter, **kwargs)
        return []
