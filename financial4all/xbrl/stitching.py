# financial4all/xbrl/stitching.py
"""
Statement stitching for multi-period analysis.

This module provides functionality for stitching together statements
from multiple XBRL instances to enable trend analysis across filings.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from financial4all.xbrl.standardization import get_synonym_groups
from financial4all.xbrl.rendering import render_statement


class StatementStitcher:
    """
    Stitches statements across multiple filings.
    
    Handles concept changes across periods and provides intelligent
    period alignment for trend analysis.
    """
    
    def __init__(self):
        """Initialize statement stitcher."""
        self.synonym_groups = get_synonym_groups()
    
    def stitch_statements(
        self,
        xbrl_list: List[Any],
        statement_type: str = 'IncomeStatement',
        max_periods: int = 3,
        standard: bool = True
    ) -> Dict[str, Any]:
        """
        Stitch statements from multiple XBRL objects.
        
        Args:
            xbrl_list: List of XBRL objects, should be from the same company and ordered by date
            statement_type: Type of statement to stitch ('IncomeStatement', 'BalanceSheet', etc.)
            max_periods: Maximum number of periods to include (default: 3)
            standard: Whether to use standardized concept labels (default: True)
            
        Returns:
            Stitched statement data
        """
        if not xbrl_list:
            return {'items': [], 'periods': []}
        
        # Collect all statement items from all XBRL instances
        all_items = []
        all_periods = []
        
        for xbrl in xbrl_list:
            try:
                statement_data = xbrl.get_statement(statement_type)
                if statement_data:
                    all_items.extend(statement_data)
                    
                    # Collect periods
                    periods = getattr(xbrl, 'reporting_periods', [])
                    all_periods.extend(periods)
            except Exception:
                continue
        
        # Group items by concept (using standardization)
        concept_groups = defaultdict(list)
        
        for item in all_items:
            concept = item.get('concept', '')
            label = item.get('label', '')
            
            # Use standardization to group similar concepts
            if standard:
                concept_info = self.synonym_groups.identify_concept(concept)
                if concept_info:
                    standard_name = concept_info.name
                else:
                    standard_name = concept
            else:
                standard_name = concept
            
            concept_groups[standard_name].append({
                'item': item,
                'concept': concept,
                'label': label,
                'standard_name': standard_name
            })
        
        # Build stitched items
        stitched_items = []
        
        for standard_name, group in concept_groups.items():
            # Merge values across periods
            merged_values = {}
            merged_decimals = {}
            merged_units = {}
            
            # Use the first item's metadata as template
            template_item = group[0]['item']
            
            for item_data in group:
                item = item_data['item']
                values = item.get('values', {})
                decimals = item.get('decimals', {})
                units = item.get('units', {})
                
                # Merge values
                for period_key, value in values.items():
                    if period_key not in merged_values or merged_values[period_key] is None:
                        merged_values[period_key] = value
                        if period_key in decimals:
                            merged_decimals[period_key] = decimals[period_key]
                        if period_key in units:
                            merged_units[period_key] = units[period_key]
            
            # Create stitched item
            stitched_item = {
                'concept': standard_name,
                'label': template_item.get('label', ''),
                'values': merged_values,
                'decimals': merged_decimals,
                'units': merged_units,
                'level': template_item.get('level', 0),
                'has_values': len(merged_values) > 0
            }
            
            stitched_items.append(stitched_item)
        
        # Sort periods by date
        all_periods.sort(key=lambda p: p.get('end_date', p.get('date', '')), reverse=True)
        unique_periods = []
        seen_keys = set()
        
        for period in all_periods[:max_periods]:
            period_key = period.get('key')
            if period_key and period_key not in seen_keys:
                unique_periods.append(period)
                seen_keys.add(period_key)
        
        return {
            'items': stitched_items,
            'periods': unique_periods,
            'statement_type': statement_type
        }


class XBRLS:
    """
    Container for multiple XBRL instances.
    
    Provides access to stitched statements across multiple filings.
    """
    
    def __init__(self, xbrl_list: List[Any]):
        """
        Initialize XBRLS container.
        
        Args:
            xbrl_list: List of XBRL instances
        """
        self.xbrl_list = xbrl_list
        self._stitcher = StatementStitcher()
        self._stitched_statements_cache = {}
    
    @classmethod
    def from_filings(cls, filings: List[Any]) -> 'XBRLS':
        """
        Create XBRLS from a list of filings.
        
        Args:
            filings: List of filing objects
            
        Returns:
            XBRLS instance
        """
        xbrl_list = []
        for filing in filings:
            try:
                # Try to get XBRL from filing
                if hasattr(filing, 'xbrl'):
                    xbrl = filing.xbrl()
                elif hasattr(filing, 'get_xbrl'):
                    xbrl = filing.get_xbrl()
                else:
                    continue
                
                if xbrl:
                    xbrl_list.append(xbrl)
            except Exception:
                continue
        
        return cls(xbrl_list)
    
    @property
    def statements(self) -> 'StitchedStatements':
        """
        Get stitched statements collection.
        
        Returns:
            StitchedStatements instance
        """
        return StitchedStatements(self)


class StitchedStatements:
    """
    Stitched statement collection.
    
    Provides access to stitched statements with methods similar to
    the regular Statements class.
    """
    
    def __init__(self, xbrls: XBRLS):
        """
        Initialize stitched statements.
        
        Args:
            xbrls: XBRLS container
        """
        self.xbrls = xbrls
        self._stitcher = xbrls._stitcher
    
    def income_statement(self, max_periods: int = 3, standard: bool = True) -> Any:
        """
        Get stitched income statement.
        
        Args:
            max_periods: Maximum number of periods to include
            standard: Whether to use standardized labels
            
        Returns:
            Stitched income statement data
        """
        return self._stitcher.stitch_statements(
            self.xbrls.xbrl_list,
            'IncomeStatement',
            max_periods,
            standard
        )
    
    def balance_sheet(self, max_periods: int = 3, standard: bool = True) -> Any:
        """
        Get stitched balance sheet.
        
        Args:
            max_periods: Maximum number of periods to include
            standard: Whether to use standardized labels
            
        Returns:
            Stitched balance sheet data
        """
        return self._stitcher.stitch_statements(
            self.xbrls.xbrl_list,
            'BalanceSheet',
            max_periods,
            standard
        )
    
    def cashflow_statement(self, max_periods: int = 3, standard: bool = True) -> Any:
        """
        Get stitched cash flow statement.
        
        Args:
            max_periods: Maximum number of periods to include
            standard: Whether to use standardized labels
            
        Returns:
            Stitched cash flow statement data
        """
        return self._stitcher.stitch_statements(
            self.xbrls.xbrl_list,
            'CashFlowStatement',
            max_periods,
            standard
        )


def stitch_statements(
    xbrl_list: List[Any],
    statement_type: str = 'IncomeStatement',
    max_periods: int = 3,
    standard: bool = True
) -> Dict[str, Any]:
    """
    Stitch statements from multiple XBRL objects.
    
    Args:
        xbrl_list: List of XBRL objects
        statement_type: Type of statement to stitch
        max_periods: Maximum number of periods
        standard: Whether to use standardized labels
        
    Returns:
        Stitched statement data
    """
    stitcher = StatementStitcher()
    return stitcher.stitch_statements(xbrl_list, statement_type, max_periods, standard)


def render_stitched_statement(
    stitched_data: Dict[str, Any],
    statement_title: str,
    statement_type: str
) -> Any:
    """
    Render a stitched statement.
    
    Args:
        stitched_data: Stitched statement data
        statement_title: Title of the statement
        statement_type: Type of statement
        
    Returns:
        Rendered statement
    """
    items = stitched_data.get('items', [])
    periods = stitched_data.get('periods', [])
    
    # Convert periods to (key, label) tuples
    # Periods are already sorted newest first (reverse=True from sort)
    periods_to_display = [
        (p.get('key', ''), p.get('label', p.get('key', '')))
        for p in periods
    ]
    
    return render_statement(items, statement_title, periods_to_display)


def to_pandas(stitched_data: Dict[str, Any]) -> 'pd.DataFrame':
    """
    Convert stitched statement to pandas DataFrame.
    
    Args:
        stitched_data: Stitched statement data
        
    Returns:
        pandas DataFrame
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for to_pandas() function")
    
    items = stitched_data.get('items', [])
    periods = stitched_data.get('periods', [])
    
    # Reverse periods so most recent appears on the left
    periods = list(reversed(periods))
    
    rows = []
    for item in items:
        row = {
            'label': item.get('label', ''),
            'concept': item.get('concept', ''),
            'level': item.get('level', 0),
        }
        
        # Add values for each period
        for period in periods:
            period_key = period.get('key', '')
            period_label = period.get('label', period_key)
            value = item.get('values', {}).get(period_key)
            if value is not None:
                row[period_label] = value
        
        rows.append(row)
    
    return pd.DataFrame(rows)
