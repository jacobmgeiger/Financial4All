# financial4all/financials/balance_sheet.py
"""
Balance sheet extraction and standardization.

This module provides functionality for extracting and standardizing
balance sheets from XBRL data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict

from financial4all.xbrl.facts import FactSet, Fact
from financial4all.xbrl.standardization import get_default_store
from financial4all.core import log


class BalanceSheet:
    """
    Balance sheet extracted from XBRL data.
    
    This class handles extraction and standardization of balance sheet metrics.
    """
    
    # Standardized balance sheet mapping
    STANDARD_MAPPING = {
        "Total Assets": ["Assets"],
        "Current Assets": ["CurrentAssets"],
        "Total Liabilities": ["Liabilities"],
        "Current Liabilities": ["CurrentLiabilities"],
        "Stockholders Equity": ["StockholdersEquity", "Equity"],
    }
    
    def __init__(self, fact_set: FactSet):
        """
        Initialize balance sheet from fact set.
        
        Args:
            fact_set: FactSet containing balance sheet facts
        """
        # Balance sheets use instant periods (point in time)
        self.fact_set = fact_set
        self.standardizer = get_default_store()
        self._dataframe: Optional[pd.DataFrame] = None
    
    @classmethod
    def from_company_facts(cls, company_facts: Dict[str, Any]) -> "BalanceSheet":
        """
        Create balance sheet from SEC company facts API response.
        
        Args:
            company_facts: Dictionary from SEC company facts API
            
        Returns:
            BalanceSheet object
        """
        fact_set = FactSet.from_company_facts(company_facts)
        return cls(fact_set)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert balance sheet to pandas DataFrame.
        
        Only includes metrics that have at least one reported value.
        Filters out completely empty columns.
        
        Returns:
            DataFrame with standardized balance sheet metrics
        """
        if self._dataframe is not None:
            return self._dataframe
        
        # Extract metrics by standard name
        metrics_data = defaultdict(dict)
        reported_metrics = set()  # Track which metrics have at least one value
        
        for std_name, xbrl_concepts in self.STANDARD_MAPPING.items():
            # Try each XBRL concept in priority order
            for concept in xbrl_concepts:
                facts = self.fact_set.get_by_concept(concept)
                
                if facts:
                    # Group by period
                    for fact in facts:
                        period_key = str(fact.period.end)
                        
                        # Use USD unit if available
                        if fact.unit == "USD" or fact.unit.startswith("USD"):
                            if period_key not in metrics_data[std_name]:
                                metrics_data[std_name][period_key] = fact.value
                                reported_metrics.add(std_name)
                    # Found data for this metric, move to next standard name
                    break
        
        # Convert to DataFrame
        if not metrics_data or not reported_metrics:
            return pd.DataFrame()
        
        # Get all unique periods
        all_periods = set()
        for metric_data in metrics_data.values():
            all_periods.update(metric_data.keys())
        
        all_periods = sorted(all_periods)
        
        # Build DataFrame - only include reported metrics
        df_data = {}
        for std_name in reported_metrics:
            df_data[std_name] = [
                metrics_data[std_name].get(period, np.nan)
                for period in all_periods
            ]
        
        df = pd.DataFrame(df_data, index=all_periods)
        df.index.name = "end"
        
        # Filter out completely empty columns
        df = df.loc[:, ~df.isna().all()]
        
        self._dataframe = df
        return df
    
    def get_metric(self, metric_name: str, period_offset: int = 0) -> Optional[float]:
        """
        Get a specific metric value.
        
        Args:
            metric_name: Standardized metric name
            period_offset: Period offset (0 = most recent)
            
        Returns:
            Metric value or None if not found
        """
        df = self.to_dataframe()
        
        if metric_name not in df.columns:
            return None
        
        if len(df) <= period_offset:
            return None
        
        df_sorted = df.sort_index(ascending=False)
        value = df_sorted.iloc[period_offset][metric_name]
        
        if pd.isna(value):
            return None
        
        return float(value)
