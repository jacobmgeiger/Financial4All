# financial4all/financials/cash_flow.py
"""
Cash flow statement extraction and standardization.

This module provides functionality for extracting and standardizing
cash flow statements from XBRL data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict

from financial4all.xbrl.facts import FactSet, Fact
from financial4all.xbrl.standardization import get_synonym_groups, get_default_store
from financial4all.core import log


class CashFlowStatement:
    """
    Cash flow statement extracted from XBRL data.

    This class handles extraction and standardization of cash flow metrics.
    """

    # Mapping from display names to normalized concept names in SynonymGroups
    DISPLAY_NAME_TO_CONCEPT = {
        "Operating Cash Flow": "operating_cash_flow",
        "Investing Cash Flow": "investing_cash_flow",
        "Financing Cash Flow": "financing_cash_flow",
        "Net Change in Cash": "net_change_in_cash",
    }

    # Cached standard mapping
    _STANDARD_MAPPING_CACHE: Optional[Dict[str, List[str]]] = None

    @classmethod
    def _get_standard_mapping(cls) -> Dict[str, List[str]]:
        """
        Get standard mapping using SynonymGroups system.

        Returns:
            Dictionary mapping display names to lists of XBRL concept synonyms
        """
        if cls._STANDARD_MAPPING_CACHE is not None:
            return cls._STANDARD_MAPPING_CACHE

        synonyms = get_synonym_groups()
        mapping = {}

        for display_name, concept_name in cls.DISPLAY_NAME_TO_CONCEPT.items():
            group = synonyms.get_group(concept_name)
            if group:
                mapping[display_name] = group.synonyms
            else:
                log.warning(
                    f"Concept '{concept_name}' not found in SynonymGroups for '{display_name}'"
                )
                mapping[display_name] = []

        cls._STANDARD_MAPPING_CACHE = mapping
        return mapping

    @property
    def STANDARD_MAPPING(self) -> Dict[str, List[str]]:
        """Get standard mapping (backward compatibility)."""
        return self._get_standard_mapping()

    def __init__(self, fact_set: FactSet):
        """
        Initialize cash flow statement from fact set.

        Args:
            fact_set: FactSet containing cash flow facts
        """
        # Use filter_annual() to capture more historical data from all form types
        self.fact_set = fact_set.filter_annual()
        self.standardizer = get_default_store()
        self._dataframe: Optional[pd.DataFrame] = None

    @classmethod
    def from_company_facts(
        cls, company_facts: Dict[str, Any], cik: Optional[str] = None
    ) -> "CashFlowStatement":
        """
        Create cash flow statement from SEC company facts API response.

        Args:
            company_facts: Dictionary from SEC company facts API
            cik: Optional CIK for entity info extraction

        Returns:
            CashFlowStatement object
        """
        fact_set = FactSet.from_company_facts(company_facts, cik=cik)
        return cls(fact_set)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert cash flow statement to pandas DataFrame.

        Only includes metrics that have at least one reported value.
        Filters out completely empty columns.

        Returns:
            DataFrame with standardized cash flow metrics
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

        # Sort periods with most recent first (for leftmost column display)
        all_periods = sorted(all_periods, reverse=True)

        # Build DataFrame - only include reported metrics
        df_data = {}
        for std_name in reported_metrics:
            df_data[std_name] = [
                metrics_data[std_name].get(period, np.nan) for period in all_periods
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
