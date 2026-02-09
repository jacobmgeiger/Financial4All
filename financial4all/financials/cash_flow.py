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

from financial4all.xbrl.facts import FactSet
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
        "Depreciation & Amortization": "depreciation_and_amortization",
        "CapEx": "capex",
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

    def _resolve_concepts_by_period(
        self,
        std_name: str,
        xbrl_concepts: List[str],
    ) -> Dict[str, Any]:
        """
        Resolve which concept to use for each period using period-aware resolution.

        Strategy:
        1. Collect ALL facts from ALL concepts
        2. Group by period end date
        3. For each period, select best fact based on priority:
           - Concept priority (earlier in list = higher priority)
           - Form type (10-K preferred over 10-Q)
           - Unit (USD)
           - Filing date (more recent preferred)

        Args:
            std_name: Standardized metric name
            xbrl_concepts: List of XBRL concept names in priority order

        Returns:
            Dictionary mapping period_key -> fact.value
        """
        # Get all facts for all concepts
        concept_facts_map = {}
        for concept in xbrl_concepts:
            facts = self.fact_set.get_by_concept(concept)
            if facts:
                concept_facts_map[concept] = facts

        if not concept_facts_map:
            return {}

        # Group facts by period
        facts_by_period = defaultdict(list)
        concept_priority = {concept: idx for idx, concept in enumerate(xbrl_concepts)}

        for concept, facts in concept_facts_map.items():
            for fact in facts:
                # Only use USD units
                if fact.unit == "USD" or fact.unit.startswith("USD"):
                    period_key = str(fact.period.end)
                    priority = concept_priority.get(concept, 999)
                    # Prefer 10-K over 10-Q (0 for 10-K, 100 for others)
                    form_bonus = 0 if fact.form == "10-K" else 100
                    # Prefer more recent filings (negate timestamp for descending sort)
                    filing_bonus = -(fact.filed.timestamp() if fact.filed else 0)

                    facts_by_period[period_key].append(
                        (
                            priority + form_bonus,  # Combined priority score
                            filing_bonus,  # Filing date (for tie-breaking)
                            concept,
                            fact,
                        )
                    )

        # For each period, select best fact
        resolved_data = {}

        for period_key, fact_candidates in facts_by_period.items():
            # Sort by priority (lower is better), then by filing date (more recent is better)
            fact_candidates.sort(key=lambda x: (x[0], x[1]))

            # Select the best fact (first in sorted list)
            _, _, selected_concept, best_fact = fact_candidates[0]

            # Handle CapEx sign: CapEx is typically negative (cash outflow) in cash flow statements
            # We want to store as positive for display purposes
            value = best_fact.value
            if std_name == "CapEx" and value is not None:
                # Ensure CapEx is stored as positive (absolute value)
                # Most XBRL tags for CapEx are already negative, but some might be positive
                # Take absolute value to ensure consistency
                try:
                    value = abs(float(value))
                except (ValueError, TypeError):
                    pass

            resolved_data[period_key] = value

            # Log which concept was selected for debugging
            log.debug(
                f"CashFlowStatement: Selected concept '{selected_concept}' for {std_name} "
                f"period {period_key} (value: {value})"
            )

        return resolved_data

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

        # Extract metrics by standard name using period-aware resolution
        metrics_data = defaultdict(dict)
        reported_metrics = set()  # Track which metrics have at least one value

        for std_name, xbrl_concepts in self.STANDARD_MAPPING.items():
            # Use period-aware resolution to fill gaps by trying all concepts for each period
            resolved_data = self._resolve_concepts_by_period(std_name, xbrl_concepts)

            if resolved_data:
                for period_key, value in resolved_data.items():
                    metrics_data[std_name][period_key] = value
                    reported_metrics.add(std_name)

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
