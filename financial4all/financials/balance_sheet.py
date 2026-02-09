# financial4all/financials/balance_sheet.py
"""
Balance sheet extraction and standardization.

This module provides functionality for extracting and standardizing
balance sheets from XBRL data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from financial4all.xbrl.facts import FactSet, Fact
from financial4all.xbrl.standardization import get_synonym_groups, get_default_store
from financial4all.xbrl.periods import PeriodType
from financial4all.core import log


class BalanceSheet:
    """
    Balance sheet extracted from XBRL data.

    This class handles extraction and standardization of balance sheet metrics.
    """

    # Mapping from display names to normalized concept names in SynonymGroups
    DISPLAY_NAME_TO_CONCEPT = {
        "Total Assets": "total_assets",
        "Current Assets": "total_current_assets",
        "Total Liabilities": "total_liabilities",
        "Current Liabilities": "total_current_liabilities",
        "Stockholders Equity": "stockholders_equity",
        "Receivables": "accounts_receivable",
        "Inventory": "inventory",
        "Payables": "accounts_payable",
        "Property, Plant & Equipment": "property_plant_equipment",
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
        Initialize balance sheet from fact set.

        Args:
            fact_set: FactSet containing balance sheet facts
        """
        # Balance sheets use instant periods (point in time)
        self.fact_set = fact_set
        self.standardizer = get_default_store()
        self._dataframe: Optional[pd.DataFrame] = None

    @classmethod
    def from_company_facts(
        cls, company_facts: Dict[str, Any], cik: Optional[str] = None
    ) -> "BalanceSheet":
        """
        Create balance sheet from SEC company facts API response.

        Args:
            company_facts: Dictionary from SEC company facts API
            cik: Optional CIK for entity info extraction

        Returns:
            BalanceSheet object
        """
        fact_set = FactSet.from_company_facts(company_facts, cik=cik)
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
    
    def _get_all_facts_for_metric(self, xbrl_concepts: List[str]) -> Dict[str, List[Fact]]:
        """
        Get all facts for a metric across all concept synonyms.
        
        Args:
            xbrl_concepts: List of XBRL concept names in priority order
            
        Returns:
            Dictionary mapping concept -> list of facts
        """
        all_facts_by_concept = {}
        
        for concept in xbrl_concepts:
            facts = self.fact_set.get_by_concept(concept)
            if facts:
                all_facts_by_concept[concept] = facts
        
        return all_facts_by_concept
    
    def _is_valid_unit_for_metric(self, unit: str, std_name: str) -> bool:
        """
        Check if a unit is valid for a given metric.
        
        Args:
            unit: Unit string (e.g., "USD", "shares")
            std_name: Standardized metric name
            
        Returns:
            True if the unit is valid for this metric
        """
        # All balance sheet metrics use USD units
        return unit == "USD" or unit.startswith("USD")
    
    def _resolve_concepts_by_period(
        self,
        std_name: str,
        xbrl_concepts: List[str],
    ) -> Dict[str, Any]:
        """
        Resolve which concept to use for each period using period-aware resolution.
        
        Strategy for balance sheet (INSTANT periods):
        1. Collect ALL facts from ALL concepts
        2. Apply multi-tier filtering to get best facts
        3. Group by period end date (instant date)
        4. For each period, select best fact based on priority:
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
        concept_facts_map = self._get_all_facts_for_metric(xbrl_concepts)
        
        if not concept_facts_map:
            return {}
        
        # Apply multi-tier filtering with preference for non-dimensional facts
        filtered_facts_by_concept = {}
        for concept, facts in concept_facts_map.items():
            # Separate dimensional and non-dimensional facts
            non_dimensional_facts = [f for f in facts if not f.dimensions]
            dimensional_facts = [f for f in facts if f.dimensions]
            
            # Tier 1: Strict filter (instant period, 10-K, valid unit, no dimensions) - PREFERRED
            tier1_facts = [
                f
                for f in non_dimensional_facts
                if f.period.period_type == PeriodType.INSTANT
                and f.form == "10-K"
                and self._is_valid_unit_for_metric(f.unit, std_name)
            ]
            
            # Tier 2: Lenient filter (instant period, valid unit, no dimensions, any form)
            if not tier1_facts:
                tier2_facts = [
                    f
                    for f in non_dimensional_facts
                    if f.period.period_type == PeriodType.INSTANT
                    and self._is_valid_unit_for_metric(f.unit, std_name)
                ]
                filtered_facts_by_concept[concept] = tier2_facts
            else:
                filtered_facts_by_concept[concept] = tier1_facts
            
            # Tier 3: Very lenient (any instant period, valid unit, no dimensions) - fallback
            if not filtered_facts_by_concept[concept]:
                tier3_facts = [
                    f
                    for f in non_dimensional_facts
                    if f.period.period_type == PeriodType.INSTANT
                    and self._is_valid_unit_for_metric(f.unit, std_name)
                ]
                filtered_facts_by_concept[concept] = tier3_facts
            
            # Tier 4: Last resort - include dimensional facts if no non-dimensional found
            if not filtered_facts_by_concept[concept]:
                tier4_facts = [
                    f
                    for f in dimensional_facts
                    if f.period.period_type == PeriodType.INSTANT
                    and self._is_valid_unit_for_metric(f.unit, std_name)
                ]
                if tier4_facts:
                    # For dimensional facts, prefer those without segment-specific dimensions
                    preferred_dimensional = []
                    for f in tier4_facts:
                        dims = f.dimensions or {}
                        # Prefer facts without segment dimensions (might be totals)
                        if not any(
                            "segment" in str(k).lower() or "product" in str(k).lower()
                            for k in dims.keys()
                        ):
                            preferred_dimensional.append(f)
                    
                    filtered_facts_by_concept[concept] = (
                        preferred_dimensional if preferred_dimensional else tier4_facts
                    )
        
        # Group facts by period and resolve conflicts
        period_facts_map: Dict[str, List[Tuple[int, Fact]]] = {}
        
        for concept_idx, (concept, facts) in enumerate(
            filtered_facts_by_concept.items()
        ):
            for fact in facts:
                period_key = str(fact.period.end)
                
                if period_key not in period_facts_map:
                    period_facts_map[period_key] = []
                
                # Store fact with concept priority for resolution
                period_facts_map[period_key].append((concept_idx, fact))
        
        # Resolve best fact for each period
        resolved_data = {}
        
        for period_key, fact_candidates in period_facts_map.items():
            if not fact_candidates:
                continue
            
            # Sort by priority: concept priority > has_dimensions > form > unit > filing date
            fact_candidates.sort(
                key=lambda x: (
                    x[0],  # Concept priority (lower = higher priority)
                    0 if not x[1].dimensions else 1,  # Prefer non-dimensional facts
                    0 if x[1].form == "10-K" else 1,  # Prefer 10-K
                    0 if self._is_valid_unit_for_metric(x[1].unit, std_name) else 1,  # Prefer valid unit
                    -(
                        x[1].filed.timestamp() if x[1].filed else float("-inf")
                    ),  # Prefer more recent (negated for descending)
                )
            )
            
            # Select best fact (first in sorted list)
            _, best_fact = fact_candidates[0]
            resolved_data[period_key] = best_fact.value
            
            # Log which concept was selected for debugging
            log.debug(
                f"BalanceSheet: Selected concept '{best_fact.concept}' for {std_name} "
                f"period {period_key} (value: {best_fact.value})"
            )
        
        return resolved_data
