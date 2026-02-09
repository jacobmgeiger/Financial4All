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
from financial4all.financials.capex_resolver import CapExResolver, CapExValidator
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

    # Depreciation-only concepts for CapEx fallback formula (PP&E identity uses depreciation, not D&A)
    DEPRECIATION_ONLY_CONCEPTS = [
        "Depreciation",
        "DepreciationExpense",
    ]

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
        # Keep raw fact set for CapEx fallback (PP&E Net is instant; filter_annual drops it)
        self._raw_fact_set = fact_set
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

    # OLD METHODS REMOVED - Now using CapExResolver class
    # _resolve_capex_with_aggregation() and _calculate_capex_from_balance_sheet() 
    # have been replaced by CapExResolver.resolve_all_periods()
    # Removed ~970 lines of code that are now centralized in capex_resolver.py
    
    def _extract_base_unit(self, unit: str) -> str:
        """
        Extract base unit from unit string, removing scale indicators.
        
        Args:
            unit: Unit string (e.g., "USD (millions)", "USD", "shares")
            
        Returns:
            Base unit string (e.g., "USD", "shares")
        """
        unit_lower = unit.lower().strip()
        
        # Remove scale indicators
        scale_patterns = ["(thousands)", "(millions)", "(billions)", "thousands", "millions", "billions"]
        base_unit = unit
        for pattern in scale_patterns:
            base_unit = base_unit.replace(pattern, "").replace(pattern.lower(), "")
        
        # Clean up
        base_unit = base_unit.replace("()", "").strip()
        base_unit = base_unit.replace("  ", " ").strip()
        if base_unit.startswith("(") and base_unit.endswith(")"):
            base_unit = base_unit[1:-1].strip()
        
        # Extract just the currency/measure part
        if "/" in base_unit:
            # For units like "USD/Pure" or "USD/Share", take the first part
            base_unit = base_unit.split("/")[0].strip()
        
        # Normalize common variations
        if "usd" in base_unit.lower() or "dollar" in base_unit.lower():
            return "USD"
        elif "share" in base_unit.lower():
            return "shares"
        
        return base_unit if base_unit else "USD"
    
    def _calculate_capex_from_balance_sheet(
        self,
        period_key: str,
        ppe_series: pd.Series,
        ppe_periods: List[str],
        depreciation_series: pd.Series,
        is_df: Optional[pd.DataFrame] = None,
    ) -> Optional[float]:
        """
        OLD METHOD - DEPRECATED
        This method has been replaced by CapExResolver._resolve_from_balance_sheet()
        """
        return None
    
    @staticmethod
    def _get_aligned_value(
        series: pd.Series,
        target_date: str,
        available_periods: list
    ) -> Optional[float]:
        """
        Get value from series aligned to target date period.
        
        Handles period alignment between balance sheet (instant periods) and 
        income statement (period-end dates). Finds the closest matching period.
        
        Args:
            series: Series with values indexed by period
            target_date: Target date string to align to
            available_periods: List of available periods in the series
            
        Returns:
            Aligned value or None if not found
        """
        if series.empty or not available_periods:
            return None
        
        # Try exact match first
        if target_date in available_periods:
            value = series.get(target_date)
            if pd.isna(value):
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        
        # Try to parse target_date and find closest period
        try:
            target_dt = pd.to_datetime(target_date)
            
            # Find closest period by comparing dates
            closest_period = None
            min_diff = None
            
            for period in available_periods:
                try:
                    period_dt = pd.to_datetime(period)
                    diff = abs((target_dt - period_dt).days)
                    
                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                        closest_period = period
                except (ValueError, TypeError):
                    continue
            
            if closest_period is not None and min_diff is not None and min_diff <= 365:
                # Only use if within 1 year
                value = series.get(closest_period)
                if pd.isna(value):
                    return None
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None
        except (ValueError, TypeError):
            pass
        
        return None

    def _get_ppe_change_for_period(
        self, 
        period_key: str, 
        bs_df: Optional[pd.DataFrame]
    ) -> Optional[float]:
        """
        Get PP&E change for a period (ending - beginning PP&E).
        
        Args:
            period_key: Period key (DURATION period end date)
            bs_df: Balance sheet DataFrame
            
        Returns:
            PP&E change (ending - beginning) or None if not available
        """
        if bs_df is None or "Property, Plant & Equipment" not in bs_df.columns:
            return None
        
        ppe_series = bs_df["Property, Plant & Equipment"]
        ending_ppe = self._get_aligned_value(ppe_series, period_key, ppe_series.index.tolist())
        
        if ending_ppe is None:
            return None
        
        # Find beginning PP&E (previous period)
        period_dt = pd.to_datetime(period_key)
        beginning_ppe = None
        
        for prev_period in sorted(ppe_series.index, reverse=True):
            try:
                prev_dt = pd.to_datetime(prev_period)
                if prev_dt < period_dt:
                    diff_days = abs((period_dt - prev_dt).days)
                    if 300 <= diff_days <= 400:  # ~1 year
                        beginning_ppe = ppe_series.get(prev_period)
                        break
            except (ValueError, TypeError):
                continue
        
        if beginning_ppe is None:
            return None
        
        try:
            return float(ending_ppe) - float(beginning_ppe)
        except (ValueError, TypeError):
            return None

    def _get_ppe_for_period(
        self,
        period_key: str,
        bs_df: Optional[pd.DataFrame]
    ) -> Optional[float]:
        """
        Get PP&E value for a period.
        
        Args:
            period_key: Period key
            bs_df: Balance sheet DataFrame
            
        Returns:
            PP&E value or None if not available
        """
        if bs_df is None or "Property, Plant & Equipment" not in bs_df.columns:
            return None
        
        ppe_series = bs_df["Property, Plant & Equipment"]
        ppe_value = self._get_aligned_value(ppe_series, period_key, ppe_series.index.tolist())
        return float(ppe_value) if ppe_value is not None else None

    def _get_revenue_for_period(
        self,
        period_key: str,
        is_df: Optional[pd.DataFrame]
    ) -> Optional[float]:
        """
        Get Revenue for a period from income statement.
        
        Args:
            period_key: Period key
            is_df: Income statement DataFrame
            
        Returns:
            Revenue value or None if not available
        """
        if is_df is None or "Revenue" not in is_df.columns:
            return None
        
        revenue_series = is_df["Revenue"]
        revenue_value = self._get_aligned_value(revenue_series, period_key, revenue_series.index.tolist())
        return float(revenue_value) if revenue_value is not None else None

    def _get_da_for_period(
        self,
        period_key: str,
        depreciation_series: Optional[pd.Series]
    ) -> Optional[float]:
        """
        Get D&A value for a period.
        
        Args:
            period_key: Period key
            depreciation_series: D&A Series
            
        Returns:
            D&A value or None if not available
        """
        if depreciation_series is None:
            return None
        
        da_value = self._get_aligned_value(
            depreciation_series, period_key, depreciation_series.index.tolist()
        )
        return float(da_value) if da_value is not None else None

    def _get_previous_capex(
        self,
        period_key: str,
        resolved_data: Dict[str, Any]
    ) -> Optional[float]:
        """
        Get previous period's CapEx for historical comparison.
        
        Args:
            period_key: Current period key
            resolved_data: Dictionary of resolved CapEx values
            
        Returns:
            Previous period CapEx or None if not available
        """
        try:
            period_dt = pd.to_datetime(period_key)
            previous_periods = []
            for k in resolved_data.keys():
                try:
                    k_dt = pd.to_datetime(k)
                    if k_dt < period_dt:
                        previous_periods.append((k, k_dt))
                except (ValueError, TypeError):
                    continue
            
            if not previous_periods:
                return None
            
            # Get most recent previous period
            previous_periods.sort(key=lambda x: x[1], reverse=True)
            prev_key = previous_periods[0][0]
            prev_value = resolved_data.get(prev_key)
            return float(prev_value) if prev_value is not None else None
        except (ValueError, TypeError, KeyError):
            return None

    def _get_previous_da(
        self,
        period_key: str,
        depreciation_series: Optional[pd.Series]
    ) -> Optional[float]:
        """
        Get previous period's D&A for historical comparison.
        
        Args:
            period_key: Current period key
            depreciation_series: D&A Series
            
        Returns:
            Previous period D&A or None if not available
        """
        if depreciation_series is None or len(depreciation_series) == 0:
            return None
        
        try:
            period_dt = pd.to_datetime(period_key)
            previous_periods = []
            for k in depreciation_series.index:
                try:
                    k_dt = pd.to_datetime(k)
                    if k_dt < period_dt:
                        previous_periods.append((k, k_dt))
                except (ValueError, TypeError):
                    continue
            
            if not previous_periods:
                return None
            
            previous_periods.sort(key=lambda x: x[1], reverse=True)
            prev_key = previous_periods[0][0]
            prev_value = depreciation_series[prev_key]
            return float(prev_value) if prev_value is not None and not pd.isna(prev_value) else None
        except (ValueError, TypeError, KeyError):
            return None

    def _get_previous_da_from_series(
        self,
        period_key: str,
        depreciation_series: pd.Series
    ) -> Optional[float]:
        """
        Get previous period's D&A from Series (alias for _get_previous_da with non-optional Series).
        
        Args:
            period_key: Current period key
            depreciation_series: D&A Series (required)
            
        Returns:
            Previous period D&A or None if not available
        """
        return self._get_previous_da(period_key, depreciation_series)

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
        
        Special handling for CapEx: aggregates component tags when comprehensive tag doesn't exist.
        
        Args:
            std_name: Standardized metric name
            xbrl_concepts: List of XBRL concept names in priority order
            
        Returns:
            Dictionary mapping period_key -> fact.value
        """
        # Special handling for CapEx: aggregate component tags
        # Note: CapEx resolution is handled directly in to_dataframe() with proper parameters
        # This method should not be called for CapEx - it's handled separately
        if std_name == "CapEx":
            log.warning(
                f"_resolve_concepts_by_period called for CapEx - this should be handled by "
                f"_resolve_capex_with_aggregation in to_dataframe()"
            )
            return {}
        
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
                    
                    facts_by_period[period_key].append((
                        priority + form_bonus,  # Combined priority score
                        filing_bonus,  # Filing date (for tie-breaking)
                        concept,
                        fact
                    ))
        
        # For each period, select best fact
        resolved_data = {}
        
        for period_key, fact_candidates in facts_by_period.items():
            # Sort by priority (lower is better), then by filing date (more recent is better)
            fact_candidates.sort(key=lambda x: (x[0], x[1]))
            
            # Select the best fact (first in sorted list)
            _, _, selected_concept, best_fact = fact_candidates[0]
            
            resolved_data[period_key] = best_fact.value
            
            # Log which concept was selected for debugging
            log.debug(
                f"CashFlowStatement: Selected concept '{selected_concept}' for {std_name} "
                f"period {period_key} (value: {best_fact.value})"
            )
        
        return resolved_data

    def to_dataframe(
        self,
        bs_df: Optional[pd.DataFrame] = None,
        depreciation_series: Optional[pd.Series] = None,
        is_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Convert cash flow statement to pandas DataFrame.

        Only includes metrics that have at least one reported value.
        Filters out completely empty columns.
        
        For CapEx, if bs_df and depreciation_series are provided, missing values
        will be calculated using the formula: (Ending PP&E - Beginning PP&E) + Depreciation.
        
        D&A values are validated during extraction to catch unit scale issues.

        Args:
            bs_df: Optional balance sheet DataFrame for CapEx fallback calculation and validation
            depreciation_series: Optional depreciation Series for CapEx fallback calculation
            is_df: Optional income statement DataFrame for revenue validation

        Returns:
            DataFrame with standardized cash flow metrics
        """
        if self._dataframe is not None:
            return self._dataframe

        # Extract metrics by standard name using period-aware resolution
        metrics_data = defaultdict(dict)
        reported_metrics = set()  # Track which metrics have at least one value

        # Extract D&A first if needed for CapEx fallback calculation.
        # Prefer depreciation-only when available (PP&E identity uses depreciation, not D&A).
        depreciation_series_for_capex = depreciation_series
        if bs_df is not None and depreciation_series_for_capex is None:
            if "Depreciation & Amortization" in self.STANDARD_MAPPING:
                da_concepts = self.STANDARD_MAPPING["Depreciation & Amortization"]
                # Try depreciation-only first for CapEx formula correctness
                depr_only_resolved = self._resolve_concepts_by_period(
                    "Depreciation & Amortization", self.DEPRECIATION_ONLY_CONCEPTS
                )
                da_resolved = self._resolve_concepts_by_period("Depreciation & Amortization", da_concepts)
                
                # Prefer depreciation-only per period; fill missing with full D&A
                combined_da = dict(da_resolved) if da_resolved else {}
                if depr_only_resolved:
                    for period_key, val in depr_only_resolved.items():
                        combined_da[period_key] = val
                
                if combined_da:
                    validated_da = {}
                    for period_key, da_value in combined_da.items():
                        ppe_value = self._get_ppe_for_period(period_key, bs_df)
                        revenue = self._get_revenue_for_period(period_key, is_df)
                        prev_da = self._get_previous_da(period_key, pd.Series(validated_da) if validated_da else None)
                        validation_result = CapExValidator.validate_da_value(
                            da_value, period_key, ppe_value, revenue, prev_da
                        )
                        if validation_result['is_valid'] or validation_result['confidence'] >= 0.2:
                            validated_da[period_key] = da_value
                            if not validation_result['is_valid']:
                                log.warning(
                                    f"D&A validation warning for {period_key}: {validation_result['issues']}"
                                )
                        else:
                            log.error(
                                f"Rejecting D&A value {da_value:.0f} for {period_key} due to validation failure. "
                                f"Issues: {validation_result['issues']}"
                            )
                    depreciation_series_for_capex = pd.Series(validated_da)
                    sample_periods = sorted(depreciation_series_for_capex.index)[-5:]
                    sample_values = {p: depreciation_series_for_capex[p] for p in sample_periods}
                    log.info(
                        f"CashFlowStatement: Extracted and validated D&A for CapEx fallback: "
                        f"{len(depreciation_series_for_capex)} periods. "
                        f"Sample values: {sample_values}"
                    )
        
        for std_name, xbrl_concepts in self.STANDARD_MAPPING.items():
            # Use period-aware resolution to fill gaps by trying all concepts for each period
            if std_name == "CapEx":
                # Use new centralized CapExResolver; pass raw fact set for PP&E Net (instant facts)
                resolver = CapExResolver(
                    self.fact_set,
                    bs_df=bs_df,
                    depreciation_series=depreciation_series_for_capex,
                    is_df=is_df,
                    ppe_fact_set=self._raw_fact_set,
                )
                resolved_data = resolver.resolve_all_periods(xbrl_concepts)
            else:
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
