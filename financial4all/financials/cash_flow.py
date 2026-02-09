# financial4all/financials/cash_flow.py
"""
Cash flow statement extraction and standardization.

This module provides functionality for extracting and standardizing
cash flow statements from XBRL data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

from datetime import date as date_type

from financial4all.xbrl.facts import FactSet
from financial4all.xbrl.periods import PeriodType
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

    # Two-tier CapEx fallback disabled: M&A (PaymentsToAcquireBusinessesNetOfCashAcquired) is not CapEx
    # and would mislabel e.g. NVDA 2021 (8,524 M&A vs 1,128 CapEx). Prefer showing — when no PP&E fact.
    CAPEX_FALLBACK_CONCEPTS: List[str] = []

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
        # Keep unfiltered fact set for fallback concept discovery (mirrors income statement)
        self._original_fact_set = fact_set
        # Use filter_annual() to capture more historical data from all form types
        self.fact_set = fact_set.filter_annual()
        self.standardizer = get_default_store()
        self._dataframe: Optional[pd.DataFrame] = None

    @staticmethod
    def _normalize_period_key(period_end: Any) -> str:
        """
        Normalize period end to YYYY-MM-DD string for consistent alignment.

        Args:
            period_end: Period end (date, datetime, or str)

        Returns:
            String in YYYY-MM-DD format
        """
        try:
            if hasattr(period_end, "strftime"):
                return period_end.strftime("%Y-%m-%d")
            return pd.to_datetime(period_end).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return str(period_end)

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
        # Get all facts for all concepts (include namespace variants for company facts API)
        concept_facts_map = {}
        for concept in xbrl_concepts:
            facts = self.fact_set.get_all_facts_for_concept(concept, include_variants=True)
            if facts:
                concept_facts_map[concept] = facts

        if not concept_facts_map:
            return {}

        # Group facts by period
        facts_by_period = defaultdict(list)
        concept_priority = {concept: idx for idx, concept in enumerate(xbrl_concepts)}

        # Fallback: facts with no unit or generic numeric unit (for periods with no USD fact)
        facts_by_period_fallback = defaultdict(list)

        for concept, facts in concept_facts_map.items():
            for fact in facts:
                period_key = self._normalize_period_key(fact.period.end)
                priority = concept_priority.get(concept, 999)
                form_bonus = 0 if fact.form == "10-K" else 100
                filing_bonus = -(fact.filed.timestamp() if fact.filed else 0)
                # Prefer non-dimensional (consolidated) over segment/dimensional facts
                dim_sort = 0 if (not fact.dimensions or len(fact.dimensions) == 0) else 1
                # CapEx: prefer combined "PP&E and intangible assets" line over PP&E-only when both exist
                combined_sort = (
                    0 if (std_name == "CapEx" and concept and "IntangibleAssets" in concept) else 1
                )
                entry = (
                    dim_sort,
                    combined_sort,
                    priority + form_bonus,
                    filing_bonus,
                    concept,
                    fact,
                )

                # Prefer USD units
                if fact.unit == "USD" or (fact.unit and fact.unit.startswith("USD")):
                    facts_by_period[period_key].append(entry)
                # Last resort for CapEx only: no unit or generic numeric unit (Investopedia: investing section)
                # Do not apply to D&A or other metrics—non-USD facts can be wrong (e.g. ratios, segments)
                elif std_name == "CapEx" and (not fact.unit or fact.unit in ("", "pure")):
                    facts_by_period_fallback[period_key].append(entry)

        # For each period, select best fact (USD only).
        # CapEx: prefer combined "PP&E and intangible assets" line first (even if dimensional), then non-dimensional
        # Others: prefer non-dimensional then concept priority/form/filing
        resolved_data = {}

        for period_key, fact_candidates in facts_by_period.items():
            if std_name == "CapEx":
                fact_candidates.sort(key=lambda x: (x[1], x[0], x[2], x[3]))  # combined_sort, dim_sort, ...
            else:
                fact_candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))  # dim_sort, combined_sort, ...
            _, _, _, _, selected_concept, best_fact = fact_candidates[0]

            value = best_fact.value
            if std_name == "CapEx" and value is not None:
                try:
                    value = abs(float(value))
                except (ValueError, TypeError):
                    pass

            resolved_data[period_key] = value

            log.debug(
                f"CashFlowStatement: Selected concept '{selected_concept}' for {std_name} "
                f"period {period_key} (value: {value})"
            )

        # Optional unit fallback (CapEx only): for periods still missing, use non-USD fact if available
        # D&A and other metrics are USD-only to avoid wrong values from ratios/segment facts
        for period_key, fact_candidates in facts_by_period_fallback.items():
            if period_key in resolved_data:
                continue
            if not fact_candidates:
                continue
            if std_name == "CapEx":
                fact_candidates.sort(key=lambda x: (x[1], x[0], x[2], x[3]))
            else:
                fact_candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
            _, _, _, _, selected_concept, best_fact = fact_candidates[0]
            value = best_fact.value
            if std_name == "CapEx" and value is not None:
                try:
                    value = abs(float(value))
                except (ValueError, TypeError):
                    pass
            resolved_data[period_key] = value
            log.debug(
                f"CashFlowStatement: Used non-USD fallback for {std_name} period {period_key} "
                f"(concept: {selected_concept}, unit: {best_fact.unit})"
            )

        return resolved_data

    def _discover_fallback_concepts(
        self, std_name: str, existing_concepts: List[str], target_periods: Set[str]
    ) -> List[str]:
        """
        Discover alternative concepts to fill gaps in period coverage.

        Uses SynonymGroups and pattern matching to find concepts that might
        represent the same metric but weren't in the original mapping.
        Mirrors income statement's _discover_fallback_concepts.

        Args:
            std_name: Standardized metric name
            existing_concepts: List of concepts already tried
            target_periods: Set of period keys (YYYY-MM-DD) that need data

        Returns:
            List of discovered alternative concept names
        """
        fallback_concepts = []
        synonym_groups = get_synonym_groups()

        # Resolve display name to concept name for SynonymGroups lookup
        concept_name = self.DISPLAY_NAME_TO_CONCEPT.get(std_name)
        if concept_name:
            group = synonym_groups.get_group(concept_name)
            if group:
                all_synonyms = group.synonyms
                log.debug(
                    f"CashFlowStatement: Using SynonymGroups for '{std_name}' "
                    f"(concept: '{concept_name}') with {len(all_synonyms)} synonyms"
                )

                all_concepts_in_factset = {
                    f.concept for f in self._original_fact_set.facts
                }

                for tag in all_synonyms:
                    for variant in [tag, f"us-gaap_{tag}", f"us-gaap:{tag}"]:
                        if variant in existing_concepts:
                            continue
                        if variant not in all_concepts_in_factset:
                            continue

                        synonym_facts = (
                            self._original_fact_set.get_all_facts_for_concept(
                                variant, include_variants=True
                            )
                        )
                        if not synonym_facts:
                            continue

                        synonym_periods = {
                            self._normalize_period_key(f.period.end)
                            for f in synonym_facts
                            if f.period.period_type == PeriodType.DURATION
                            and f.period.is_annual()
                        }
                        missing_periods_covered = synonym_periods.intersection(
                            target_periods
                        )

                        if missing_periods_covered:
                            fallback_concepts.append(variant)
                            log.debug(
                                f"CashFlowStatement: Found fallback concept '{variant}' "
                                f"for '{std_name}' covering periods: {missing_periods_covered}"
                            )

        # Pattern-based synonym detection if SynonymGroups didn't help
        if not fallback_concepts:
            for concept in existing_concepts:
                synonyms = self._original_fact_set.find_synonym_concepts(concept)
                for synonym in synonyms:
                    if synonym in existing_concepts:
                        continue
                    synonym_facts = (
                        self._original_fact_set.get_all_facts_for_concept(
                            synonym, include_variants=True
                        )
                    )
                    if not synonym_facts:
                        continue
                    synonym_periods = {
                        self._normalize_period_key(f.period.end)
                        for f in synonym_facts
                        if f.period.period_type == PeriodType.DURATION
                        and f.period.is_annual()
                    }
                    missing_periods_covered = synonym_periods.intersection(
                        target_periods
                    )
                    if missing_periods_covered:
                        fallback_concepts.append(synonym)
                        log.debug(
                            f"CashFlowStatement: Found pattern-based fallback '{synonym}' "
                            f"for '{std_name}' covering periods: {missing_periods_covered}"
                        )

        return fallback_concepts

    def _aggregate_quarterly_capex_to_annual(self) -> Dict[str, float]:
        """
        Sum quarterly CapEx facts into annual values for fiscal years that lack 10-K annual facts.

        Many filers (e.g. NVIDIA 2013–2020) report CapEx only in 10-Q with no 10-K annual
        fact; the SEC API may provide single-quarter (90d) and cumulative (181d, 272d) facts.
        We sum four single quarters when available, or derive Q1+Q2+Q3 from cumulative and
        add Q4 when present.

        Returns:
            Dict mapping period_key (YYYY-MM-DD) to annual CapEx from summed quarters.
        """
        out: Dict[str, float] = {}
        capex_concepts = self.STANDARD_MAPPING.get("CapEx", [])
        if not capex_concepts:
            return out
        # Collect duration USD facts by inferred fiscal year end
        by_fy: Dict[str, List[tuple]] = defaultdict(list)  # fy_key -> [(end_date, days, value)]
        for concept in capex_concepts:
            facts = self._original_fact_set.get_all_facts_for_concept(concept, include_variants=True)
            for fact in facts:
                if fact.period.period_type != PeriodType.DURATION or not fact.period.start:
                    continue
                if fact.unit != "USD" and not (fact.unit and str(fact.unit).startswith("USD")):
                    continue
                try:
                    val = abs(float(fact.value))
                except (ValueError, TypeError):
                    continue
                s, e = fact.period.start, fact.period.end
                start_d = s if isinstance(s, date_type) else date_type.fromisoformat(str(s)[:10])
                end_d = e if isinstance(e, date_type) else date_type.fromisoformat(str(e)[:10])
                days = (end_d - start_d).days
                y, m = end_d.year, end_d.month
                fy_key = self._normalize_period_key(end_d) if m <= 3 else f"{y + 1}-01-31"
                by_fy[fy_key].append((end_d, days, val))
        for fy_key, candidates in by_fy.items():
            # Dedupe by (end_date, days): keep one value per period
            seen: Set[tuple] = set()
            unique: List[tuple] = []
            for end_d, days, val in sorted(candidates, key=lambda x: (x[0], x[1])):
                key = (end_d, days)
                if key in seen:
                    continue
                seen.add(key)
                unique.append((end_d, days, val))
            if not unique:
                continue
            # Prefer single annual fact
            annual = next((v for end_d, d, v in unique if 330 <= d <= 400), None)
            if annual is not None:
                out[fy_key] = annual
                continue
            # Four single quarters (each ~90 days)
            quarters = [(end_d, d, v) for end_d, d, v in unique if 80 <= d <= 100]
            if len(quarters) >= 4:
                # One value per quarter end (same end can appear with different durations in API)
                by_end: Dict[date_type, float] = {}
                for end_d, _d, v in sorted(quarters, key=lambda x: x[0]):
                    by_end[end_d] = v
                if len(by_end) >= 4:
                    out[fy_key] = sum(by_end.values())
                continue
            # Cumulative (e.g. 90, 181, 272 days): Q1 = v1, Q2 = v2-v1, Q3 = v3-v2; need Q4
            sorted_by_end = sorted(unique, key=lambda x: x[0])
            if len(sorted_by_end) >= 3:
                ends = [x[0] for x in sorted_by_end]
                days_list = [x[1] for x in sorted_by_end]
                vals = [x[2] for x in sorted_by_end]
                if 80 <= days_list[0] <= 100 and 175 <= days_list[1] <= 190 and 265 <= days_list[2] <= 280:
                    q1, q2, q3 = vals[0], vals[1] - vals[0], vals[2] - vals[1]
                    # Q4: fact ending in Jan with ~90 days in same FY
                    fy_year = date_type.fromisoformat(fy_key[:10]).year
                    q4_candidates = [(e, d, v) for e, d, v in unique if e.month <= 3 and 80 <= d <= 100 and e.year == fy_year]
                    if q4_candidates:
                        q4 = max(q4_candidates, key=lambda x: x[0])[2]
                        out[fy_key] = q1 + q2 + q3 + q4
                    else:
                        # SEC API often has no Q4 for these years (e.g. NVDA 2013–2020). Skip 9-month proxy
                        # so we don't show understated "annual" figures; leave period unfilled (—).
                        pass
        return out

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

        # First pass: extract metrics by standard name using period-aware resolution
        metrics_data = defaultdict(dict)
        reported_metrics = set()  # Track which metrics have at least one value

        for std_name, xbrl_concepts in self.STANDARD_MAPPING.items():
            resolved_data = self._resolve_concepts_by_period(std_name, xbrl_concepts)

            if resolved_data:
                for period_key, value in resolved_data.items():
                    metrics_data[std_name][period_key] = value
                    reported_metrics.add(std_name)

        # Second pass: fill missing periods via fallback concept discovery (mirrors income statement)
        all_periods = set()
        for metric_data in metrics_data.values():
            for k in metric_data:
                all_periods.add(self._normalize_period_key(k))

        for std_name, xbrl_concepts in self.STANDARD_MAPPING.items():
            covered_periods = set(metrics_data[std_name].keys())
            missing_periods = all_periods - covered_periods
            if not missing_periods:
                continue
            fallback_concepts = self._discover_fallback_concepts(
                std_name, xbrl_concepts, missing_periods
            )
            if not fallback_concepts:
                continue
            fallback_data = self._resolve_concepts_by_period(
                std_name, fallback_concepts
            )
            for period_key, value in fallback_data.items():
                if (
                    period_key in missing_periods
                    and period_key not in metrics_data[std_name]
                ):
                    metrics_data[std_name][period_key] = value
                    log.debug(
                        f"CashFlowStatement: Added fallback data for {std_name} "
                        f"period {period_key} from concept discovery"
                    )

        # Two-tier CapEx: fill still-missing periods with acquisition-style concepts only
        if "CapEx" in metrics_data:
            capex_missing = all_periods - set(metrics_data["CapEx"].keys())
            if capex_missing and self.CAPEX_FALLBACK_CONCEPTS:
                capex_fallback_data = self._resolve_concepts_by_period(
                    "CapEx", self.CAPEX_FALLBACK_CONCEPTS
                )
                for period_key, value in capex_fallback_data.items():
                    if period_key in capex_missing and period_key not in metrics_data["CapEx"]:
                        metrics_data["CapEx"][period_key] = value
                        log.debug(
                            "CashFlowStatement: CapEx period %s filled from acquisition fallback "
                            "(no PP&E concept had data)",
                            period_key,
                        )
            # CapEx from summed quarters when no 10-K annual fact (e.g. NVDA 2013–2020 in SEC API).
            # Reuse existing period key for that fiscal year when present (avoids duplicate FY columns).
            def _fiscal_year_key(pk: str) -> str:
                try:
                    y, m = int(pk[:4]), int(pk[5:7])
                    return f"{y}-01" if m <= 3 else f"{y + 1}-01"
                except (ValueError, IndexError):
                    return pk
            aggregated = self._aggregate_quarterly_capex_to_annual()
            for period_key, value in aggregated.items():
                if period_key in metrics_data["CapEx"]:
                    continue
                fy_key = _fiscal_year_key(period_key)
                # Prefer putting value on an existing period for this FY (e.g. 2013-01-27 from other metrics)
                existing_for_fy = [p for p in all_periods if _fiscal_year_key(p) == fy_key]
                target = existing_for_fy[0] if existing_for_fy else period_key
                if target not in metrics_data["CapEx"]:
                    metrics_data["CapEx"][target] = value
                    if target == period_key:
                        all_periods.add(period_key)
                    log.debug(
                        "CashFlowStatement: CapEx period %s filled from summed quarterly facts",
                        target,
                    )

        # Convert to DataFrame
        if not metrics_data or not reported_metrics:
            return pd.DataFrame()

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
