# financial4all/financials/income_statement.py
"""
Income statement extraction and standardization.

This module provides functionality for extracting and standardizing
income statements from XBRL data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Set
from collections import defaultdict

from financial4all.xbrl.facts import FactSet, Fact
from financial4all.xbrl.periods import PeriodType, classify_fiscal_period
from financial4all.xbrl.standardization import get_synonym_groups, get_default_store
from financial4all.xbrl.calculations import CalculationEngine
from financial4all.core import log


class IncomeStatement:
    """
    Income statement extracted from XBRL data.

    This class handles extraction, standardization, and calculation
    of income statement metrics.
    """

    # Mapping from display names to normalized concept names in SynonymGroups
    # This allows us to use user-friendly display names while leveraging
    # the comprehensive synonym groups from the standardization system
    DISPLAY_NAME_TO_CONCEPT = {
        "Revenue": "revenue",
        "Cost of Revenue": "cost_of_revenue",
        "Gross Profit": "gross_profit",
        "R&D Expenses": "research_and_development",
        "SG&A Expenses": "sga_expense",
        "Operating Expenses": "operating_expenses",
        "Operating Income": "operating_income",
        "Interest Income": "interest_income",
        "Interest Expense": "interest_expense",
        "Other, net": "other_net",
        "Interest Income (Net)": "interest_income_net",
        # Note: "Other income (expense), net" is calculated, not extracted
        # It is calculated as: Interest Income + Interest Expense + Other, net
        # Note: "Income Before Taxes" is calculated, not extracted
        # It is calculated as: Operating Income + Other income (expense), net
        "Taxes": "income_tax_expense",
        "Net Income": "net_income",
        "Basic EPS": "earnings_per_share_basic",
        "Diluted EPS": "earnings_per_share_diluted",
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

    # Standard order for income statement metrics (matching user's reference)
    METRIC_ORDER = [
        "Revenue",  # Primary revenue metric (no duplicates)
        "Cost of Revenue",
        "Gross Profit",
        "R&D Expenses",
        "SG&A Expenses",
        "Operating Expenses",
        "Operating Income",
        "Interest Income",
        "Interest Expense",
        "Other, net",  # NVIDIA reports this
        "Other income (expense), net",  # Calculated: Interest Income + Interest Expense + Other, net
        "Interest Income (Net)",  # Only if separate Interest Income/Expense don't exist
        "Income Before Taxes",  # Calculated: Operating Income + Other income (expense), net
        "Taxes",
        "Net Income",
        "Basic EPS",
        "Diluted EPS",
    ]

    # Mapping of combined fields to their component fields
    # Used to detect when companies report combined fields instead of separate ones
    COMBINED_FIELD_MAPPING = {
        "InterestIncomeExpenseNet": {
            "components": ["Interest Income", "Interest Expense"],
            "description": "Net interest income/expense (Income - Expense)",
        },
        "InterestIncomeExpenseAfterProvisionForLoanLoss": {
            "components": ["Interest Income", "Interest Expense"],
            "description": "Net interest after loan loss provision",
        },
        "RevenuesNetOfInterestExpense": {
            "components": ["Revenue", "Interest Expense"],
            "description": "Revenue net of interest expense",
        },
    }

    def __init__(
        self, fact_set: FactSet, calculation_engine: Optional[CalculationEngine] = None
    ):
        """
        Initialize income statement from fact set.

        Args:
            fact_set: FactSet containing income statement facts
            calculation_engine: Optional calculation engine for deriving missing values
        """
        # Store original fact_set for broader searches
        # Use filter_annual() instead of filter_annual_10k() to capture more historical data
        # This includes annual data from 10-K, 10-K/A, and other form types
        self.fact_set = fact_set.filter_annual()
        self._original_fact_set = fact_set  # Keep original for debugging/discovery
        self.calculation_engine = calculation_engine or CalculationEngine()
        self.standardizer = get_default_store()
        self._dataframe: Optional[pd.DataFrame] = None

    @classmethod
    def from_company_facts(
        cls, company_facts: Dict[str, Any], cik: Optional[str] = None
    ) -> "IncomeStatement":
        """
        Create income statement from SEC company facts API response.

        Args:
            company_facts: Dictionary from SEC company facts API
            cik: Optional CIK for entity info extraction

        Returns:
            IncomeStatement object
        """
        fact_set = FactSet.from_company_facts(company_facts, cik=cik)
        return cls(fact_set)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert income statement to pandas DataFrame.

        Only includes metrics that have at least one reported value.
        Handles combined fields (e.g., InterestIncomeExpenseNet) intelligently.
        Filters out completely empty columns.

        Returns:
            DataFrame with standardized income statement metrics
        """
        if self._dataframe is not None:
            return self._dataframe

        # Extract metrics by standard name
        metrics_data = defaultdict(dict)
        reported_metrics = set()  # Track which metrics have at least one value

        # Step 1: Period-aware concept resolution
        # Collect ALL facts from ALL concepts, then resolve by period
        # This handles companies that change reporting formats across years
        # (e.g., AAPL using "Revenues" 2007-2017, then "RevenueFromContractWithCustomer" 2018+)

        # First pass: resolve all metrics with primary concepts
        # Resolve Operating Income first so we can use it for validation
        operating_income_concepts = self.STANDARD_MAPPING.get("Operating Income", [])
        if operating_income_concepts:
            resolved_data = self._resolve_concepts_by_period(
                "Operating Income", operating_income_concepts
            )
            if resolved_data:
                metrics_data["Operating Income"].update(resolved_data)
                reported_metrics.add("Operating Income")

        # Now resolve other metrics, passing Operating Income data for Interest Income validation
        for std_name, xbrl_concepts in self.STANDARD_MAPPING.items():
            if std_name == "Operating Income":
                continue  # Already resolved

            # Get all facts for this metric using period-aware resolution
            # Pass other metrics data for cross-validation (especially for Interest Income)
            resolved_data = self._resolve_concepts_by_period(
                std_name, xbrl_concepts, other_metrics_data=metrics_data
            )
            if resolved_data:
                metrics_data[std_name].update(resolved_data)
                reported_metrics.add(std_name)

        # Second pass: identify gaps and try fallback concepts
        # Collect all periods from successfully extracted metrics
        all_extracted_periods = set()
        for metric_data in metrics_data.values():
            all_extracted_periods.update(metric_data.keys())

        # For each metric, check if it's missing periods that other metrics have
        for std_name, xbrl_concepts in self.STANDARD_MAPPING.items():
            if std_name not in metrics_data:
                continue

            covered_periods = set(metrics_data[std_name].keys())
            # Only look for gaps if we have at least some data and other metrics have more periods
            if covered_periods and all_extracted_periods:
                missing_periods = all_extracted_periods - covered_periods

                # Try to discover fallback concepts for missing periods
                if missing_periods:
                    fallback_concepts = self._discover_fallback_concepts(
                        std_name, xbrl_concepts, missing_periods
                    )
                    if fallback_concepts:
                        # Try fallback concepts with very lenient filtering
                        fallback_data = self._resolve_concepts_by_period(
                            std_name, fallback_concepts
                        )
                        # Only add periods that were missing
                        for period_key, value in fallback_data.items():
                            if (
                                period_key in missing_periods
                                and period_key not in metrics_data[std_name]
                            ):
                                metrics_data[std_name][period_key] = value
                                log.debug(
                                    f"Added fallback data for {std_name} period {period_key} from concept discovery"
                                )

        # Step 2: Handle combined fields (only if separate components don't exist)
        # Pass metrics_data so we can check if separate components already exist
        combined_field_data = self._detect_and_handle_combined_fields(metrics_data)
        for std_name, period_data in combined_field_data.items():
            if period_data:  # Only add if we have data
                metrics_data[std_name].update(period_data)
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
        # Use local function references for performance
        df_data = {}
        df_data_setitem = df_data.__setitem__

        for std_name in reported_metrics:
            metric_data = metrics_data.get(std_name, {})
            metric_get = metric_data.get
            df_data_setitem(
                std_name, [metric_get(period, np.nan) for period in all_periods]
            )

        df = pd.DataFrame(df_data, index=all_periods)
        df.index.name = "end"

        # Add fiscal period classification if entity info is available
        if self._original_fact_set.entity_info:
            entity_info = self._original_fact_set.entity_info
            fy_end_month = entity_info.fiscal_year_end_month
            fy_end_day = entity_info.fiscal_year_end_day

            # Classify each period and add as metadata
            fiscal_years = []
            fiscal_periods = []

            for period_str in all_periods:
                # Find the corresponding fact to get Period object
                # We'll need to reconstruct the period from the string
                try:
                    from datetime import datetime
                    from financial4all.xbrl.periods import Period, PeriodType

                    # Parse period end date
                    end_date = datetime.strptime(period_str, "%Y-%m-%d").date()

                    # Try to find a fact with this end date to get the full period
                    # For now, we'll create a period assuming it's annual if we can't find it
                    period_obj = None
                    for fact in self._original_fact_set.facts:
                        if str(fact.period.end) == period_str:
                            period_obj = fact.period
                            break

                    # If we couldn't find the period, create a synthetic one
                    # Assume annual period (start = end - 365 days)
                    if period_obj is None:
                        from datetime import timedelta

                        start_date = end_date - timedelta(days=365)
                        period_obj = Period(
                            start=start_date,
                            end=end_date,
                            period_type=PeriodType.DURATION,
                        )

                    # Classify the period
                    fiscal_year, fiscal_period = classify_fiscal_period(
                        period_obj,
                        fiscal_year_end_month=fy_end_month,
                        fiscal_year_end_day=fy_end_day,
                    )

                    fiscal_years.append(fiscal_year)
                    fiscal_periods.append(fiscal_period)
                except Exception as e:
                    log.debug(f"Error classifying fiscal period for {period_str}: {e}")
                    fiscal_years.append(None)
                    fiscal_periods.append(None)

            # Add fiscal period information as DataFrame attributes
            df.attrs["fiscal_years"] = fiscal_years
            df.attrs["fiscal_periods"] = fiscal_periods
            df.attrs["fiscal_year_end"] = (
                f"{fy_end_month:02d}-{fy_end_day:02d}"
                if fy_end_month and fy_end_day
                else None
            )

        # Step 3: Filter out completely empty columns
        df = df.loc[:, ~df.isna().all()]

        # Step 3.5: Validate and normalize Interest Income/Expense
        # Some companies report Interest Expense as positive, but if they also report
        # Interest Income, we should ensure Interest Expense is always negative (it's an expense)
        # This ensures consistent accounting treatment: expenses reduce income

        # First, validate Interest Income - it should not match Operating Income values
        if "Interest Income" in df.columns and "Operating Income" in df.columns:
            interest_income_col = df["Interest Income"].copy()
            operating_income_col = df["Operating Income"]

            # Check each period where both values exist
            for idx in df.index:
                interest_val = interest_income_col.loc[idx]
                operating_val = operating_income_col.loc[idx]

                if (
                    pd.notna(interest_val)
                    and pd.notna(operating_val)
                    and operating_val != 0
                ):
                    # Calculate how similar the values are
                    ratio = (
                        abs(interest_val) / abs(operating_val)
                        if operating_val != 0
                        else 0
                    )
                    diff_ratio = (
                        abs(interest_val - operating_val) / abs(operating_val)
                        if operating_val != 0
                        else 1
                    )

                    # If Interest Income is suspiciously close to Operating Income (> 50% similar), it's likely misclassified
                    # Also check if they're within 10% of each other (very suspicious)
                    if ratio > 0.5 or diff_ratio < 0.1:
                        log.warning(
                            f"Interest Income ({interest_val}) is suspiciously similar to Operating Income "
                            f"({operating_val}) for period {idx} (ratio: {ratio:.1%}, diff: {diff_ratio:.1%}). "
                            f"Setting Interest Income to NaN as this is likely misclassified."
                        )
                        interest_income_col.loc[idx] = np.nan
                    # Also check if Interest Income is negative (shouldn't happen)
                    elif interest_val < 0:
                        log.warning(
                            f"Interest Income is negative for period {idx}: {interest_val}. "
                            f"Setting to NaN as this is likely misclassified."
                        )
                        interest_income_col.loc[idx] = np.nan

            df["Interest Income"] = interest_income_col

        if "Interest Expense" in df.columns and "Interest Income" in df.columns:
            # Check if Interest Expense has any non-null values
            if df["Interest Expense"].notna().any():
                # Ensure Interest Expense is negative (multiply positive values by -1)
                # Only modify non-null values that are positive (leave already-negative values as-is)
                interest_expense_col = df["Interest Expense"].copy()
                positive_mask = (
                    interest_expense_col > 0
                ) & interest_expense_col.notna()
                if positive_mask.any():
                    interest_expense_col[positive_mask] = -interest_expense_col[
                        positive_mask
                    ]
                    df["Interest Expense"] = interest_expense_col

        # Step 4: Add calculated fields that should always be present
        # Add "Other income (expense), net" if components exist (will be calculated)
        if "Other income (expense), net" not in df.columns:
            if any(
                col in df.columns
                for col in ["Interest Income", "Interest Expense", "Other, net"]
            ):
                df["Other income (expense), net"] = np.nan

        # Add "Income Before Taxes" if components exist (will be calculated)
        if "Income Before Taxes" not in df.columns:
            if (
                "Operating Income" in df.columns
                and "Other income (expense), net" in df.columns
            ):
                df["Income Before Taxes"] = np.nan

        # Apply calculations to fill missing values and calculate derived metrics
        df = self._apply_calculations(df)

        # Step 5: Final filter - remove any columns that became empty after calculations
        df = df.loc[:, ~df.isna().all()]

        # Step 6: Remove redundant metrics and reorder columns
        df = self._remove_redundant_metrics(df)
        df = self._reorder_dataframe_columns(df)

        self._dataframe = df
        return df

    def _get_all_facts_for_metric(
        self, xbrl_concepts: List[str]
    ) -> Dict[str, List[Fact]]:
        """
        Get all facts for a standard metric across all concept variations.

        Uses comprehensive fact discovery to find facts from all concepts,
        trying multiple namespace variations and filtering strategies.
        Also searches for synonym concepts using SynonymGroups if primary concepts don't yield enough data.

        Args:
            xbrl_concepts: List of XBRL concept names in priority order

        Returns:
            Dictionary mapping concept_name -> list of facts
        """
        all_facts_by_concept = {}

        for concept in xbrl_concepts:
            # Use comprehensive fact discovery
            facts = self._original_fact_set.get_all_facts_for_concept(
                concept, include_variants=True
            )

            if facts:
                all_facts_by_concept[concept] = facts

        # If we didn't find enough facts, try synonym discovery using SynonymGroups
        # This helps find alternative concepts that might be used
        # BUT: We need to be careful not to match concepts from other metric groups
        if not all_facts_by_concept and xbrl_concepts:
            primary_concept = xbrl_concepts[0]

            # First, try SynonymGroups for comprehensive synonym discovery
            synonym_groups = get_synonym_groups()
            concept_info = synonym_groups.identify_concept(primary_concept)

            if concept_info:
                # Found in SynonymGroups - get all synonyms from the SAME group only
                synonym_tags = concept_info.synonyms
                log.debug(
                    f"Found concept '{primary_concept}' in SynonymGroups as '{concept_info.name}' with {len(synonym_tags)} synonyms"
                )

                # Check which synonyms exist in the fact set
                all_concepts_in_factset = {
                    f.concept for f in self._original_fact_set.facts
                }

                # Get concepts from other groups to avoid cross-contamination
                # For Interest Income, we should NOT match Operating Income concepts
                excluded_concepts = set()
                if concept_info.name == "interest_income":
                    # Get Operating Income concepts to exclude
                    operating_income_group = synonym_groups.get_group(
                        "operating_income"
                    )
                    if operating_income_group:
                        for op_concept in operating_income_group.synonyms:
                            excluded_concepts.add(op_concept)
                            excluded_concepts.add(f"us-gaap_{op_concept}")
                            excluded_concepts.add(f"us-gaap:{op_concept}")

                for tag in synonym_tags:
                    # Try with and without namespace prefix
                    for variant in [tag, f"us-gaap_{tag}", f"us-gaap:{tag}"]:
                        if variant in excluded_concepts:
                            continue  # Skip concepts from other groups
                        if (
                            variant in all_concepts_in_factset
                            and variant not in xbrl_concepts
                        ):
                            synonym_facts = (
                                self._original_fact_set.get_all_facts_for_concept(
                                    variant, include_variants=True
                                )
                            )
                            if synonym_facts:
                                all_facts_by_concept[variant] = synonym_facts
                                log.debug(
                                    f"Found SynonymGroups synonym '{variant}' for '{primary_concept}' with {len(synonym_facts)} facts"
                                )

            # Fallback: Use pattern-based synonym discovery if SynonymGroups didn't help
            if not all_facts_by_concept:
                synonyms = self._original_fact_set.find_synonym_concepts(
                    primary_concept
                )

                for synonym in synonyms:
                    if synonym not in xbrl_concepts:  # Don't duplicate
                        synonym_facts = (
                            self._original_fact_set.get_all_facts_for_concept(
                                synonym, include_variants=True
                            )
                        )
                        if synonym_facts:
                            all_facts_by_concept[synonym] = synonym_facts
                            log.debug(
                                f"Found pattern-based synonym concept '{synonym}' for '{primary_concept}' with {len(synonym_facts)} facts"
                            )

        return all_facts_by_concept

    def _resolve_concepts_by_period(
        self,
        std_name: str,
        xbrl_concepts: List[str],
        other_metrics_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve which concept to use for each period using period-aware resolution.

        Strategy:
        1. Collect ALL facts from ALL concepts
        2. Apply multi-tier filtering to get best facts
        3. Group by period end date
        4. For each period, select best fact based on priority:
           - Concept priority (earlier in list = higher priority)
           - Form type (10-K preferred over 10-Q)
           - Unit (USD preferred)
           - Filing date (more recent preferred)
        5. Cross-validate with other metrics to prevent misclassification

        Args:
            std_name: Standardized metric name
            xbrl_concepts: List of XBRL concept names in priority order
            other_metrics_data: Dictionary of other already-resolved metrics (for validation)

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

            # Tier 1: Strict filter (annual 10-K, USD, no dimensions) - PREFERRED
            tier1_facts = [
                f
                for f in non_dimensional_facts
                if f.is_annual_10k() and (f.unit == "USD" or f.unit.startswith("USD"))
            ]

            # Tier 2: Lenient filter (annual, USD, no dimensions, any form)
            if not tier1_facts:
                tier2_facts = [
                    f
                    for f in non_dimensional_facts
                    if f.period.period_type == PeriodType.DURATION
                    and f.period.is_annual()
                    and (f.unit == "USD" or f.unit.startswith("USD"))
                ]
                filtered_facts_by_concept[concept] = tier2_facts
            else:
                filtered_facts_by_concept[concept] = tier1_facts

            # Tier 3: Very lenient (any annual period, USD, no dimensions) - fallback
            if not filtered_facts_by_concept[concept]:
                tier3_facts = [
                    f
                    for f in non_dimensional_facts
                    if f.period.period_type == PeriodType.DURATION
                    and f.period.is_annual()
                    and (f.unit == "USD" or f.unit.startswith("USD"))
                ]
                filtered_facts_by_concept[concept] = tier3_facts

            # Tier 4: Last resort - include dimensional facts if no non-dimensional found
            # Allow annual data from any form type (10-K, 10-K/A, etc.) to capture more historical data
            if not filtered_facts_by_concept[concept]:
                tier4_facts = [
                    f
                    for f in dimensional_facts
                    if f.period.period_type == PeriodType.DURATION
                    and f.period.is_annual()
                    and (f.unit == "USD" or f.unit.startswith("USD"))
                ]
                if tier4_facts:
                    # For dimensional facts, prefer those with common total/consolidated dimensions
                    # or those without segment-specific dimensions
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
        period_facts_map: Dict[str, List[Fact]] = {}

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
        resolved_data_setitem = resolved_data.__setitem__  # Local function reference

        for period_key, fact_candidates in period_facts_map.items():
            if not fact_candidates:
                continue

            # Sort by priority: concept priority > has_dimensions (prefer non-dimensional) > form > unit > filing date
            # Use tuple unpacking optimization
            fact_candidates.sort(
                key=lambda x: (
                    x[0],  # Concept priority (lower = higher priority)
                    0 if not x[1].dimensions else 1,  # Prefer non-dimensional facts
                    0 if x[1].form == "10-K" else 1,  # Prefer 10-K
                    0
                    if (x[1].unit == "USD" or x[1].unit.startswith("USD"))
                    else 1,  # Prefer USD
                    -(
                        x[1].filed.timestamp() if x[1].filed else float("-inf")
                    ),  # Prefer more recent (negated for descending)
                )
            )

            # Select best fact - try candidates in order until we find a valid one
            fact_value = None
            best_fact = None

            for candidate_idx, (concept_idx, fact) in enumerate(fact_candidates):
                candidate_value = fact.value
                is_valid = True

                # Cross-validate with other metrics to prevent misclassification
                # For Interest Income, check against Operating Income
                if std_name == "Interest Income" and other_metrics_data:
                    operating_income_data = other_metrics_data.get(
                        "Operating Income", {}
                    )
                    operating_val = operating_income_data.get(period_key)

                    if (
                        operating_val is not None
                        and isinstance(candidate_value, (int, float))
                        and isinstance(operating_val, (int, float))
                    ):
                        ratio = (
                            abs(candidate_value) / abs(operating_val)
                            if operating_val != 0
                            else 0
                        )
                        diff_ratio = (
                            abs(candidate_value - operating_val) / abs(operating_val)
                            if operating_val != 0
                            else 1
                        )

                        # If Interest Income is suspiciously similar to Operating Income, skip this candidate
                        if ratio > 0.5 or diff_ratio < 0.1:
                            log.debug(
                                f"Skipping Interest Income fact candidate {candidate_idx} for period {period_key}: "
                                f"value {candidate_value} is suspiciously similar to Operating Income {operating_val} "
                                f"(ratio: {ratio:.1%}, diff: {diff_ratio:.1%})"
                            )
                            is_valid = False

                if is_valid:
                    fact_value = candidate_value
                    best_fact = fact
                    break

            # If no valid fact found after validation, skip this period
            if fact_value is None or best_fact is None:
                log.debug(
                    f"No valid fact found for {std_name} period {period_key} after cross-validation"
                )
                continue

            # Validate fact value - check for obviously wrong values
            # For revenue and other income statement metrics, values should generally be reasonable
            if std_name in [
                "Revenue",
                "Gross Profit",
                "Operating Income",
                "Net Income",
            ]:
                if isinstance(fact_value, (int, float)):
                    # Check if value seems suspiciously small or wrong sign
                    # For revenue, should be positive and substantial (typically millions+)
                    is_suspicious = False

                    if std_name == "Revenue":
                        # Revenue should be positive and typically > 1 million
                        is_suspicious = fact_value < 0 or (
                            abs(fact_value) < 1e6 and fact_value != 0
                        )
                    elif std_name in ["Gross Profit", "Operating Income", "Net Income"]:
                        # These can be negative but shouldn't be suspiciously tiny
                        # If absolute value is < 1000, might be wrong unit or concept
                        is_suspicious = abs(fact_value) < 1000 and fact_value != 0

                    if is_suspicious:
                        log.warning(
                            f"Potentially incorrect {std_name} value for period {period_key}: "
                            f"{fact_value} from concept {best_fact.concept} "
                            f"(form={best_fact.form}, unit={best_fact.unit}, "
                            f"has_dimensions={bool(best_fact.dimensions)})"
                        )

                        # Try to find alternative fact for this period
                        # Look for facts with larger absolute values and no dimensions
                        alternative_facts = [
                            f
                            for f in fact_candidates[1:]
                            if isinstance(f[1].value, (int, float))
                            and not f[
                                1
                            ].dimensions  # Prefer non-dimensional alternatives
                            and abs(f[1].value)
                            > abs(fact_value) * 5  # At least 5x larger
                        ]

                        # If no non-dimensional alternatives, try any larger value
                        if not alternative_facts:
                            alternative_facts = [
                                f
                                for f in fact_candidates[1:]
                                if isinstance(f[1].value, (int, float))
                                and abs(f[1].value)
                                > abs(fact_value) * 10  # At least 10x larger
                            ]

                        if alternative_facts:
                            alt_fact = alternative_facts[0][1]
                            log.info(
                                f"Using alternative fact for {std_name} period {period_key}: "
                                f"{alt_fact.value} from {alt_fact.concept} "
                                f"(was {fact_value} from {best_fact.concept})"
                            )
                            fact_value = alt_fact.value

            resolved_data_setitem(period_key, fact_value)

        return resolved_data

    def _analyze_period_coverage(
        self, facts_by_period: Dict[str, List[Fact]]
    ) -> Dict[str, int]:
        """
        Analyze period coverage to identify gaps.

        Args:
            facts_by_period: Dictionary mapping period_key -> list of facts

        Returns:
            Dictionary with coverage statistics
        """
        if not facts_by_period:
            return {"total_periods": 0, "covered_periods": 0, "coverage_percent": 0.0}

        total_periods = len(facts_by_period)
        covered_periods = sum(1 for facts in facts_by_period.values() if facts)
        coverage_percent = (
            (covered_periods / total_periods * 100) if total_periods > 0 else 0.0
        )

        return {
            "total_periods": total_periods,
            "covered_periods": covered_periods,
            "coverage_percent": coverage_percent,
        }

    def _discover_fallback_concepts(
        self, std_name: str, existing_concepts: List[str], target_periods: Set[str]
    ) -> List[str]:
        """
        Discover alternative concepts to fill gaps in period coverage.

        Uses SynonymGroups and pattern matching to find concepts that might
        represent the same metric but weren't in the original mapping.

        Args:
            std_name: Standardized metric name
            existing_concepts: List of concepts already tried
            target_periods: Set of period keys that need data

        Returns:
            List of discovered alternative concept names
        """
        fallback_concepts = []
        synonym_groups = get_synonym_groups()

        # First, try to get all synonyms from SynonymGroups using the standardized name
        concept_name = self.DISPLAY_NAME_TO_CONCEPT.get(std_name)
        if concept_name:
            group = synonym_groups.get_group(concept_name)
            if group:
                # Get all synonyms from the group
                all_synonyms = group.synonyms
                log.debug(
                    f"Using SynonymGroups for '{std_name}' (concept: '{concept_name}') with {len(all_synonyms)} synonyms"
                )

                # Check which synonyms exist in fact set and cover missing periods
                all_concepts_in_factset = {
                    f.concept for f in self._original_fact_set.facts
                }

                for tag in all_synonyms:
                    # Try with and without namespace prefix
                    for variant in [tag, f"us-gaap_{tag}", f"us-gaap:{tag}"]:
                        if variant in existing_concepts:
                            continue  # Already tried

                        if variant not in all_concepts_in_factset:
                            continue

                        synonym_facts = (
                            self._original_fact_set.get_all_facts_for_concept(variant)
                        )
                        if not synonym_facts:
                            continue

                        # Check if this synonym covers any missing periods
                        synonym_periods = {
                            str(f.period.end)
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
                                f"Found SynonymGroups fallback concept '{variant}' for '{std_name}' covering periods: {missing_periods_covered}"
                            )

        # Fallback: Use pattern-based synonym detection for concepts not in SynonymGroups
        if not fallback_concepts:
            for concept in existing_concepts:
                synonyms = self._original_fact_set.find_synonym_concepts(concept)

                # Check if synonyms have data for missing periods
                for synonym in synonyms:
                    if synonym in existing_concepts:
                        continue  # Already tried

                    synonym_facts = self._original_fact_set.get_all_facts_for_concept(
                        synonym
                    )
                    if not synonym_facts:
                        continue

                    # Check if this synonym covers any missing periods
                    synonym_periods = {
                        str(f.period.end)
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
                            f"Found pattern-based fallback concept '{synonym}' for '{std_name}' covering periods: {missing_periods_covered}"
                        )

        return fallback_concepts

    def _detect_and_handle_combined_fields(
        self, metrics_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect and handle combined fields like InterestIncomeExpenseNet.

        Only adds combined fields if separate components don't exist in the extracted metrics.

        Args:
            metrics_data: Dictionary of already extracted metrics (to check for existing components)

        Returns:
            Dictionary mapping standard names to period-value dictionaries
        """
        combined_data = {}

        # Check if we already have separate Interest Income and/or Interest Expense in extracted metrics
        has_interest_income = (
            "Interest Income" in metrics_data and metrics_data["Interest Income"]
        )
        has_interest_expense = (
            "Interest Expense" in metrics_data and metrics_data["Interest Expense"]
        )

        # Also check fact_set as fallback (in case metrics weren't extracted but facts exist)
        if not has_interest_income:
            for concept in self.STANDARD_MAPPING.get("Interest Income", []):
                if self._original_fact_set.has_reported_data(concept):
                    has_interest_income = True
                    break

        if not has_interest_expense:
            for concept in self.STANDARD_MAPPING.get("Interest Expense", []):
                if self._original_fact_set.has_reported_data(concept):
                    has_interest_expense = True
                    break

        # Check for combined interest fields only if we don't have separate fields
        # AND we don't have "Other income (expense), net" calculated (which uses separate components)
        has_other_income_expense = (
            "Other income (expense), net" in metrics_data
            and metrics_data["Other income (expense), net"]
        )

        if (
            not has_interest_income
            and not has_interest_expense
            and not has_other_income_expense
        ):
            interest_net_concepts = [
                "InterestIncomeExpenseNet",
                "InterestIncomeExpenseAfterProvisionForLoanLoss",
            ]

            for net_concept in interest_net_concepts:
                facts = self.fact_set.get_by_concept(net_concept)
                if facts:
                    period_data = {}
                    for fact in facts:
                        if fact.unit == "USD" or fact.unit.startswith("USD"):
                            period_key = str(fact.period.end)
                            if period_key not in period_data:
                                period_data[period_key] = fact.value

                    if period_data:
                        # Store as "Interest Income (Net)"
                        combined_data["Interest Income (Net)"] = period_data
                        log.debug(f"Found combined interest field: {net_concept}")
                        break  # Use first found combined field

        # Check for RevenuesNetOfInterestExpense
        # Only use if Revenue is not already found
        has_revenue = False
        for concept in self.STANDARD_MAPPING.get("Revenue", []):
            if self._original_fact_set.has_reported_data(concept):
                has_revenue = True
                break

        if not has_revenue:
            revenues_net_concept = "RevenuesNetOfInterestExpense"
            facts = self.fact_set.get_by_concept(revenues_net_concept)
            if facts:
                period_data = {}
                for fact in facts:
                    if fact.unit == "USD" or fact.unit.startswith("USD"):
                        period_key = str(fact.period.end)
                        if period_key not in period_data:
                            period_data[period_key] = fact.value

                if period_data:
                    combined_data["Revenue"] = period_data
                    log.debug(
                        f"Found revenue net of interest expense: {revenues_net_concept}"
                    )

        return combined_data

    def _remove_redundant_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove redundant metrics (e.g., if Revenue exists, remove Revenues/SalesRevenueNet).
        Also removes "Interest Income (Net)" if separate Interest Income/Expense exist.

        Args:
            df: DataFrame with income statement data

        Returns:
            DataFrame with redundant metrics removed
        """
        # Define redundant metric groups (primary name first, alternatives after)
        redundant_groups = {
            "Revenue": [
                "Revenues",
                "SalesRevenueNet",
                "RevenueFromContractWithCustomer",
            ],
        }

        df_cleaned = df.copy()

        for primary, alternatives in redundant_groups.items():
            if primary in df_cleaned.columns:
                # Primary exists, remove alternatives
                for alt in alternatives:
                    if alt in df_cleaned.columns:
                        df_cleaned = df_cleaned.drop(columns=[alt])
                        log.debug(
                            f"Removed redundant metric '{alt}' (primary '{primary}' exists)"
                        )

        # Remove "Interest Income (Net)" if we have separate Interest Income and/or Interest Expense
        # and "Other income (expense), net" is calculated
        if "Interest Income (Net)" in df_cleaned.columns:
            has_separate_interest = (
                "Interest Income" in df_cleaned.columns
                or "Interest Expense" in df_cleaned.columns
            )
            has_other_income_expense = (
                "Other income (expense), net" in df_cleaned.columns
            )

            if has_separate_interest and has_other_income_expense:
                df_cleaned = df_cleaned.drop(columns=["Interest Income (Net)"])
                log.debug(
                    "Removed 'Interest Income (Net)' - separate components and calculated 'Other income (expense), net' exist"
                )

        return df_cleaned

    def _reorder_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder DataFrame columns according to standard income statement order.

        Args:
            df: DataFrame with income statement data

        Returns:
            DataFrame with reordered columns
        """
        # Get list of columns that exist in DataFrame
        existing_cols = list(df.columns)

        # Build ordered list: metrics in METRIC_ORDER that exist, then any extras
        ordered_cols = []
        for metric in self.METRIC_ORDER:
            if metric in existing_cols:
                ordered_cols.append(metric)

        # Add any remaining columns that weren't in METRIC_ORDER
        for col in existing_cols:
            if col not in ordered_cols:
                ordered_cols.append(col)

        # Reorder DataFrame
        return df[ordered_cols]

    def _apply_calculations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply calculation formulas to fill missing values and calculate derived metrics.

        Args:
            df: DataFrame with income statement data

        Returns:
            DataFrame with calculated values filled in
        """
        df_calc = df.copy()

        # Apply standard calculations
        # Gross Profit = Revenue - Cost of Revenue
        if "Gross Profit" in df_calc.columns:
            mask = df_calc["Gross Profit"].isna()
            if "Revenue" in df_calc.columns and "Cost of Revenue" in df_calc.columns:
                df_calc.loc[mask, "Gross Profit"] = (
                    df_calc.loc[mask, "Revenue"] - df_calc.loc[mask, "Cost of Revenue"]
                )

        # Operating Income = Gross Profit - Operating Expenses
        if "Operating Income" in df_calc.columns:
            mask = df_calc["Operating Income"].isna()
            if (
                "Gross Profit" in df_calc.columns
                and "Operating Expenses" in df_calc.columns
            ):
                df_calc.loc[mask, "Operating Income"] = (
                    df_calc.loc[mask, "Gross Profit"]
                    - df_calc.loc[mask, "Operating Expenses"]
                )

        # Other income (expense), net = Interest Income - Interest Expense + Other, net
        # IMPORTANT: Interest Expense is normalized to be negative (see Step 3.5),
        # so we ADD it (not subtract) because subtracting a negative = adding
        # Always calculate this field if components are available
        if (
            len(
                [
                    col
                    for col in ["Interest Income", "Interest Expense", "Other, net"]
                    if col in df_calc.columns
                ]
            )
            >= 2
        ):
            # Create or update the column
            if "Other income (expense), net" not in df_calc.columns:
                df_calc["Other income (expense), net"] = np.nan

            # Calculate: Interest Income + Interest Expense + Other, net
            # Note: Interest Expense is already negative (normalized), so we add it
            # Treating NaN as 0 for calculation
            other_income_expense = pd.Series(0.0, index=df_calc.index)

            # Add Interest Income (positive)
            if "Interest Income" in df_calc.columns:
                other_income_expense = other_income_expense + df_calc[
                    "Interest Income"
                ].fillna(0)

            # Add Interest Expense (already negative due to normalization, so adding = subtracting)
            if "Interest Expense" in df_calc.columns:
                other_income_expense = other_income_expense + df_calc[
                    "Interest Expense"
                ].fillna(0)

            # Add Other, net (can be positive or negative)
            if "Other, net" in df_calc.columns:
                other_income_expense = other_income_expense + df_calc[
                    "Other, net"
                ].fillna(0)

            # Update all values (calculated field, so always recalculate)
            df_calc["Other income (expense), net"] = other_income_expense

        # Income Before Taxes = Operating Income + Other income (expense), net
        # Always calculate this field if components are available
        if "Operating Income" in df_calc.columns:
            # Create or update the column
            if "Income Before Taxes" not in df_calc.columns:
                df_calc["Income Before Taxes"] = np.nan

            # Calculate: Operating Income + Other income (expense), net
            # Prefer "Other income (expense), net" if it exists, otherwise try "Interest Income (Net)"
            other_income_component = None

            if "Other income (expense), net" in df_calc.columns:
                other_income_component = df_calc["Other income (expense), net"]
            elif "Interest Income (Net)" in df_calc.columns:
                # Fallback to Interest Income (Net) if Other income (expense), net doesn't exist
                other_income_component = df_calc["Interest Income (Net)"]

            if other_income_component is not None:
                # Calculate: Operating Income + Other income component
                # Treating NaN as 0 for calculation
                income_before_taxes = df_calc["Operating Income"].fillna(
                    0
                ) + other_income_component.fillna(0)

                # Update all values (calculated field, so always recalculate)
                df_calc["Income Before Taxes"] = income_before_taxes

        # Net Income = Income Before Taxes - Taxes
        if "Net Income" in df_calc.columns:
            mask = df_calc["Net Income"].isna()
            if "Income Before Taxes" in df_calc.columns and "Taxes" in df_calc.columns:
                df_calc.loc[mask, "Net Income"] = (
                    df_calc.loc[mask, "Income Before Taxes"]
                    - df_calc.loc[mask, "Taxes"]
                )

        return df_calc

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

        # Sort by index (date) descending to get most recent first
        df_sorted = df.sort_index(ascending=False)

        value = df_sorted.iloc[period_offset][metric_name]

        if pd.isna(value):
            return None

        return float(value)
