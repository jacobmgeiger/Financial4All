# financial4all/financials/income_statement.py
"""
Income statement extraction and standardization from XBRL.

IncomeStatement is built from a FactSet (e.g. from SEC company facts). It maps
standardized display names (Revenue, Cost of Revenue, Gross Profit, R&D, SG&A,
Operating Income, etc.) to XBRL concepts via SynonymGroups, uses period-aware
resolution and fallback concept discovery, applies calculations (e.g. Gross
Profit, Operating Income), removes redundant metrics (e.g. Operating Expenses
when R&D+SG&A exist), and returns a period-indexed DataFrame in METRIC_ORDER.

EdgarTools compatibility:
    to_dataframe() supports presentation, show_date_range, label_column,
    include_standardization, and view="summary"|"standard"|"detailed" for
    alignment with EdgarTools display conventions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Set
from collections import defaultdict
from datetime import date as date_type, datetime

from financial4all.xbrl.facts import FactSet, Fact
from financial4all.xbrl.periods import PeriodType, classify_fiscal_period
from financial4all.xbrl.fact_resolution import sort_fact_candidates_by_priority
from financial4all.xbrl.structural_filter import is_xbrl_structural_element
from financial4all.xbrl.standardization import (
    get_synonym_groups,
    get_default_store,
    get_display_name as get_std_display_name,
    _load_company_tags_by_display,
)
from financial4all.xbrl.standardization.reverse_index import get_reverse_index
from financial4all.xbrl.calculations import CalculationEngine
from financial4all.xbrl.dimension_classifier import is_breakdown_dimension
from financial4all.core import log


class IncomeStatement:
    """
    Income statement built from XBRL facts with standardized metric names and calculations.

    Uses DISPLAY_NAME_TO_CONCEPT and SynonymGroups; supports period-aware resolution,
    fallback concept discovery, and derived metrics (Gross Profit, Operating Income,
    Other income (expense) net). Redundant columns (e.g. Operating Expenses) are
    dropped when components exist. from_company_facts() builds from SEC company
    facts; to_dataframe() returns a period-indexed DataFrame in METRIC_ORDER.
    """

    # Mapping from display names to concept names in SynonymGroups.
    # Aligned with EdgarTools concept_mappings.json (concept names = normalized display labels).
    DISPLAY_NAME_TO_CONCEPT = {
        "Revenue": "revenue",
        "Cost of Revenue": "total_cost_of_revenue",
        "Gross Profit": "gross_profit",
        "R&D Expenses": "research_and_development_expense",
        "SG&A Expenses": "selling_general_and_administrative_expense",
        "General and Administrative Expense": "general_and_administrative_expense",
        "Selling and Marketing Expense": "selling_expense",
        "Operating Expenses": "total_operating_expenses",
        "Restructuring and other charges": "restructuring_expense",
        "Other Operating Expense": "other_operating_expense",
        "Asset Impairment Charges": "goodwill_impairment",
        "Operating Income": "operating_income",
        "Interest Income": "interest_income",
        "Interest Expense": "interest_expense",
        "Other, net": "nonoperating_income_expense",
        "Interest Income (Net)": "interest_expense",
        "Income Before Taxes": "income_before_tax",
        "Taxes": "income_tax_expense",
        "Net Income": "net_income",
        "Outstanding Shares Basic": "shares_outstanding_basic",
        "Outstanding Shares Diluted": "shares_outstanding_diluted",
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
                mapping[display_name] = list(group.synonyms)
            else:
                log.warning(
                    f"Concept '{concept_name}' not found in SynonymGroups for '{display_name}'"
                )
                mapping[display_name] = []

        # Merge company-specific tags (e.g. NVDA revenue/cost variants)
        company_tags = _load_company_tags_by_display()
        for display_name, extra_tags in company_tags.items():
            if display_name in mapping and extra_tags:
                seen = set(mapping[display_name])
                for t in extra_tags:
                    if t not in seen:
                        seen.add(t)
                        mapping[display_name].append(t)

        cls._STANDARD_MAPPING_CACHE = mapping
        return mapping

    @property
    def STANDARD_MAPPING(self) -> Dict[str, List[str]]:
        """Get standard mapping (backward compatibility)."""
        return self._get_standard_mapping()

    # Metrics that must have numeric values only (no HTML, strings, or disclosure text).
    # Prevents R&D and other cells from showing raw HTML or policy text.
    NUMERIC_METRICS: Set[str] = {
        "Revenue",
        "Cost of Revenue",
        "Gross Profit",
        "R&D Expenses",
        "SG&A Expenses",
        "General and Administrative Expense",
        "Selling and Marketing Expense",
        "Operating Expenses",
        "Restructuring and other charges",
        "Other Operating Expense",
        "Asset Impairment Charges",
        "Operating Income",
        "Interest Income",
        "Interest Expense",
        "Other, net",
        "Other income (expense), net",
        "Interest Income (Net)",
        "Income Before Taxes",
        "Taxes",
        "Net Income",
        "Outstanding Shares Basic",
        "Outstanding Shares Diluted",
        "Basic EPS",
        "Diluted EPS",
    }

    # Metrics that support segment/breakdown extraction in detailed view.
    SEGMENT_METRICS: Set[str] = frozenset({"Revenue", "Cost of Revenue"})

    # Redundant metrics: preferred name -> alternates to collapse (EdgarTools-style deduplication)
    # When both exist with same/similar data, keep preferred and drop alternate.
    DUPLICATE_METRIC_GROUPS: Dict[str, List[str]] = {
        "R&D Expenses": ["Research and Development Expense"],
        "SG&A Expenses": ["Selling, General and Administrative Expense"],
        "Income Tax Expense": ["Taxes"],
    }

    # Concepts that typically use totalLabel in presentation (EdgarTools parity).
    # When multiple facts exist for the same period, prefer facts from these concepts.
    IS_TOTAL_CONCEPTS: Set[str] = frozenset({
        "OperatingExpenses", "OperatingCostsAndExpenses",
        "SellingGeneralAndAdministrativeExpense",
        "CostOfRevenue", "CostOfGoodsSold", "CostOfGoodsAndServicesSold", "CostOfSales",
        "GrossProfit", "OperatingIncomeLoss", "OperatingIncome",
        "IncomeLossBeforeIncomeTaxes", "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
        "IncomeTaxExpenseBenefit", "IncomeTaxExpenseBenefitContinuingOperations",
        "NetIncomeLoss", "NetIncome",
        "Revenues", "Revenue", "SalesRevenueNet", "RevenueFromContractWithCustomer",
    })

    # Standard order for income statement metrics (matching user's reference).
    # Operating Expenses omitted: it is R&D + SG&A and redundant for display.
    METRIC_ORDER = [
        "Revenue",  # Primary revenue metric (no duplicates)
        "Cost of Revenue",
        "Gross Profit",
        "R&D Expenses",
        "SG&A Expenses",
        "General and Administrative Expense",
        "Selling and Marketing Expense",
        "Restructuring and other charges",
        "Other Operating Expense",
        "Asset Impairment Charges",
        "Operating Income",
        "Interest Income",
        "Interest Expense",
        "Other, net",  # NVIDIA reports this
        "Other income (expense), net",  # Extract first, calculate if not available: Interest Income + Interest Expense + Other, net
        "Interest Income (Net)",  # Only if separate Interest Income/Expense don't exist
        "Income Before Taxes",  # Extract first, calculate if not available: Operating Income + Other income (expense), net
        "Taxes",
        "Income Tax Expense",  # Preferred over Taxes when both exist (dedup)
        "Net Income",
        "Outstanding Shares Basic",
        "Outstanding Shares Diluted",
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
        self,
        fact_set: Optional[FactSet] = None,
        calculation_engine: Optional[CalculationEngine] = None,
        _presentation_dataframe: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize income statement from fact set or presentation-tree data.

        Args:
            fact_set: FactSet containing income statement facts (None when using
                presentation-tree path)
            calculation_engine: Optional calculation engine for deriving missing values
            _presentation_dataframe: Pre-built DataFrame from XBRL presentation tree
                (EdgarTools parity); when set, to_dataframe returns this
        """
        self._presentation_dataframe = _presentation_dataframe
        if fact_set is None:
            fact_set = FactSet(facts=[], entity_info=None)
        self.fact_set = fact_set.filter_annual()
        self._original_fact_set = fact_set
        self.calculation_engine = calculation_engine or CalculationEngine()
        self.standardizer = get_default_store()
        self._dataframe: Optional[pd.DataFrame] = None

    # Fallback display names for common unmapped income-statement concepts
    _UNMAPPED_DISPLAY_NAMES: Dict[str, str] = {
        "EarningsPerShareBasic": "Basic EPS",
        "EarningsPerShareDiluted": "Diluted EPS",
        "WeightedAverageNumberOfSharesOutstandingBasic": "Weighted Average Shares Outstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding": "Weighted Average Shares Outstanding, Diluted",
        "BusinessCombinationAdvancedConsiderationWrittenOff": "Business Combination Consideration Written Off",
    }

    @classmethod
    def _clean_unmapped_display_name(cls, concept: str) -> str:
        """
        Produce a readable display name for unmapped XBRL concepts.
        Uses fallback map for common concepts, else strips prefix and title-cases.
        """
        if not concept:
            return ""
        local = concept.split("_")[-1] if "_" in concept else concept
        return cls._UNMAPPED_DISPLAY_NAMES.get(
            local, local.replace("_", " ").replace("-", " ").title()
        )

    @classmethod
    def _dataframe_from_line_items(
        cls,
        line_items: List[Dict[str, Any]],
        reporting_periods: List[Dict[str, Any]],
        standardizer: Any,
    ) -> Optional[pd.DataFrame]:
        """
        Build DataFrame from presentation-tree line items (EdgarTools parity).

        Returns DataFrame with index=periods, columns=metrics to match app expectation.
        Filters to duration (annual) periods only for income statement.

        Args:
            line_items: From xbrl.get_statement('IncomeStatement')
            reporting_periods: From xbrl.reporting_periods
            standardizer: MappingStore for concept -> display name

        Returns:
            DataFrame with index=periods, columns=metrics (or None if empty)
        """
        if not line_items:
            return None

        # Collect period keys and labels - filter to duration only, prefer annual (330-400 days)
        raw_periods = reporting_periods or []
        period_tuples = []
        for p in raw_periods:
            if p.get("type") == "instant":
                continue
            key = p.get("key", "")
            if not key.startswith("duration_"):
                continue
            # Parse duration key (duration_YYYY-MM-DD_YYYY-MM-DD) to compute span
            parts = key.replace("duration_", "").split("_")
            if len(parts) >= 2:
                try:
                    start_d = datetime.strptime(parts[0], "%Y-%m-%d").date()
                    end_d = datetime.strptime(parts[1], "%Y-%m-%d").date()
                    days = (end_d - start_d).days
                    if days < 330 or days > 400:
                        continue  # Skip quarterly and irregular periods
                except (ValueError, IndexError):
                    pass
            label = p.get("label", p.get("end_date", key))
            period_tuples.append((key, label))
        # Fallback: if no annual periods found, include all duration periods
        if not period_tuples:
            for p in raw_periods:
                if p.get("type") == "instant":
                    continue
                key = p.get("key", "")
                if key.startswith("duration_"):
                    label = p.get("label", p.get("end_date", key))
                    period_tuples.append((key, label))
        if not period_tuples:
            all_keys = set()
            for item in line_items:
                for k in (item.get("values") or {}).keys():
                    if k.startswith("duration_"):
                        all_keys.add(k)
            period_tuples = [(k, k) for k in sorted(all_keys, reverse=True)]

        # Build rows: one per metric, columns = periods
        rows = []
        seen_labels = set()

        for item in line_items:
            values = item.get("values") or {}
            if not values:
                continue
            concept = item.get("concept", "")
            label = item.get("label", "")
            # Context for disambiguation (EdgarTools parity: is_total, section, calculation_parent)
            context = {
                "statement_type": "IncomeStatement",
                "label": label,
                "is_total": item.get("is_total", False),
                "section": item.get("section"),
                "calculation_parent": item.get("calculation_parent"),
            }
            display_name = get_std_display_name(concept, context) if concept else None
            if not display_name and hasattr(standardizer, "get_standard_name"):
                standard_concept = standardizer.get_standard_name(concept)
                display_name = (
                    get_std_display_name(standard_concept, context)
                    if standard_concept
                    else cls._clean_unmapped_display_name(standard_concept)
                )
            row_label = display_name or cls._clean_unmapped_display_name(concept) or label or concept
            if not row_label or row_label in seen_labels:
                continue
            seen_labels.add(row_label)
            row_data = {"Metric": row_label}
            for period_key, period_label in period_tuples:
                val = values.get(period_key)
                row_data[period_label] = val
            rows.append(row_data)

        if not rows:
            return None
        df = pd.DataFrame(rows)
        df = df.set_index("Metric")
        # App expects index=periods, columns=metrics
        df = df.T
        df.index.name = "end"

        # Recalculate Operating Income = Gross Profit - Total Operating Expenses
        # XBRL can report wrong Operating Income (e.g., segment/discontinued-ops values)
        gp_col = "Gross Profit" if "Gross Profit" in df.columns else None
        oi_col = "Operating Income" if "Operating Income" in df.columns else None
        opex_col = "Total Operating Expenses" if "Total Operating Expenses" in df.columns else "Operating Expenses" if "Operating Expenses" in df.columns else None
        component_cols = [
            c for c in [
                "Research and Development Expense", "R&D Expenses",
                "Selling, General and Administrative Expense", "SG&A Expenses",
                "Restructuring and other charges", "Other Operating Expense",
                "Asset Impairment Charges", "Business Combination Consideration Written Off",
            ]
            if c in df.columns
        ]
        if gp_col and oi_col:
            for idx in df.index:
                gp = df.loc[idx, gp_col]
                if pd.isna(gp):
                    continue
                opex = None
                if opex_col and pd.notna(df.loc[idx, opex_col]):
                    opex = float(df.loc[idx, opex_col])
                elif component_cols:
                    opex = sum(
                        float(df.loc[idx, c]) if pd.notna(df.loc[idx, c]) else 0
                        for c in component_cols
                    )
                if opex is not None:
                    try:
                        df.loc[idx, oi_col] = float(gp) - opex
                    except (TypeError, ValueError):
                        pass

        # Recalculate Taxes = Income Before Tax - Net Income (ensures equation holds)
        tax_col = "Taxes" if "Taxes" in df.columns else "Income Tax Expense" if "Income Tax Expense" in df.columns else None
        ibt_col = "Income Before Taxes" if "Income Before Taxes" in df.columns else "Income Before Tax" if "Income Before Tax" in df.columns else None
        ni_col = "Net Income" if "Net Income" in df.columns else None
        if tax_col and ibt_col and ni_col:
            for idx in df.index:
                ibt = df.loc[idx, ibt_col]
                ni = df.loc[idx, ni_col]
                if pd.notna(ibt) and pd.notna(ni):
                    try:
                        df.loc[idx, tax_col] = float(ibt) - float(ni)
                    except (TypeError, ValueError):
                        pass

        return df

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

    @classmethod
    def from_filing(
        cls, filing: Any, client: Optional[Any] = None
    ) -> "IncomeStatement":
        """
        Create income statement from a SEC filing (presentation-tree or fact-centric).

        When XBRL has presentation linkbases, uses presentation-tree driven extraction
        (EdgarTools parity). Falls back to fact-centric extraction otherwise.

        Args:
            filing: Filing object with get_xbrl_content()
            client: Optional SECClient for fetching XBRL content

        Returns:
            IncomeStatement object
        """
        from financial4all.xbrl.xbrl import XBRL

        xbrl = XBRL.from_filing(filing, client=client)

        # EdgarTools parity: prefer presentation-tree path when linkbases available
        if xbrl.presentation_trees:
            line_items = xbrl.get_statement("IncomeStatement")
            if line_items:
                df = cls._dataframe_from_line_items(
                    line_items, xbrl.reporting_periods, get_default_store()
                )
                if df is not None and not df.empty:
                    return cls(
                        fact_set=None,
                        _presentation_dataframe=df,
                    )

        # Fallback: fact-centric extraction
        fact_set = FactSet.from_xbrl_instance(
            xbrl, cik=getattr(filing, "cik", None)
        )
        return cls(fact_set)

    @classmethod
    def from_filings(
        cls,
        filings: List[Any],
        client: Optional[Any] = None,
    ) -> "IncomeStatement":
        """
        Create income statement from multiple SEC filings (multi-year extraction).

        Parses XBRL from each filing, merges FactSets (preferring most recent
        filing for overlapping periods), and builds a unified statement.

        Args:
            filings: List of Filing objects (most recent first)
            client: Optional SECClient for fetching XBRL content

        Returns:
            IncomeStatement object with merged multi-year data
        """
        from financial4all.xbrl.xbrl import XBRL

        fact_sets: List[FactSet] = []
        for filing in filings:
            try:
                xbrl = XBRL.from_filing(filing, client=client)
                fs = FactSet.from_xbrl_instance(
                    xbrl, cik=getattr(filing, "cik", None)
                )
                if fs.facts:
                    fact_sets.append(fs)
            except Exception as e:
                log.debug(f"Skipping filing {getattr(filing, 'accession_number', '?')}: {e}")
                continue

        if not fact_sets:
            raise ValueError("No XBRL data could be extracted from any filing")
        merged = FactSet.merge(fact_sets, prefer_most_recent=True)
        return cls(merged)

    def supplement_from_company_facts(
        self, company_facts: Dict[str, Any], cik: Optional[str] = None
    ) -> None:
        """
        Fill gaps in this statement by merging facts from SEC company facts API.

        Only adds facts for (concept, period_end) where we have no value.
        Filing-sourced facts remain primary (EdgarTools-style hybrid extraction).

        Args:
            company_facts: Dictionary from SEC company facts API
            cik: Optional CIK for entity info extraction
        """
        cf_fs = FactSet.from_company_facts(company_facts, cik=cik)
        cf_annual = cf_fs.filter_annual()
        self._original_fact_set.supplement_from(cf_annual)
        self.fact_set = self._original_fact_set.filter_annual()
        self._dataframe = None  # Invalidate cache so to_dataframe() reflects new facts

    # Flow metrics that can be gap-filled by summing 4 quarters when fill_gaps_from_10q=True
    _10Q_FILL_METRICS: Set[str] = {
        "Revenue", "Cost of Revenue", "R&D Expenses", "SG&A Expenses",
        "General and Administrative Expense", "Selling and Marketing Expense",
        "Restructuring and other charges", "Other Operating Expense",
        "Asset Impairment Charges", "Operating Expenses", "Operating Income",
        "Interest Income",
        "Interest Expense", "Other, net", "Interest Income (Net)",
        "Income Before Taxes", "Taxes", "Net Income",
    }

    def _get_depreciation_by_period(self) -> Dict[str, float]:
        """
        Get Depreciation and Amortization (D&A) by period for SG&A derivation.

        Used when deriving SG&A from the income statement equation for periods where
        SG&A is not reported (e.g. ANF pre-2023). D&A is typically reported on the
        cash flow statement; the fact set includes all company facts.

        Returns:
            Dict mapping period_key (YYYY-MM-DD) -> D&A value
        """
        synonyms = get_synonym_groups()
        group = synonyms.get_group("depreciation_and_amortization") if synonyms else None
        concepts = list(group.synonyms) if group else ["DepreciationAndAmortization", "Depreciation"]
        out: Dict[str, float] = {}
        for concept in concepts:
            facts = self._original_fact_set.get_all_facts_for_concept(
                concept, include_variants=True
            )
            for fact in facts:
                if fact.period.period_type != PeriodType.DURATION or not fact.period.start:
                    continue
                if fact.unit != "USD" and not (fact.unit or "").startswith("USD"):
                    continue
                if not isinstance(fact.value, (int, float)):
                    continue
                if not fact.period.is_annual():
                    continue
                period_key = str(fact.period.end)
                if period_key not in out:
                    out[period_key] = float(fact.value)
        return out

    def _aggregate_quarterly_to_annual(
        self, concepts: List[str]
    ) -> Dict[str, Any]:
        """
        Sum quarterly facts into annual values for fiscal years lacking 10-K annual facts.

        When fill_gaps_from_10q is True, use this to fill gaps for flow metrics
        (Revenue, Cost of Revenue, etc.) from 4 consecutive 10-Q quarters.

        Returns:
            Dict mapping period_key (YYYY-MM-DD) -> annual value from summed quarters
        """
        out: Dict[str, Any] = {}
        if not concepts:
            return out
        by_fy: Dict[str, List[tuple]] = defaultdict(list)
        for concept in concepts:
            facts = self._original_fact_set.get_all_facts_for_concept(
                concept, include_variants=True
            )
            for fact in facts:
                if fact.period.period_type != PeriodType.DURATION or not fact.period.start:
                    continue
                if fact.unit != "USD" and not (fact.unit and str(fact.unit).startswith("USD")):
                    continue
                try:
                    val = float(fact.value)
                except (ValueError, TypeError):
                    continue
                s, e = fact.period.start, fact.period.end
                start_d = s if isinstance(s, date_type) else date_type.fromisoformat(str(s)[:10])
                end_d = e if isinstance(e, date_type) else date_type.fromisoformat(str(e)[:10])
                days = (end_d - start_d).days
                y, m = end_d.year, end_d.month
                fy_key = str(end_d) if m <= 3 else f"{y + 1}-01-31"
                by_fy[fy_key].append((end_d, days, val))
        for fy_key, candidates in by_fy.items():
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
            annual = next((v for _e, d, v in unique if 330 <= d <= 400), None)
            if annual is not None:
                out[fy_key] = annual
                continue
            quarters = [(e, d, v) for e, d, v in unique if 80 <= d <= 100]
            if len(quarters) >= 4:
                by_end: Dict[date_type, float] = {}
                for end_d, _d, v in sorted(quarters, key=lambda x: x[0]):
                    by_end[end_d] = v
                if len(by_end) >= 4:
                    out[fy_key] = sum(by_end.values())
        return out

    def to_dataframe(
        self,
        view: str = "standard",
        presentation: Optional[bool] = None,
        show_date_range: bool = False,
        label_column: Optional[str] = None,
        include_standardization: bool = False,
    ) -> pd.DataFrame:
        """
        Convert income statement to pandas DataFrame.

        Only includes metrics that have at least one reported value.
        Handles combined fields (e.g., InterestIncomeExpenseNet) intelligently.
        Filters out completely empty columns.

        Args:
            view: Output level. "standard" (default): no segment rows.
                  "summary": totals only (EdgarTools SUMMARY). "detailed": include
                  segment breakdown rows (e.g., Revenue — Americas) inline.
            presentation: When True, apply sign transformation (expenses positive).
                         When False, skip. When None, use config.apply_presentation_signs.
                         EdgarTools parity.
            show_date_range: When True, duration period labels use "start to end".
                            Single-filing path supports this; multi-period uses end_date only.
            label_column: Name for row identifier column when transposed. "Metric" (default)
                          or "label" for EdgarTools API parity.
            include_standardization: When True, attach standard_concept map. See
                                   get_standard_concept_map() for cross-company use.

        Returns:
            DataFrame with standardized income statement metrics
        """
        # EdgarTools parity: use pre-built DataFrame from presentation-tree path
        # (_presentation_dataframe is source data, not cache - always use when present)
        _use_cache = True
        try:
            from financial4all.config import get_config
            _use_cache = not get_config().disable_statement_cache
        except Exception:
            pass
        if self._presentation_dataframe is not None:
            df = self._presentation_dataframe.copy()
            # Apply presentation signs if requested (presentation param overrides config)
            try:
                from financial4all.config import get_config
                _apply = presentation if presentation is not None else get_config().apply_presentation_signs
            except Exception:
                _apply = presentation if presentation is not None else False
            if _apply:
                from financial4all.xbrl.presentation import apply_presentation
                df = apply_presentation(df, "IncomeStatement")
            # show_date_range: when False, use end date only for period index (EdgarTools parity)
            if not show_date_range and not df.empty:
                import re
                new_index = []
                for idx in df.index:
                    s = str(idx)
                    m = re.match(r"\d{4}-\d{2}-\d{2} to (\d{4}-\d{2}-\d{2})", s)
                    new_index.append(m.group(1) if m else s)
                df.index = new_index
            if include_standardization:
                df.attrs["standard_concept_map"] = self.get_standard_concept_map(df)
            if label_column is not None:
                df.attrs["label_column"] = label_column
            return df

        if _use_cache and self._dataframe is not None and view in ("standard", "summary"):
            df = self._dataframe.copy()
            try:
                from financial4all.config import get_config
                _apply = presentation if presentation is not None else get_config().apply_presentation_signs
            except Exception:
                _apply = presentation if presentation is not None else False
            if _apply:
                from financial4all.xbrl.presentation import apply_presentation
                df = apply_presentation(df, "IncomeStatement")
            if include_standardization:
                df.attrs["standard_concept_map"] = self.get_standard_concept_map(df)
            if label_column is not None:
                df.attrs["label_column"] = label_column
            return df

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

        # Step 1a2: 10-Q summation for gaps (configurable; off by default)
        try:
            from financial4all.config import get_config
            fill_10q = get_config().fill_gaps_from_10q
        except Exception:
            fill_10q = False
        if fill_10q:
            def _fiscal_year_key(pk: str) -> str:
                try:
                    y, m = int(pk[:4]), int(pk[5:7])
                    return f"{y}-01" if m <= 3 else f"{y + 1}-01"
                except (ValueError, IndexError):
                    return pk
            for std_name in self._10Q_FILL_METRICS:
                if std_name not in metrics_data:
                    continue
                covered = set(metrics_data[std_name].keys())
                missing = all_extracted_periods - covered
                if not missing:
                    continue
                concepts = self.STANDARD_MAPPING.get(std_name, [])
                if not concepts:
                    continue
                aggregated = self._aggregate_quarterly_to_annual(concepts)
                for period_key, value in aggregated.items():
                    if period_key in metrics_data[std_name]:
                        continue
                    fy_key = _fiscal_year_key(period_key)
                    existing_for_fy = [p for p in all_extracted_periods if _fiscal_year_key(p) == fy_key]
                    target = existing_for_fy[0] if existing_for_fy else period_key
                    if target in missing or target not in metrics_data[std_name]:
                        metrics_data[std_name][target] = value
                        if target == period_key:
                            all_extracted_periods.add(period_key)
                        log.debug(
                            f"IncomeStatement: {std_name} period {target} filled from summed 10-Q"
                        )

        # Step 1b: Revenue/COGS swap detection
        # For retailers, Revenue should exceed Cost of Revenue. If Revenue < COGS for a period,
        # concepts may be swapped (e.g., store count vs revenue). Swap values and log.
        if "Revenue" in metrics_data and "Cost of Revenue" in metrics_data:
            rev_keys = set(metrics_data["Revenue"].keys())
            cogs_keys = set(metrics_data["Cost of Revenue"].keys())
            for period_key in rev_keys & cogs_keys:
                rev = metrics_data["Revenue"][period_key]
                cog = metrics_data["Cost of Revenue"][period_key]
                if isinstance(rev, (int, float)) and isinstance(cog, (int, float)):
                    if rev > 0 and cog > 0 and rev < cog:
                        log.warning(
                            f"Revenue/COGS swap detected for period {period_key}: "
                            f"Revenue={rev:,.0f} < Cost of Revenue={cog:,.0f}. Swapping values."
                        )
                        metrics_data["Revenue"][period_key] = cog
                        metrics_data["Cost of Revenue"][period_key] = rev

        # Step 1c: SG&A from components when total not reported
        # Retailers (e.g. ANF) report GeneralAndAdministrativeExpense + SellingExpense separately;
        # derive SG&A = G&A + Selling for periods where total is missing
        gna = metrics_data.get("General and Administrative Expense", {})
        sm = metrics_data.get("Selling and Marketing Expense", {})
        sga = metrics_data.get("SG&A Expenses", {})
        for period_key in set(gna) | set(sm):
            if period_key not in sga or (isinstance(sga.get(period_key), (int, float)) and pd.isna(sga[period_key])):
                ga_val = gna.get(period_key)
                sm_val = sm.get(period_key)
                if isinstance(ga_val, (int, float)) and isinstance(sm_val, (int, float)):
                    combined = ga_val + sm_val
                    metrics_data.setdefault("SG&A Expenses", {})[period_key] = combined
                    reported_metrics.add("SG&A Expenses")
                    log.debug(
                        f"Derived SG&A ({combined:,.0f}) from G&A + Selling for period {period_key}"
                    )

        # Step 1d: SG&A from income statement equation when not reported at all
        # Some companies (e.g. ANF pre-2023) reported total SG&A or G&A+Selling only in recent
        # years; older filings use different tags. Derive SG&A = Gross Profit - Operating Income
        # - Restructuring - Other Operating - Asset Impairment - D&A for periods where missing.
        sga = metrics_data.get("SG&A Expenses", {})
        gp = metrics_data.get("Gross Profit", {})
        oi = metrics_data.get("Operating Income", {})
        rest = metrics_data.get("Restructuring and other charges", {})
        other_op = metrics_data.get("Other Operating Expense", {})
        impair = metrics_data.get("Asset Impairment Charges", {})
        da_by_period = self._get_depreciation_by_period()
        for period_key in all_extracted_periods:
            if period_key in sga and isinstance(sga.get(period_key), (int, float)) and not pd.isna(sga[period_key]):
                continue
            gp_val = gp.get(period_key)
            oi_val = oi.get(period_key)
            if not isinstance(gp_val, (int, float)) or not isinstance(oi_val, (int, float)):
                continue
            if pd.isna(gp_val) or pd.isna(oi_val):
                continue
            rest_val = rest.get(period_key) if isinstance(rest.get(period_key), (int, float)) else 0
            other_val = other_op.get(period_key) if isinstance(other_op.get(period_key), (int, float)) else 0
            impair_val = impair.get(period_key) if isinstance(impair.get(period_key), (int, float)) else 0
            if pd.isna(rest_val):
                rest_val = 0
            if pd.isna(other_val):
                other_val = 0
            if pd.isna(impair_val):
                impair_val = 0
            da_val = da_by_period.get(period_key, 0)
            if da_val is None or pd.isna(da_val):
                da_val = 0
            derived = gp_val - oi_val - rest_val - other_val - impair_val - da_val
            if derived > 0:
                metrics_data.setdefault("SG&A Expenses", {})[period_key] = derived
                reported_metrics.add("SG&A Expenses")
                log.debug(
                    f"Derived SG&A ({derived:,.0f}) from income statement equation for period {period_key}"
                )

        # Step 2: Handle combined fields (only if separate components don't exist)
        # Pass metrics_data so we can check if separate components already exist
        combined_field_data = self._detect_and_handle_combined_fields(metrics_data)
        for std_name, period_data in combined_field_data.items():
            if period_data:  # Only add if we have data
                metrics_data[std_name].update(period_data)
                reported_metrics.add(std_name)

        # Step 2b: For detailed view, extract segment breakdowns
        segment_data: Dict[str, Dict[str, Any]] = {}
        if view == "detailed":
            segment_data = self._extract_segment_breakdowns()
            for label, period_map in segment_data.items():
                if period_map:
                    metrics_data[label] = period_map
                    reported_metrics.add(label)

        # Convert to DataFrame
        if not metrics_data or not reported_metrics:
            return pd.DataFrame()

        # Get all unique periods
        all_periods = set()
        for metric_data in metrics_data.values():
            all_periods.update(metric_data.keys())

        # Sort periods with most recent first (for leftmost column display)
        all_periods = sorted(all_periods, reverse=True)

        # Collapse duplicate metrics (e.g. Research and Development -> R&D, Taxes -> Income Tax Expense)
        for preferred, alternates in self.DUPLICATE_METRIC_GROUPS.items():
            if preferred not in metrics_data and alternates:
                for alt in alternates:
                    if alt in metrics_data:
                        metrics_data[preferred] = metrics_data.pop(alt)
                        reported_metrics.discard(alt)
                        reported_metrics.add(preferred)
                        break
            elif preferred in metrics_data:
                for alt in alternates:
                    if alt not in metrics_data:
                        continue
                    # Merge alternate into preferred for missing periods
                    pref_data = metrics_data[preferred]
                    alt_data = metrics_data[alt]
                    for pk, pv in alt_data.items():
                        if pk not in pref_data or (pv is not None and pd.notna(pv) and (pref_data[pk] is None or pd.isna(pref_data[pk]))):
                            pref_data[pk] = pv
                    del metrics_data[alt]
                    reported_metrics.discard(alt)

        # Prefer SG&A (section total) over G&A component: when SG&A exists, drop G&A
        # (EdgarTools parity - is_total prioritization; avoids multi-filing breakdown pollution)
        if "General and Administrative Expense" in metrics_data and "SG&A Expenses" in metrics_data:
            del metrics_data["General and Administrative Expense"]
            reported_metrics.discard("General and Administrative Expense")
        if "Selling and Marketing Expense" in metrics_data and "SG&A Expenses" in metrics_data:
            del metrics_data["Selling and Marketing Expense"]
            reported_metrics.discard("Selling and Marketing Expense")

        # Suppress Restructuring/Other/Impairment when op ex structure is clean (EdgarTools parity).
        # When (Gross Profit - Operating Income) ≈ (R&D + SG&A), the face statement has no
        # separate Restructuring/Other lines - avoid pollution from disclosure/other-year facts.
        self._suppress_component_metrics_when_clean(metrics_data, all_extracted_periods)

        # Build ordered column list: main metrics with segment rows inline (when view=detailed)
        # view="summary" (EdgarTools SUMMARY): totals only, no segment breakdowns - same as standard
        ordered_columns: List[str] = []
        segment_labels_by_parent: Dict[str, List[str]] = {}
        if view == "detailed" and segment_data:
            for label in segment_data.keys():
                parent = label.split(" — ", 1)[0] if " — " in label else None
                if parent:
                    segment_labels_by_parent.setdefault(parent, []).append(label)

        for std_name in self.METRIC_ORDER:
            if std_name not in reported_metrics:
                continue
            ordered_columns.append(std_name)
            if view == "detailed" and std_name in segment_labels_by_parent:
                for seg_label in sorted(segment_labels_by_parent[std_name]):
                    if seg_label in reported_metrics:
                        ordered_columns.append(seg_label)
        for std_name in reported_metrics:
            if std_name not in ordered_columns:
                ordered_columns.append(std_name)

        # Build DataFrame with ordered columns
        df_data = {}
        for std_name in ordered_columns:
            metric_data = metrics_data.get(std_name, {})
            metric_get = metric_data.get
            df_data[std_name] = [metric_get(period, np.nan) for period in all_periods]

        # show_date_range: when True, use "start to end" for period index (EdgarTools parity)
        # Fact-centric path approximates start = end - 365 days when start unavailable
        index_labels = list(all_periods)
        if show_date_range and all_periods:
            from datetime import datetime as _dt, timedelta
            new_labels = []
            for end_str in all_periods:
                try:
                    end_d = _dt.strptime(end_str, "%Y-%m-%d").date()
                    start_d = end_d - timedelta(days=365)
                    new_labels.append(f"{start_d} to {end_str}")
                except (ValueError, TypeError):
                    new_labels.append(end_str)
            index_labels = new_labels

        df = pd.DataFrame(df_data, index=index_labels)
        df.index.name = "end"

        if label_column is not None:
            df.attrs["label_column"] = label_column

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

        # First, validate Interest Income - it should not match Operating Income or Income Before Taxes values
        if "Interest Income" in df.columns and "Operating Income" in df.columns:
            interest_income_col = df["Interest Income"].copy()
            operating_income_col = df["Operating Income"]
            income_before_taxes_col = df["Income Before Taxes"] if "Income Before Taxes" in df.columns else None

            # Check each period where both values exist
            for idx in df.index:
                interest_val = interest_income_col.loc[idx]
                operating_val = operating_income_col.loc[idx]
                income_before_taxes_val = income_before_taxes_col.loc[idx] if income_before_taxes_col is not None else None

                if (
                    pd.notna(interest_val)
                    and pd.notna(operating_val)
                    and operating_val != 0
                ):
                    # Calculate how similar the values are
                    ratio = (
                        abs(interest_val) / abs(operating_val)
                        if operating_val != 0
                        else float('inf')
                    )
                    diff_ratio = (
                        abs(interest_val - operating_val) / abs(operating_val)
                        if operating_val != 0
                        else 1
                    )
                    abs_diff = abs(interest_val - operating_val)

                    # Enhanced validation: Focus on detecting values suspiciously CLOSE to Operating Income
                    # Values within 1% are almost certainly misclassified Operating Income
                    # Values 1-5% close AND large (> 50% ratio) are also suspicious
                    # We DON'T reject values that are just large but NOT close (legitimate for cash-rich companies)
                    if diff_ratio < 0.01:  # Within 1% - very suspicious, likely misclassified
                        log.warning(
                            f"Interest Income ({interest_val:,.0f}) is suspiciously close to Operating Income "
                            f"({operating_val:,.0f}) for period {idx} (diff: {diff_ratio:.3%}, abs_diff: {abs_diff:,.0f}). "
                            f"Setting Interest Income to NaN as this is likely misclassified Operating Income."
                        )
                        interest_income_col.loc[idx] = np.nan
                    elif 0.01 <= diff_ratio < 0.05 and ratio > 0.50:  # Moderately close (1-5%) AND large (> 50% ratio)
                        log.warning(
                            f"Interest Income ({interest_val:,.0f}) is suspiciously close to Operating Income "
                            f"({operating_val:,.0f}) for period {idx} (diff: {diff_ratio:.2%}, ratio: {ratio:.1%}). "
                            f"Setting Interest Income to NaN as this may be misclassified Operating Income."
                        )
                        interest_income_col.loc[idx] = np.nan
                    
                    # Also check against Income Before Taxes (which might be misclassified as Interest Income)
                    if (
                        income_before_taxes_val is not None
                        and pd.notna(income_before_taxes_val)
                        and pd.notna(interest_income_col.loc[idx])  # Only check if not already rejected
                        and income_before_taxes_val != 0
                    ):
                        ibt_diff_ratio = (
                            abs(interest_val - income_before_taxes_val) / abs(income_before_taxes_val)
                            if income_before_taxes_val != 0
                            else 1
                        )
                        ibt_abs_diff = abs(interest_val - income_before_taxes_val)
                        
                        # If Interest Income is suspiciously close to Income Before Taxes, it's likely misclassified
                        if ibt_diff_ratio < 0.01:  # Within 1% - very suspicious, likely misclassified Income Before Taxes
                            log.warning(
                                f"Interest Income ({interest_val:,.0f}) is suspiciously close to Income Before Taxes "
                                f"({income_before_taxes_val:,.0f}) for period {idx} (diff: {ibt_diff_ratio:.3%}, abs_diff: {ibt_abs_diff:,.0f}). "
                                f"Setting Interest Income to NaN as this is likely misclassified Income Before Taxes."
                            )
                            interest_income_col.loc[idx] = np.nan
                    
                    # Negative Interest Income: allow for debt-heavy companies (e.g. ANF) where
                    # net interest is negative; log at debug only.
                    if interest_val < 0:
                        log.debug(
                            f"Interest Income is negative for period {idx}: {interest_val} "
                            f"(may be net interest for companies with interest expense > interest income)."
                        )
                    
                    # NOTE: We DON'T reject values that are just large (> 10% ratio) but NOT close
                    # Cash-rich companies can legitimately have Interest Income > 10% of Operating Income

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

        # Step 7: EdgarTools-aligned presentation and validation
        try:
            from financial4all.config import get_config
            config = get_config()
            if config.run_datapoint_validation:
                from financial4all.xbrl.datapoint_validation import validate_statement_df
                result = validate_statement_df(df, "IncomeStatement")
                for issue in result.warnings:
                    log.warning("[Datapoint] %s", issue)
        except Exception as e:
            log.debug("Presentation/validation step skipped: %s", e)

        # Cache raw df (before presentation) for standard/summary view
        if _use_cache and view in ("standard", "summary"):
            self._dataframe = df.copy()

        # Apply presentation based on param (EdgarTools parity: param overrides config)
        try:
            from financial4all.config import get_config
            _apply = presentation if presentation is not None else get_config().apply_presentation_signs
        except Exception:
            _apply = presentation if presentation is not None else False
        if _apply:
            from financial4all.xbrl.presentation import apply_presentation
            df = apply_presentation(df, "IncomeStatement")

        if include_standardization:
            df.attrs["standard_concept_map"] = self.get_standard_concept_map(df)
        if label_column is not None:
            df.attrs["label_column"] = label_column
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

        # If we didn't find enough facts, try synonym discovery using reverse_index and SynonymGroups
        # This helps find alternative concepts that might be used
        # BUT: We need to be careful not to match concepts from other metric groups
        if not all_facts_by_concept and xbrl_concepts:
            primary_concept = xbrl_concepts[0]

            # First, try reverse_index for O(1) lookup and context-aware disambiguation
            concept_info = None
            synonym_tags = []
            
            try:
                reverse_index = get_reverse_index()
                # Provide context for better disambiguation (income statement context)
                context = {
                    'statement_type': 'IncomeStatement',
                    'section': 'Other Income/Expense'  # Context helps disambiguate ambiguous tags
                }
                standard_concept = reverse_index.get_standard_concept(primary_concept, context)
                
                if standard_concept:
                    # Get display name and find synonyms from SynonymGroups
                    display_name = reverse_index.get_display_name(primary_concept, context)
                    log.debug(
                        f"Found concept '{primary_concept}' via reverse_index -> '{standard_concept}' (display: '{display_name}')"
                    )
                    
                    # Use SynonymGroups to get all synonyms for this standard concept
                    synonym_groups = get_synonym_groups()
                    # Normalize standard_concept to group name format
                    normalized_concept = standard_concept.lower().replace(' ', '_').replace('-', '_')
                    concept_info = synonym_groups.get_group(normalized_concept)
                    
                    if concept_info:
                        # Use synonyms from the identified group
                        synonym_tags = concept_info.synonyms
                        log.debug(
                            f"Found {len(synonym_tags)} synonyms for '{normalized_concept}' via reverse_index + SynonymGroups"
                        )
            except (ImportError, AttributeError, Exception) as e:
                # Fallback to SynonymGroups if reverse_index fails
                log.debug(f"Reverse index lookup failed for '{primary_concept}': {e}, falling back to SynonymGroups")
            
            # Fallback to SynonymGroups if reverse_index didn't find a match
            if not concept_info:
                synonym_groups = get_synonym_groups()
                concept_info = synonym_groups.identify_concept(primary_concept)
                if concept_info:
                    synonym_tags = concept_info.synonyms
                    log.debug(
                        f"Found concept '{primary_concept}' in SynonymGroups as '{concept_info.name}' with {len(synonym_tags)} synonyms"
                    )

            if concept_info and synonym_tags:

                # Check which synonyms exist in the fact set
                all_concepts_in_factset = {
                    f.concept for f in self._original_fact_set.facts
                }

                # Get concepts from other groups to avoid cross-contamination
                # For Interest Income, we should NOT match Operating Income or Income Before Taxes concepts
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
                    
                    # Get Income Before Taxes concepts to exclude (these are often similar magnitude to Operating Income)
                    income_before_tax_group = synonym_groups.get_group(
                        "income_before_tax"
                    )
                    if income_before_tax_group:
                        for ibt_concept in income_before_tax_group.synonyms:
                            excluded_concepts.add(ibt_concept)
                            excluded_concepts.add(f"us-gaap_{ibt_concept}")
                            excluded_concepts.add(f"us-gaap:{ibt_concept}")
                    
                    log.debug(
                        f"Excluding {len(excluded_concepts)} concepts from Interest Income search "
                        f"(Operating Income + Income Before Taxes) to prevent misclassification"
                    )

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

    def _is_valid_unit_for_metric(self, unit: str, std_name: str) -> bool:
        """
        Check if a unit is valid for a given metric.

        Args:
            unit: Unit string (e.g., "USD", "shares")
            std_name: Standardized metric name

        Returns:
            True if the unit is valid for this metric
        """
        # Outstanding shares metrics use "shares" units, not "USD"
        if std_name in ("Outstanding Shares Basic", "Outstanding Shares Diluted"):
            return unit == "shares" or unit.startswith("shares")
        # All other metrics use USD units
        return unit == "USD" or unit.startswith("USD")

    def _is_disclosure_concept(self, concept: str) -> bool:
        """
        Return True if concept appears to be a disclosure/text-block (non-numeric).

        Excludes concepts that typically contain narrative text rather than
        numeric financial data (e.g., DesignAndDevelopmentCostsDisclosure).

        Args:
            concept: XBRL concept name (may include namespace prefix)

        Returns:
            True if concept should be excluded from numeric metric resolution
        """
        local = concept.split(":")[-1].split("_")[-1] if ":" in concept or "_" in concept else concept
        local_lower = local.lower()
        return (
            "disclosure" in local_lower
            or "textblock" in local_lower
            or local_lower.endswith("policy")
            or         local_lower.endswith("description")
        )

    def _format_dimension_member(self, member: Any) -> str:
        """
        Format dimension member for display (e.g., us-gaap:AmericasMember -> Americas).

        Args:
            member: Dimension member value (QName string or similar)

        Returns:
            Human-readable label
        """
        if member is None:
            return "Unknown"
        s = str(member).strip()
        if ":" in s:
            s = s.split(":")[-1]
        if s.endswith("Member"):
            s = s[:-6]  # Remove "Member" suffix
        return s or "Unknown"

    def _extract_segment_breakdowns(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract segment breakdown facts for SEGMENT_METRICS when view=detailed.

        Returns dimensional facts (Revenue by geography, Cost of Revenue by segment, etc.)
        with breakdown-type dimensions only.

        Returns:
            Dict mapping segment label (e.g., "Revenue — Americas") to period->value map
        """
        result: Dict[str, Dict[str, Any]] = {}

        for std_name in self.SEGMENT_METRICS:
            xbrl_concepts = self.STANDARD_MAPPING.get(std_name, [])
            if not xbrl_concepts:
                continue

            concept_facts_map = self._get_all_facts_for_metric(xbrl_concepts)
            if not concept_facts_map:
                continue

            # Collect dimensional facts with breakdown-type dimensions
            segment_rows: Dict[tuple, Dict[str, Any]] = {}  # (axis, member) -> {period: value}

            for concept, facts in concept_facts_map.items():
                if self._is_disclosure_concept(concept):
                    continue
                for fact in facts:
                    if not fact.dimensions:
                        continue
                    if not isinstance(fact.value, (int, float)):
                        continue
                    if fact.period.period_type != PeriodType.DURATION or not fact.period.is_annual():
                        continue
                    if not self._is_valid_unit_for_metric(fact.unit, std_name):
                        continue

                    dims = fact.dimensions or {}
                    for dim_axis, dim_member in dims.items():
                        if not is_breakdown_dimension(dim_axis):
                            continue
                        member_label = self._format_dimension_member(dim_member)
                        key = (dim_axis, member_label)
                        if key not in segment_rows:
                            segment_rows[key] = {}
                        period_key = str(fact.period.end)
                        segment_rows[key][period_key] = fact.value

            for (axis, member_label), period_map in segment_rows.items():
                label = f"{std_name} — {member_label}"
                result[label] = period_map

        return result

    def _is_total_concept(self, concept: str) -> bool:
        """Check if concept is typically a total (totalLabel) in presentation."""
        local = concept.split("_")[-1] if "_" in concept else concept
        local = local.split(":")[-1] if ":" in local else local
        return local in self.IS_TOTAL_CONCEPTS

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
           - is_total: prefer concepts with totalLabel (EdgarTools parity)
           - Concept priority (earlier in list = higher priority)
           - Form type (10-K preferred over 10-Q)
           - Unit (USD for most metrics, shares for outstanding shares)
           - Filing date (more recent preferred)
        5. Cross-validate with other metrics to prevent misclassification

        Args:
            std_name: Standardized metric name
            xbrl_concepts: List of XBRL concept names in priority order
            other_metrics_data: Dictionary of other already-resolved metrics (for validation)

        Returns:
            Dictionary mapping period_key -> fact.value
        """
        # Prioritize is_total concepts (EdgarTools parity): totals first, then rest
        xbrl_concepts = sorted(
            xbrl_concepts,
            key=lambda c: (0 if self._is_total_concept(c) else 1, xbrl_concepts.index(c)),
        )
        # Get all facts for all concepts
        concept_facts_map = self._get_all_facts_for_metric(xbrl_concepts)

        if not concept_facts_map:
            return {}

        # Exclude disclosure/text-block concepts for numeric metrics
        # Prevents R&D and similar from using narrative disclosures (e.g., DesignAndDevelopmentCostsDisclosure)
        if std_name in self.NUMERIC_METRICS:
            concept_facts_map = {
                c: facts
                for c, facts in concept_facts_map.items()
                if not self._is_disclosure_concept(c)
            }
            if not concept_facts_map:
                return {}

        # Outstanding Shares: exclude balance-sheet instant concepts; prefer duration WeightedAverage*
        # (EdgarTools alignment - SharesAverage vs SharesYearEnd disambiguation)
        if std_name in ("Outstanding Shares Basic", "Outstanding Shares Diluted"):
            excluded_shares = {"CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"}
            concept_facts_map = {
                c: facts
                for c, facts in concept_facts_map.items()
                if not any(
                    ex in c for ex in excluded_shares
                )
            }
            if not concept_facts_map:
                return {}

        # Apply multi-tier filtering with preference for non-dimensional facts
        filtered_facts_by_concept = {}
        for concept, facts in concept_facts_map.items():
            # Separate dimensional and non-dimensional facts
            non_dimensional_facts = [f for f in facts if not f.dimensions]
            dimensional_facts = [f for f in facts if f.dimensions]

            # Tier 1: Strict filter (annual 10-K or 10-K/A, valid unit, no dimensions) - PREFERRED
            # Accept 10-K/A for restatements and improved historical coverage (EdgarTools alignment)
            tier1_facts = [
                f
                for f in non_dimensional_facts
                if f.is_annual_10k_or_amended()
                and self._is_valid_unit_for_metric(f.unit, std_name)
            ]

            # Tier 2: Lenient filter (annual, valid unit, no dimensions, any form)
            if not tier1_facts:
                tier2_facts = [
                    f
                    for f in non_dimensional_facts
                    if f.period.period_type == PeriodType.DURATION
                    and f.period.is_annual()
                    and self._is_valid_unit_for_metric(f.unit, std_name)
                ]
                filtered_facts_by_concept[concept] = tier2_facts
            else:
                filtered_facts_by_concept[concept] = tier1_facts

            # Tier 3: Very lenient (any annual period, valid unit, no dimensions) - fallback
            if not filtered_facts_by_concept[concept]:
                tier3_facts = [
                    f
                    for f in non_dimensional_facts
                    if f.period.period_type == PeriodType.DURATION
                    and f.period.is_annual()
                    and self._is_valid_unit_for_metric(f.unit, std_name)
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
                    and self._is_valid_unit_for_metric(f.unit, std_name)
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

        try:
            from financial4all.config import get_config
            exclude_structural = get_config().exclude_structural_elements
        except Exception:
            exclude_structural = True

        for concept_idx, (concept, facts) in enumerate(
            filtered_facts_by_concept.items()
        ):
            for fact in facts:
                # EdgarTools-aligned: exclude structural elements (Axis, Domain, Member, etc.)
                if exclude_structural and is_xbrl_structural_element(
                    getattr(fact, "concept", "") or concept,
                    getattr(fact, "label", None),
                ):
                    continue
                # For numeric metrics, exclude facts with non-numeric values early
                # (HTML, disclosure text, policy strings)
                if std_name in self.NUMERIC_METRICS and not isinstance(
                    fact.value, (int, float)
                ):
                    continue

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

            # EdgarTools-aligned: sort by fact_resolution priority (non-dimensional > form > concept > unit > recency)
            try:
                from financial4all.config import get_config
                exclude_amended = get_config().exclude_amended_filings
            except Exception:
                exclude_amended = False
            fact_candidates = sort_fact_candidates_by_priority(
                fact_candidates,
                self._is_valid_unit_for_metric,
                std_name,
                exclude_amended=exclude_amended,
            )

            # Select best fact - try candidates in order until we find a valid one
            fact_value = None
            best_fact = None

            for candidate_idx, (concept_idx, fact) in enumerate(fact_candidates):
                candidate_value = fact.value
                is_valid = True

                # Numeric-only filter: skip non-numeric values for numeric metrics
                # Prevents HTML, disclosure text, and policy strings from appearing in cells
                if std_name in self.NUMERIC_METRICS and not isinstance(
                    candidate_value, (int, float)
                ):
                    is_valid = False
                    log.debug(
                        f"Skipping {std_name} fact for period {period_key}: "
                        f"non-numeric value (type={type(candidate_value).__name__})"
                    )

                # Cross-validate with other metrics to prevent misclassification
                # Cost of Revenue: for retailers, Revenue > COGS; reject if candidate COGS > Revenue
                if (
                    std_name == "Cost of Revenue"
                    and other_metrics_data
                    and isinstance(candidate_value, (int, float))
                    and candidate_value > 0
                ):
                    rev_val = other_metrics_data.get("Revenue", {}).get(period_key)
                    if (
                        isinstance(rev_val, (int, float))
                        and rev_val > 0
                        and candidate_value > rev_val
                    ):
                        log.debug(
                            f"Skipping Cost of Revenue candidate for period {period_key}: "
                            f"value {candidate_value:,.0f} > Revenue {rev_val:,.0f} (likely swapped)"
                        )
                        is_valid = False

                # For Interest Income, reject NET concepts (Income - Expense, not income itself)
                # InterestIncomeExpenseNet maps to InterestIncome in gaap but is the NET, not gross income
                if std_name == "Interest Income":
                    fact_concept = (getattr(fact, "concept", "") or "").upper()
                    if "INTERESTINCOMEEXPENSENET" in fact_concept or "INTERESTINCOMEEXPENSENONOPERATINGNET" in fact_concept:
                        log.debug(
                            f"Skipping Interest Income fact for period {period_key}: "
                            f"concept is NET (income-expense), not gross Interest Income"
                        )
                        is_valid = False

                # For Interest Income, check against Operating Income AND Income Before Taxes
                if std_name == "Interest Income" and other_metrics_data and is_valid:
                    operating_income_data = other_metrics_data.get(
                        "Operating Income", {}
                    )
                    operating_val = operating_income_data.get(period_key)
                    
                    income_before_taxes_data = other_metrics_data.get(
                        "Income Before Taxes", {}
                    )
                    income_before_taxes_val = income_before_taxes_data.get(period_key)

                    # Check against Operating Income
                    if (
                        operating_val is not None
                        and isinstance(candidate_value, (int, float))
                        and isinstance(operating_val, (int, float))
                    ):
                        ratio = (
                            abs(candidate_value) / abs(operating_val)
                            if operating_val != 0
                            else float('inf')
                        )
                        diff_ratio = (
                            abs(candidate_value - operating_val) / abs(operating_val)
                            if operating_val != 0
                            else 1
                        )
                        abs_diff = abs(candidate_value - operating_val)

                        # Enhanced validation: Focus on detecting values suspiciously CLOSE to Operating Income
                        # Values within 1% are almost certainly misclassified Operating Income
                        # Values 1-5% close AND large (> 50% ratio) are also suspicious
                        # We DON'T reject values that are just large but NOT close (legitimate for cash-rich companies)
                        if diff_ratio < 0.01:  # Within 1% - very suspicious, likely misclassified
                            log.warning(
                                f"Skipping Interest Income fact candidate {candidate_idx} for period {period_key}: "
                                f"value {candidate_value} is suspiciously close to Operating Income {operating_val} "
                                f"(diff: {diff_ratio:.3%}, abs_diff: {abs_diff:,.0f}). This is likely misclassified Operating Income."
                            )
                            is_valid = False
                        elif 0.01 <= diff_ratio < 0.05 and ratio > 0.50:  # Moderately close (1-5%) AND large (> 50% ratio)
                            log.warning(
                                f"Skipping Interest Income fact candidate {candidate_idx} for period {period_key}: "
                                f"value {candidate_value} is suspiciously close to Operating Income {operating_val} "
                                f"(diff: {diff_ratio:.2%}, ratio: {ratio:.1%}). This may be misclassified Operating Income."
                            )
                            is_valid = False
                    
                    # Also check against Income Before Taxes (which might be misclassified as Interest Income)
                    if (
                        income_before_taxes_val is not None
                        and isinstance(candidate_value, (int, float))
                        and isinstance(income_before_taxes_val, (int, float))
                        and is_valid  # Only check if not already rejected
                    ):
                        ibt_diff_ratio = (
                            abs(candidate_value - income_before_taxes_val) / abs(income_before_taxes_val)
                            if income_before_taxes_val != 0
                            else 1
                        )
                        ibt_abs_diff = abs(candidate_value - income_before_taxes_val)
                        
                        # If Interest Income is suspiciously close to Income Before Taxes, it's likely misclassified
                        if ibt_diff_ratio < 0.01:  # Within 1% - very suspicious, likely misclassified Income Before Taxes
                            log.warning(
                                f"Skipping Interest Income fact candidate {candidate_idx} for period {period_key}: "
                                f"value {candidate_value} is suspiciously close to Income Before Taxes {income_before_taxes_val} "
                                f"(diff: {ibt_diff_ratio:.3%}, abs_diff: {ibt_abs_diff:,.0f}). This is likely misclassified Income Before Taxes."
                            )
                            is_valid = False
                    
                    # Negative Interest Income: some companies (e.g. ANF with net interest expense) report
                    # negative values when interest expense exceeds interest income. Allow but log at debug.
                    if isinstance(candidate_value, (int, float)) and candidate_value < 0:
                        log.debug(
                            f"Interest Income fact candidate {candidate_idx} for period {period_key}: "
                            f"value {candidate_value} is negative (may be net interest for debt-heavy companies)."
                        )
                    elif not isinstance(candidate_value, (int, float)):
                        # Non-numeric (e.g. string) cannot be used for Interest Income
                        is_valid = False
                    
                    # NOTE: We DON'T reject values that are just large (> 10% ratio) but NOT close
                    # Cash-rich companies can legitimately have Interest Income > 10% of Operating Income

                # For Interest Expense, handle NET concepts when we have separate Interest Income
                # NET concepts (InterestIncomeExpenseNet, InterestIncomeExpenseNonoperatingNet) represent
                # (Income - Expense), not the expense itself. When we have Interest Income, derive the
                # actual expense: Expense = Interest Income - NET (e.g. ANF 2025: 39,934 - 27,857 = 12,077).
                if std_name == "Interest Expense" and other_metrics_data:
                    interest_income_data = other_metrics_data.get("Interest Income", {})
                    interest_income_val = interest_income_data.get(period_key)
                    fact_concept = (getattr(fact, "concept", "") or "").upper()
                    is_net_concept = (
                        "INTERESTINCOMEEXPENSENET" in fact_concept
                        or "INTERESTINCOMEEXPENSENONOPERATINGNET" in fact_concept
                    )
                    if (
                        is_net_concept
                        and interest_income_val is not None
                        and isinstance(candidate_value, (int, float))
                        and isinstance(interest_income_val, (int, float))
                    ):
                        # Derive actual expense from NET: Expense = Interest Income - NET
                        net_value = candidate_value
                        derived_expense = interest_income_val - net_value
                        candidate_value = derived_expense
                        log.debug(
                            f"Interest Expense for period {period_key}: derived {derived_expense:,.0f} "
                            f"from Interest Income ({interest_income_val:,.0f}) - NET ({net_value:,.0f})"
                        )

                if is_valid:
                    fact_value = candidate_value
                    best_fact = fact
                    break

            # If no valid fact found after validation, skip this period
            if fact_value is None or best_fact is None:
                log.debug(
                    f"No valid fact found for {std_name} period {period_key} after cross-validation "
                    f"(checked {len(fact_candidates)} candidates)"
                )
                continue

            # Log which candidate was selected (for debugging)
            if std_name == "Interest Income":
                selected_idx = next(
                    (i for i, (ci, f) in enumerate(fact_candidates) if f == best_fact),
                    -1
                )
                log.debug(
                    f"Selected Interest Income fact for period {period_key}: value {fact_value:,.0f} "
                    f"(from candidate {selected_idx} of {len(fact_candidates)}, concept: {best_fact.concept})"
                )

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
                        # BUT: For Interest Income, validate alternatives against Operating Income
                        alternative_facts = []
                        for f in fact_candidates[1:]:
                            if not isinstance(f[1].value, (int, float)):
                                continue
                            if f[1].dimensions:  # Skip dimensional facts
                                continue
                            if abs(f[1].value) <= abs(fact_value) * 5:  # Must be at least 5x larger
                                continue
                            
                            # For Interest Income, validate alternative against Operating Income
                            if std_name == "Interest Income" and other_metrics_data:
                                operating_income_data = other_metrics_data.get("Operating Income", {})
                                operating_val = operating_income_data.get(period_key)
                                if operating_val is not None and isinstance(operating_val, (int, float)):
                                    alt_ratio = abs(f[1].value) / abs(operating_val) if operating_val != 0 else float('inf')
                                    alt_diff_ratio = abs(f[1].value - operating_val) / abs(operating_val) if operating_val != 0 else 1
                                    alt_abs_diff = abs(f[1].value - operating_val)
                                    
                                    # Skip if alternative is also suspiciously close to Operating Income
                                    if alt_diff_ratio < 0.05 or alt_abs_diff < abs(operating_val) * 0.05:
                                        log.debug(
                                            f"Skipping alternative Interest Income fact for period {period_key}: "
                                            f"value {f[1].value} is also suspiciously close to Operating Income"
                                        )
                                        continue
                                    if alt_ratio > 0.10:
                                        log.debug(
                                            f"Skipping alternative Interest Income fact for period {period_key}: "
                                            f"value {f[1].value} is also suspiciously large relative to Operating Income"
                                        )
                                        continue
                            
                            alternative_facts.append(f)

                        # If no non-dimensional alternatives, try any larger value (but still validate for Interest Income)
                        if not alternative_facts:
                            for f in fact_candidates[1:]:
                                if not isinstance(f[1].value, (int, float)):
                                    continue
                                if abs(f[1].value) <= abs(fact_value) * 10:  # Must be at least 10x larger
                                    continue
                                
                                # For Interest Income, validate alternative against Operating Income
                                if std_name == "Interest Income" and other_metrics_data:
                                    operating_income_data = other_metrics_data.get("Operating Income", {})
                                    operating_val = operating_income_data.get(period_key)
                                    if operating_val is not None and isinstance(operating_val, (int, float)):
                                        alt_ratio = abs(f[1].value) / abs(operating_val) if operating_val != 0 else float('inf')
                                        alt_diff_ratio = abs(f[1].value - operating_val) / abs(operating_val) if operating_val != 0 else 1
                                        alt_abs_diff = abs(f[1].value - operating_val)
                                        
                                        if alt_diff_ratio < 0.05 or alt_abs_diff < abs(operating_val) * 0.05:
                                            continue
                                        if alt_ratio > 0.10:
                                            continue
                                
                                alternative_facts.append(f)

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

    @staticmethod
    def _suppress_component_metrics_when_clean(
        metrics_data: Dict[str, Dict[str, Any]],
        periods: Set[str],
    ) -> None:
        """
        For periods where (Gross Profit - Operating Income) ≈ (R&D + SG&A), suppress
        Restructuring, Other Operating Expense, Asset Impairment - they likely come from
        disclosures/other years, not the face statement (EdgarTools parity).
        """
        gp = metrics_data.get("Gross Profit", {})
        oi = metrics_data.get("Operating Income", {})
        rd = metrics_data.get("R&D Expenses", {})
        sga = metrics_data.get("SG&A Expenses", {})
        if not gp or not oi or (not rd and not sga):
            return
        components = [
            "Restructuring and other charges",
            "Other Operating Expense",
            "Asset Impairment Charges",
        ]
        for period in periods:
            gpv = gp.get(period)
            oiv = oi.get(period)
            rdv = rd.get(period) or 0
            sgav = sga.get(period) or 0
            if not isinstance(gpv, (int, float)) or not isinstance(oiv, (int, float)):
                continue
            if pd.isna(gpv) or pd.isna(oiv):
                continue
            implied_opex = float(gpv) - float(oiv)
            core_sum = (float(rdv) if isinstance(rdv, (int, float)) and not pd.isna(rdv) else 0) + (
                float(sgav) if isinstance(sgav, (int, float)) and not pd.isna(sgav) else 0
            )
            if implied_opex <= 0:
                continue
            tol = max(abs(implied_opex) * 0.005, 1.0)  # 0.5% or $1
            if abs(implied_opex - core_sum) <= tol:
                for comp in components:
                    if comp in metrics_data and period in metrics_data[comp]:
                        metrics_data[comp][period] = np.nan

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

        # Remove "Operating Expenses" when we have component breakdown (SG&A or G&A+Selling)
        if "Operating Expenses" in df_cleaned.columns:
            has_sga = "SG&A Expenses" in df_cleaned.columns
            has_components = ("General and Administrative Expense" in df_cleaned.columns
                             and "Selling and Marketing Expense" in df_cleaned.columns)
            if "R&D Expenses" in df_cleaned.columns and (has_sga or has_components):
                df_cleaned = df_cleaned.drop(columns=["Operating Expenses"])
                log.debug(
                    "Removed 'Operating Expenses' - redundant with R&D Expenses + SG&A Expenses"
                )

        return df_cleaned

    def _reorder_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder DataFrame columns according to standard income statement order.

        Segment columns (e.g., "Revenue — Americas") are placed inline after
        their parent metric.

        Args:
            df: DataFrame with income statement data

        Returns:
            DataFrame with reordered columns
        """
        existing_cols = list(df.columns)
        segment_cols_by_parent: Dict[str, List[str]] = {}
        remaining = []
        for col in existing_cols:
            if " — " in col:
                parent = col.split(" — ", 1)[0]
                segment_cols_by_parent.setdefault(parent, []).append(col)
            else:
                remaining.append(col)

        ordered_cols = []
        for metric in self.METRIC_ORDER:
            if metric not in remaining:
                continue
            ordered_cols.append(metric)
            remaining.remove(metric)
            if metric in segment_cols_by_parent:
                for seg in sorted(segment_cols_by_parent[metric]):
                    ordered_cols.append(seg)

        for col in remaining:
            ordered_cols.append(col)
        for parent, segs in segment_cols_by_parent.items():
            for seg in segs:
                if seg not in ordered_cols:
                    ordered_cols.append(seg)

        return df[[c for c in ordered_cols if c in df.columns]]

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

        # Operating Income = Gross Profit - sum(all operating expenses)
        # Use Operating Expenses (total) if only that; else sum component columns
        if "Operating Income" in df_calc.columns and "Gross Profit" in df_calc.columns:
            mask = df_calc["Operating Income"].isna()
            if mask.any():
                component_cols = [
                    "R&D Expenses",
                    "SG&A Expenses",
                    "Restructuring and other charges",
                    "Other Operating Expense",
                    "Asset Impairment Charges",
                ]
                present_components = [c for c in component_cols if c in df_calc.columns]
                if "Operating Expenses" in df_calc.columns and not present_components:
                    df_calc.loc[mask, "Operating Income"] = (
                        df_calc.loc[mask, "Gross Profit"]
                        - df_calc.loc[mask, "Operating Expenses"]
                    )
                elif present_components:
                    total_opex = df_calc[present_components[0]].fillna(0)
                    for c in present_components[1:]:
                        total_opex = total_opex + df_calc[c].fillna(0)
                    df_calc.loc[mask, "Operating Income"] = (
                        df_calc.loc[mask, "Gross Profit"] - total_opex
                    )

        # Detect when "Other, net" is actually the total "Other income (expense), net"
        # NonoperatingIncomeExpense can tag the total line (e.g. NVDA). Two signals:
        # 1) Other, net ≈ IBT - OI (or NI+Taxes - OI) => it's the total
        # 2) Interest Income + Interest Expense + Other_net >> Other_net => double-count, Other_net is the total
        if "Other, net" in df_calc.columns and "Operating Income" in df_calc.columns:
            # Get expected total (IBT - OI) from IBT or Net Income + Taxes
            has_ibt = "Income Before Taxes" in df_calc.columns
            has_ni_tax = "Net Income" in df_calc.columns and "Taxes" in df_calc.columns
            if has_ibt or has_ni_tax:
                if "Other income (expense), net" not in df_calc.columns:
                    df_calc["Other income (expense), net"] = np.nan
                for idx in df_calc.index:
                    other_net_val = df_calc.loc[idx, "Other, net"]
                    oi_val = df_calc.loc[idx, "Operating Income"]
                    if pd.isna(other_net_val) or pd.isna(oi_val):
                        continue
                    # expected_total = IBT - OI
                    if has_ibt:
                        ibt_val = df_calc.loc[idx, "Income Before Taxes"]
                    else:
                        ibt_val = None
                    if pd.notna(ibt_val):
                        expected_total = ibt_val - oi_val
                    elif has_ni_tax:
                        ni = df_calc.loc[idx, "Net Income"]
                        tax = df_calc.loc[idx, "Taxes"]
                        if pd.notna(ni) and pd.notna(tax):
                            expected_total = (ni + tax) - oi_val
                        else:
                            continue
                    else:
                        continue
                    if abs(expected_total) < 1e-6:
                        continue
                    # Use 5% tolerance - companies may report IBT from different concepts (e.g. continuing ops)
                    if abs(other_net_val - expected_total) < max(1, abs(expected_total) * 0.05):
                        # Other, net is the total; use for Other income (expense), net
                        df_calc.loc[idx, "Other income (expense), net"] = other_net_val
                        # Derive component: Other, net = Total - Interest Income - Interest Expense
                        int_inc = df_calc.loc[idx, "Interest Income"] if "Interest Income" in df_calc.columns else 0
                        int_exp = df_calc.loc[idx, "Interest Expense"] if "Interest Expense" in df_calc.columns else 0
                        int_inc = int_inc if pd.notna(int_inc) else 0
                        int_exp = int_exp if pd.notna(int_exp) else 0
                        derived_other_net = other_net_val - int_inc - int_exp
                        df_calc.loc[idx, "Other, net"] = derived_other_net
                        log.debug(
                            f"Other, net was total for {idx}: set Other income (expense), net={other_net_val:,.0f}, "
                            f"derived Other, net={derived_other_net:,.0f}"
                        )

        # Fallback: when Other_net is the total (not component), calc_total = Int_inc + Int_exp + Other_net
        # would double-count. Signal: Other_net >= Interest_inc (total is large) AND calc_total >> Other_net.
        # Don't trigger when Other_net < Interest_inc (Other_net is the small component, e.g. NVDA 2024).
        if (
            "Other, net" in df_calc.columns
            and "Interest Income" in df_calc.columns
            and ("Other income (expense), net" not in df_calc.columns or df_calc["Other income (expense), net"].isna().any())
        ):
            if "Other income (expense), net" not in df_calc.columns:
                df_calc["Other income (expense), net"] = np.nan
            for idx in df_calc.index:
                if pd.notna(df_calc.loc[idx, "Other income (expense), net"]):
                    continue
                other_net_val = df_calc.loc[idx, "Other, net"]
                int_inc = df_calc.loc[idx, "Interest Income"] if pd.notna(df_calc.loc[idx, "Interest Income"]) else 0
                int_exp = df_calc.loc[idx, "Interest Expense"] if "Interest Expense" in df_calc.columns and pd.notna(df_calc.loc[idx, "Interest Expense"]) else 0
                if pd.isna(other_net_val) or other_net_val == 0:
                    continue
                calc_total = int_inc + int_exp + other_net_val
                # Only when Other_net looks like the total (>= 50% of Interest_inc) and we'd double-count
                other_is_large = abs(other_net_val) >= abs(int_inc) * 0.5
                if other_is_large and calc_total > abs(other_net_val) * 1.15:
                    df_calc.loc[idx, "Other income (expense), net"] = other_net_val
                    derived_other_net = other_net_val - int_inc - int_exp
                    df_calc.loc[idx, "Other, net"] = derived_other_net
                    log.debug(
                        f"Other, net double-count fix for {idx}: calc_total {calc_total:,.0f} >> other_net {other_net_val:,.0f}; "
                        f"set total, derived Other, net={derived_other_net:,.0f}"
                    )

        # PRIORITY 3 (run first): Calculate Other income (expense), net from components if not already calculated
        # Other income (expense), net = Interest Income + Interest Expense + Other, net
        # Must run before PRIORITY 1/2 so that IBT = OI + Other income can use the correct Other income
        # when companies (e.g. ANF) report Interest Income/Expense separately but IBT is mis-extracted as OI
        if (
            "Other income (expense), net" not in df_calc.columns
            or df_calc["Other income (expense), net"].isna().any()
        ):
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
                if "Other income (expense), net" not in df_calc.columns:
                    df_calc["Other income (expense), net"] = np.nan

                mask = df_calc["Other income (expense), net"].isna()
                if mask.any():
                    other_income_expense = pd.Series(0.0, index=df_calc.index[mask])
                    if "Interest Income" in df_calc.columns:
                        other_income_expense = other_income_expense + df_calc.loc[mask, "Interest Income"].fillna(0)
                    if "Interest Expense" in df_calc.columns:
                        other_income_expense = other_income_expense + df_calc.loc[mask, "Interest Expense"].fillna(0)
                    if "Other, net" in df_calc.columns:
                        other_income_expense = other_income_expense + df_calc.loc[mask, "Other, net"].fillna(0)
                    df_calc.loc[mask, "Other income (expense), net"] = other_income_expense

        # PRIORITY 1: If Income Before Taxes was extracted directly, use it and calculate backwards
        # Income Before Taxes (extracted) -> Other income (expense), net = Income Before Taxes - Operating Income
        # Only fills Other income where still missing (PRIORITY 3 may have already populated from components)
        income_before_taxes_extracted = False
        if "Income Before Taxes" in df_calc.columns:
            # Check if Income Before Taxes has any extracted (non-null) values
            if df_calc["Income Before Taxes"].notna().any():
                income_before_taxes_extracted = True
                
                # Cross-validate Income Before Taxes = Net Income + Taxes
                # Note: We may be using "Net Income Attributable to Common Stockholders" which excludes
                # noncontrolling interests, so the validation needs to account for this
                if "Net Income" in df_calc.columns and "Taxes" in df_calc.columns:
                    for idx in df_calc.index:
                        ibt_val = df_calc.loc[idx, "Income Before Taxes"]
                        net_income_val = df_calc.loc[idx, "Net Income"]
                        taxes_val = df_calc.loc[idx, "Taxes"]
                        
                        if pd.notna(ibt_val) and pd.notna(net_income_val) and pd.notna(taxes_val):
                            calculated_ibt = net_income_val + taxes_val
                            diff = abs(ibt_val - calculated_ibt)
                            
                            # Check if difference could be explained by noncontrolling interests
                            # If Income Before Taxes - Taxes > Net Income, the difference is likely noncontrolling interests
                            noncontrolling_interest = ibt_val - taxes_val - net_income_val
                            
                            # Calculate relative difference
                            if calculated_ibt != 0:
                                diff_ratio = diff / abs(calculated_ibt)
                            else:
                                diff_ratio = 1 if diff > 0 else 0
                            
                            # Only warn if difference is significant AND doesn't look like noncontrolling interests
                            # Noncontrolling interests are typically small relative to Net Income (< 5% for most companies)
                            # If the difference is large relative to Net Income, it's likely a real issue
                            if diff_ratio > 0.01:  # More than 1% difference
                                # Check if difference could be noncontrolling interests
                                if abs(noncontrolling_interest) > 0 and abs(noncontrolling_interest) < abs(net_income_val) * 0.10:
                                    # Difference is small relative to Net Income (< 10%), likely noncontrolling interests
                                    log.debug(
                                        f"Income Before Taxes ({ibt_val:,.0f}) - Taxes ({taxes_val:,.0f}) = "
                                        f"{ibt_val - taxes_val:,.0f}, but Net Income ({net_income_val:,.0f}) is "
                                        f"{noncontrolling_interest:,.0f} less. This is likely due to noncontrolling interests. "
                                        f"Using extracted Income Before Taxes value."
                                    )
                                else:
                                    # Difference is large, likely a real issue
                                    log.warning(
                                        f"Income Before Taxes ({ibt_val:,.0f}) doesn't match Net Income + Taxes "
                                        f"({calculated_ibt:,.0f}) for period {idx} (diff: {diff:,.0f}, {diff_ratio:.2%}). "
                                        f"Using extracted Income Before Taxes value."
                                    )
                
                # Calculate Other income (expense), net backwards: Other income = Income Before Taxes - Operating Income
                if "Operating Income" in df_calc.columns:
                    if "Other income (expense), net" not in df_calc.columns:
                        df_calc["Other income (expense), net"] = np.nan
                    
                    # Calculate: Other income (expense), net = Income Before Taxes - Operating Income
                    # Only calculate where Income Before Taxes exists and Other income (expense), net is missing
                    mask = df_calc["Income Before Taxes"].notna() & df_calc["Other income (expense), net"].isna()
                    if mask.any() and "Operating Income" in df_calc.columns:
                        df_calc.loc[mask, "Other income (expense), net"] = (
                            df_calc.loc[mask, "Income Before Taxes"]
                            - df_calc.loc[mask, "Operating Income"]
                        )
                        log.debug(
                            f"Calculated Other income (expense), net from extracted Income Before Taxes "
                            f"for {mask.sum()} periods"
                        )

        # PRIORITY 2: Calculate Income Before Taxes = Operating Income + Other income (expense), net
        # When we have both OI and Other income, use the calculated value—it is more reliable than
        # extracted IBT, which may equal Operating Income (e.g. ANF uses same/similar concept for both).
        if "Operating Income" in df_calc.columns:
            # Create or update the column if it doesn't exist
            if "Income Before Taxes" not in df_calc.columns:
                df_calc["Income Before Taxes"] = np.nan

            other_income_component = None
            if "Other income (expense), net" in df_calc.columns:
                other_income_component = df_calc["Other income (expense), net"]
            elif "Interest Income (Net)" in df_calc.columns:
                other_income_component = df_calc["Interest Income (Net)"]

            if other_income_component is not None:
                # Calculate IBT = OI + Other for all periods where both are available
                calc_ibt = df_calc["Operating Income"].fillna(0) + other_income_component.fillna(0)
                mask = df_calc["Operating Income"].notna() & other_income_component.notna()
                if mask.any():
                    # Use calculated IBT when it differs from extracted (fixes ANF-type misclassification)
                    # or when IBT was missing
                    use_calc = df_calc["Income Before Taxes"].isna()
                    diff_significant = (
                        (df_calc["Income Before Taxes"] - calc_ibt).abs()
                        > (calc_ibt.abs() * 0.001 + 1)
                    )
                    use_calc = use_calc | (mask & diff_significant)
                    if use_calc.any():
                        df_calc.loc[use_calc, "Income Before Taxes"] = calc_ibt.loc[use_calc]
                        if diff_significant.any():
                            log.debug(
                                "Income Before Taxes: using OI + Other income (extracted IBT differed)"
                            )

        # Net Income = Income Before Taxes - Taxes
        if "Net Income" in df_calc.columns:
            mask = df_calc["Net Income"].isna()
            if "Income Before Taxes" in df_calc.columns and "Taxes" in df_calc.columns:
                df_calc.loc[mask, "Net Income"] = (
                    df_calc.loc[mask, "Income Before Taxes"]
                    - df_calc.loc[mask, "Taxes"]
                )

        return df_calc

    def get_standard_concept_map(self, df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        Get mapping of display metrics to standard concept identifiers (EdgarTools parity).

        Useful for cross-company analysis and filtering. Segment columns (e.g. "Revenue — Americas")
        map to their parent standard concept ("Revenue").

        Args:
            df: DataFrame with metric columns. If None, uses to_dataframe().

        Returns:
            Dict mapping each column name to its standard concept (base metric name)
        """
        if df is None:
            df = self.to_dataframe()
        if df is None or df.empty:
            return {}
        result = {}
        for col in df.columns:
            # Segment columns "Revenue — Americas" -> "Revenue"
            base = col.split(" — ", 1)[0].strip() if " — " in col else col
            result[col] = base
        return result

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
