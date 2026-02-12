# financial4all/financials/balance_sheet.py
"""
Balance sheet extraction and standardization from XBRL.

BalanceSheet is built from a FactSet (e.g. from SEC company facts). It maps
standardized display names (Total Assets, Current Assets, Stockholders Equity,
etc.) to XBRL concepts via SynonymGroups, resolves one value per period
(instant balance sheet dates), and returns a period-indexed DataFrame.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

from financial4all.xbrl.facts import FactSet, Fact
from financial4all.xbrl.structural_filter import is_xbrl_structural_element
from financial4all.xbrl.standardization import (
    get_synonym_groups,
    get_default_store,
    _load_company_tags_by_display,
)
from financial4all.core import log


class BalanceSheet:
    """
    Balance sheet built from XBRL facts with standardized metric names.

    Uses DISPLAY_NAME_TO_CONCEPT and SynonymGroups to resolve concepts;
    from_company_facts() builds from the SEC company facts API response.
    to_dataframe() returns a period-indexed DataFrame (instant dates).
    """

    # Mapping from display names to concept names in SynonymGroups.
    # Aligned with EdgarTools concept_mappings.json.
    DISPLAY_NAME_TO_CONCEPT = {
        "Total Assets": "total_assets",
        "Current Assets": "total_current_assets",
        "Total Liabilities": "total_liabilities",
        "Current Liabilities": "total_current_liabilities",
        "Stockholders Equity": "total_stockholders_equity",
        "Receivables": "accounts_receivable",
        "Inventory": "inventory",
        "Payables": "accounts_payable",
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

        # Merge company-specific tags for balance sheet concepts
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

    # Concepts that are typically totals (EdgarTools parity - prefer over components)
    IS_TOTAL_CONCEPTS: Set[str] = frozenset({
        "AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "AccountsReceivableNet",
        "InventoryNet", "InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings",
        "AccountsPayableCurrent", "AccountsPayableTradeCurrent", "TradePayables",
        "Assets", "Liabilities", "StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    })

    def _is_total_concept(self, concept: str) -> bool:
        """Check if concept is typically a total (prefer over components)."""
        local = concept.split("_")[-1] if "_" in concept else concept
        local = local.split(":")[-1] if ":" in local else local
        return local in self.IS_TOTAL_CONCEPTS

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

    @classmethod
    def from_filing(
        cls, filing: Any, client: Optional[Any] = None
    ) -> "BalanceSheet":
        """
        Create balance sheet from a SEC filing (statement-centric extraction).

        Parses XBRL from the filing and builds the statement for consistent
        per-filing structure.

        Args:
            filing: Filing object with get_xbrl_content()
            client: Optional SECClient for fetching XBRL content

        Returns:
            BalanceSheet object
        """
        from financial4all.xbrl.xbrl import XBRL

        xbrl = XBRL.from_filing(filing, client=client)
        fact_set = FactSet.from_xbrl_instance(xbrl, cik=getattr(filing, "cik", None))
        return cls(fact_set)

    @classmethod
    def from_filings(
        cls,
        filings: List[Any],
        client: Optional[Any] = None,
    ) -> "BalanceSheet":
        """
        Create balance sheet from multiple SEC filings (multi-year extraction).

        Parses XBRL from each filing, merges FactSets (preferring most recent
        filing for overlapping periods), and builds a unified statement.

        Args:
            filings: List of Filing objects (most recent first)
            client: Optional SECClient for fetching XBRL content

        Returns:
            BalanceSheet object with merged multi-year data
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
        Fill gaps by merging facts from SEC company facts API (instant facts only).

        Balance sheet uses instant periods; only supplements with instant facts
        to avoid polluting with duration/quarterly data.

        Args:
            company_facts: Dictionary from SEC company facts API
            cik: Optional CIK for entity info extraction
        """
        from financial4all.xbrl.periods import PeriodType

        cf_fs = FactSet.from_company_facts(company_facts, cik=cik)
        instant_only = lambda f: f.period.period_type == PeriodType.INSTANT
        self.fact_set.supplement_from(cf_fs, filter_fn=instant_only)
        self._dataframe = None  # Invalidate cache so to_dataframe() reflects new facts

    def to_dataframe(
        self,
        presentation: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Convert balance sheet to pandas DataFrame.

        Only includes metrics that have at least one reported value.
        Filters out completely empty columns.

        Args:
            presentation: For API parity with EdgarTools. Balance sheet has no
                         sign transformation; this param is accepted but ignored.

        Returns:
            DataFrame with standardized balance sheet metrics
        """
        _use_cache = True
        try:
            from financial4all.config import get_config
            _use_cache = not get_config().disable_statement_cache
        except Exception:
            pass
        if _use_cache and self._dataframe is not None:
            return self._dataframe.copy()

        # Extract metrics by standard name
        metrics_data = defaultdict(dict)
        reported_metrics = set()  # Track which metrics have at least one value

        try:
            from financial4all.config import get_config
            exclude_structural = get_config().exclude_structural_elements
        except Exception:
            exclude_structural = True

        for std_name, xbrl_concepts in self.STANDARD_MAPPING.items():
            # Prefer is_total concepts (EdgarTools parity)
            xbrl_concepts = sorted(
                xbrl_concepts,
                key=lambda c: (0 if self._is_total_concept(c) else 1, xbrl_concepts.index(c)),
            )
            # Merge across ALL concepts: first concept with data for a period wins.
            for concept in xbrl_concepts:
                facts = self.fact_set.get_by_concept(concept)
                if not facts:
                    continue
                for fact in facts:
                    if exclude_structural and is_xbrl_structural_element(
                        getattr(fact, "concept", "") or concept,
                        getattr(fact, "label", None),
                    ):
                        continue
                    period_key = str(fact.period.end)
                    if fact.unit == "USD" or fact.unit.startswith("USD"):
                        if period_key not in metrics_data[std_name]:
                            metrics_data[std_name][period_key] = fact.value
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

        if _use_cache:
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
