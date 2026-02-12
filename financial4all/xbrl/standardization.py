# financial4all/xbrl/standardization.py
"""
Unified Standardization Infrastructure for Financial4All.

This module provides centralized standardization components for XBRL concepts,
enabling consistent cross-company financial analysis. Inspired by EdgarTools'
standardization approach.

Components:
 - SynonymGroups: Unified synonym management for XBRL tags
 - ConceptInfo: Rich metadata about identified concepts
 - StandardizationStore: Legacy compatibility layer

Example:
 >>> from financial4all.xbrl.standardization import get_synonym_groups
 >>>
 >>> # Get default singleton instance
 >>> synonyms = get_synonym_groups()
 >>>
 >>> # Look up synonyms for a concept
 >>> tags = synonyms.get_synonyms('revenue')
 >>> print(tags[:2])
 ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues']
 >>>
 >>> # Identify what concept a tag represents
 >>> info = synonyms.identify_concept('us-gaap:Revenues')
 >>> print(info.name)
 'revenue'
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from financial4all.core import log

# Module-level caches
_default_instance: Optional["SynonymGroups"] = None
_builtin_groups_cache: Optional[List["SynonymGroup"]] = None

# Map company mapping concept keys (from company_mappings JSON) to our display names.
# Used when merging company-specific tags into STANDARD_MAPPING for extraction.
# Keys must match DISPLAY_NAME_TO_CONCEPT keys in IncomeStatement/BalanceSheet/CashFlow.
COMPANY_CONCEPT_TO_DISPLAY: Dict[str, str] = {
    "Capital Expenditures": "CapEx",
    "Depreciation and Amortization": "Depreciation & Amortization",
    "Revenue": "Revenue",
    "Cost of Revenue": "Cost of Revenue",
    "Accounts Receivable": "Receivables",
    "Receivables": "Receivables",
    "Inventory": "Inventory",
    "Accounts Payable": "Payables",
    "Payables": "Payables",
}


def _load_company_tags_by_display() -> Dict[str, List[str]]:
    """
    Load all company concept_mappings and merge into display_name -> [tags].

    Company mappings use keys like "Capital Expenditures", "Revenue"; we map
    these to our display names (CapEx, Revenue) so statement classes can merge
    company-specific tags (e.g. NVDA PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets)
    into STANDARD_MAPPING for comprehensive extraction.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    company_dir = os.path.join(module_dir, "standardization", "company_mappings")
    if not os.path.exists(company_dir):
        company_dir = os.path.join(module_dir, "company_mappings")
    if not os.path.exists(company_dir):
        return {}

    out: Dict[str, List[str]] = {}
    for file in os.listdir(company_dir):
        if not file.endswith("_mappings.json"):
            continue
        try:
            path = os.path.join(company_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cm = data.get("concept_mappings", {})
            for company_key, tags in cm.items():
                if company_key.startswith("_"):
                    continue
                if not isinstance(tags, list):
                    continue
                our_display = COMPANY_CONCEPT_TO_DISPLAY.get(company_key)
                if our_display:
                    out.setdefault(our_display, []).extend(tags)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.debug("Could not load company mapping %s: %s", file, e)
    for k in out:
        out[k] = list(dict.fromkeys(out[k]))
    return out


def _normalize_name(name: str) -> str:
    """Normalize a concept name to lowercase with underscores (EdgarTools convention)."""
    s = name.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    s = s.replace(",", "").replace("(", "").replace(")", "").replace("'", "").replace("__", "_").strip("_")
    return s


@dataclass
class SynonymGroup:
    """
    A group of XBRL tags that represent the same financial concept.

    Attributes:
        name: Canonical name for the concept (e.g., 'revenue', 'net_income')
        synonyms: List of XBRL tag names that represent this concept
        description: Human-readable description of the concept
        namespace: Default namespace for tags (default: 'us-gaap')
        priority_order: How to order synonyms when resolving
            - 'listed': Use order as specified in synonyms list
            - 'frequency': Order by usage frequency (most common first)
            - 'specificity': Order by tag specificity (most specific first)
        category: Financial statement category (e.g., 'income_statement', 'balance_sheet')
    """

    name: str
    synonyms: List[str]
    description: str = ""
    namespace: str = "us-gaap"
    priority_order: str = "listed"
    category: str = ""
    # Internal set for O(1) tag membership lookup (not serialized)
    _synonym_set: Set[str] = field(default_factory=set, repr=False, compare=False)

    def __post_init__(self):
        """Normalize the synonym group after initialization."""
        # Ensure name is lowercase with underscores
        self.name = _normalize_name(self.name)
        # Remove namespace prefixes and deduplicate while preserving order
        seen: Set[str] = set()
        deduped: List[str] = []
        for s in self.synonyms:
            stripped = self._strip_namespace(s)
            key = stripped.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(stripped)
        self.synonyms = deduped
        # Reuse the set we already built for O(1) lookup
        self._synonym_set = seen

    @staticmethod
    def _strip_namespace(tag: str) -> str:
        """Remove namespace prefix from tag (e.g., 'us-gaap:Revenue' -> 'Revenue')."""
        if ":" in tag:
            return tag.split(":", 1)[1]
        # Handle underscore format (us-gaap_Revenue)
        if "_" in tag:
            parts = tag.split("_", 1)
            if parts[0].replace("-", "") in ("usgaap", "dei", "srt", "ifrs"):
                return parts[1]
        return tag

    def get_tags_with_namespace(self, namespace: Optional[str] = None) -> List[str]:
        """
        Get synonyms with namespace prefix.

        Args:
            namespace: Namespace to use (default: self.namespace)

        Returns:
            List of tags with namespace prefix
        """
        ns = namespace or self.namespace
        return [f"{ns}:{tag}" for tag in self.synonyms]

    def contains_tag(self, tag: str) -> bool:
        """
        Check if this group contains the given tag.

        Args:
            tag: XBRL tag to check (with or without namespace)

        Returns:
            True if tag is in this group's synonyms
        """
        normalized = self._strip_namespace(tag).lower()
        return normalized in self._synonym_set  # O(1) lookup

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "synonyms": self.synonyms,
            "description": self.description,
            "namespace": self.namespace,
            "priority_order": self.priority_order,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SynonymGroup":
        """Create SynonymGroup from dictionary."""
        return cls(
            name=data["name"],
            synonyms=data["synonyms"],
            description=data.get("description", ""),
            namespace=data.get("namespace", "us-gaap"),
            priority_order=data.get("priority_order", "listed"),
            category=data.get("category", ""),
        )


@dataclass
class ConceptInfo:
    """
    Information about an identified concept from a tag lookup.

    Attributes:
        name: Canonical concept name
        tag: The original tag that was looked up
        group: The full SynonymGroup containing this concept
        match_type: How the match was found ('exact', 'normalized', 'fuzzy')
    """

    name: str
    tag: str
    group: SynonymGroup
    match_type: str = "exact"

    @property
    def synonyms(self) -> List[str]:
        """Get all synonyms for this concept."""
        return self.group.synonyms

    @property
    def description(self) -> str:
        """Get concept description."""
        return self.group.description

    @property
    def category(self) -> str:
        """Get concept category."""
        return self.group.category


def _strip_tag_from_concept_mapping(tag: str) -> str:
    """
    Strip namespace prefix from concept_mappings tag format (us-gaap_Revenue -> Revenue).

    EdgarTools concept_mappings uses 'us-gaap_TagName' or 'orcl_TagName' format.
    """
    if "_" in tag:
        parts = tag.split("_", 1)
        if parts[0].lower().replace("-", "") in ("usgaap", "dei", "srt", "ifrs", "orcl"):
            return parts[1]
    if ":" in tag:
        return tag.split(":", 1)[1]
    return tag


def _get_synonym_groups_from_concept_mappings() -> List[SynonymGroup]:
    """
    Build SynonymGroups from EdgarTools concept_mappings.json (display_label -> [tags]).

    This is EdgarTools' primary extraction source. Each display label maps to XBRL tags.
    Concept names are normalized display labels (lowercase, underscores).
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    cm_path = os.path.join(module_dir, "standardization", "concept_mappings.json")
    if not os.path.exists(cm_path):
        cm_path = os.path.join(module_dir, "concept_mappings.json")
    try:
        with open(cm_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.warning("Could not load concept_mappings for SynonymGroups: %s", e)
        return []

    groups: List[SynonymGroup] = []
    for display_label, tags in data.items():
        if display_label.startswith("_") or not isinstance(tags, list):
            continue
        concept_name = _normalize_name(display_label)
        stripped = [_strip_tag_from_concept_mapping(t) for t in tags]
        cat = ""
        if any(k in concept_name for k in ("revenue", "cost", "gross_profit", "operating", "income", "expense", "tax")):
            cat = "income_statement"
        elif any(k in concept_name for k in ("assets", "liabilities", "equity", "receivable", "payable", "inventory")):
            cat = "balance_sheet"
        elif any(k in concept_name for k in ("cash", "operating_activities", "investing", "financing", "dividends", "payments")):
            cat = "cash_flow"
        groups.append(
            SynonymGroup(
                name=concept_name,
                synonyms=stripped,
                description=display_label,
                category=cat,
            )
        )
    return groups


def _get_synonym_groups_from_gaap() -> List[SynonymGroup]:
    """
    Build SynonymGroups by inverting gaap_mappings (tag -> standard_tags).

    Provides ~3000-tag coverage. Used to supplement concept_mappings for
    concepts present in gaap/display_names but not in concept_mappings.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    gaap_path = os.path.join(module_dir, "standardization", "gaap_mappings.json")
    if not os.path.exists(gaap_path):
        gaap_path = os.path.join(module_dir, "gaap_mappings.json")
    display_path = os.path.join(module_dir, "standardization", "display_names.json")
    if not os.path.exists(display_path):
        display_path = os.path.join(module_dir, "display_names.json")

    try:
        with open(gaap_path, "r", encoding="utf-8") as f:
            gaap = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.warning("Could not load gaap_mappings for SynonymGroups: %s", e)
        return []

    try:
        with open(display_path, "r", encoding="utf-8") as f:
            display_names = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.warning("Could not load display_names for SynonymGroups: %s", e)
        display_names = {}

    first_val = next(iter(gaap.values()), None)
    is_tag_centric = isinstance(first_val, dict) and "standard_tags" in first_val

    concept_to_tags: Dict[str, List[str]] = {}
    for key, val in gaap.items():
        if is_tag_centric and isinstance(val, dict):
            std_tags = val.get("standard_tags", val.get("standard_tag", []))
            if isinstance(std_tags, str):
                std_tags = [std_tags]
            for std_tag in std_tags:
                if std_tag not in concept_to_tags:
                    concept_to_tags[std_tag] = []
                if key not in concept_to_tags[std_tag]:
                    concept_to_tags[std_tag].append(key)
        else:
            if isinstance(val, list):
                for tag in val:
                    if key not in concept_to_tags:
                        concept_to_tags[key] = []
                    if tag not in concept_to_tags[key]:
                        concept_to_tags[key].append(tag)

    concept_name_to_tags: Dict[str, Set[str]] = {}
    for standard_tag, tags_list in concept_to_tags.items():
        display_name = display_names.get(standard_tag, standard_tag)
        concept_name = _normalize_name(display_name)
        if concept_name not in concept_name_to_tags:
            concept_name_to_tags[concept_name] = set()
        concept_name_to_tags[concept_name].update(tags_list)

    groups: List[SynonymGroup] = []
    for concept_name, tags_set in concept_name_to_tags.items():
        if any(k in concept_name for k in ("revenue", "cost", "gross_profit", "operating", "income", "expense")):
            category = "income_statement"
        elif any(k in concept_name for k in ("assets", "liabilities", "equity", "receivable", "payable")):
            category = "balance_sheet"
        elif "capital" in concept_name or "expense" in concept_name:
            category = "cash_flow" if "capital" in concept_name or "payments" in concept_name else "income_statement"
        else:
            category = ""
        groups.append(
            SynonymGroup(
                name=concept_name,
                synonyms=list(tags_set),
                description=concept_name.replace("_", " ").title(),
                category=category,
            )
        )
    return groups


def _get_builtin_groups_cached() -> List[SynonymGroup]:
    """
    Get synonym groups aligned with EdgarTools (cached at module level).

    Primary: concept_mappings.json (EdgarTools extraction source, display_label -> tags).
    Supplement: gaap_mappings inversion for concepts in gaap/display_names but not
    in concept_mappings. No builtin overrides - tags and concepts match EdgarTools.
    """
    global _builtin_groups_cache
    if _builtin_groups_cache is not None:
        return _builtin_groups_cache

    # Primary: concept_mappings (EdgarTools extraction source)
    cm_groups = _get_synonym_groups_from_concept_mappings()
    cm_by_name = {g.name: g for g in cm_groups}
    cm_concept_names = set(cm_by_name)

    # Supplement: gaap-derived for concepts only in gaap (broader tag coverage)
    gaap_groups = _get_synonym_groups_from_gaap()
    result_by_name = dict(cm_by_name)
    for g in gaap_groups:
        if g.name not in cm_concept_names:
            result_by_name[g.name] = g

    _builtin_groups_cache = list(result_by_name.values())
    return _builtin_groups_cache


# Legacy builtin groups removed - now using EdgarTools concept_mappings + gaap only.


class SynonymGroups:
    """
    Centralized manager for XBRL tag synonym groups.

    Provides a unified interface for managing synonym groups that can be used
    across all financial statement classes. This is the foundation for the
    shared standardization infrastructure.

    The manager comes pre-loaded with 40+ common financial concept groups
    (revenue, net_income, capex, etc.) and supports user-defined custom groups.

    Example:
    >>> synonyms = SynonymGroups()
    >>>
    >>> # Get pre-built group
    >>> revenue = synonyms.get_group('revenue')
    >>> print(revenue.synonyms[:3])
    ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'Revenue']
    >>>
    >>> # Identify concept from tag
    >>> info = synonyms.identify_concept('NetIncomeLoss')
    >>> print(info.name)
    'net_income'
    >>>
    >>> # Register custom group
    >>> synonyms.register_group(
    ...     name='my_revenue',
    ...     synonyms=['CustomRevenue1', 'CustomRevenue2']
    ... )

    Attributes:
        _groups: Dictionary of name -> SynonymGroup
        _tag_index: Reverse index of tag -> group name for fast lookups
    """

    def __init__(self, load_builtin: bool = True):
        """
        Initialize SynonymGroups manager.

        Args:
            load_builtin: Whether to load pre-built synonym groups (default: True)
        """
        self._groups: Dict[str, SynonymGroup] = {}
        self._tag_index: Dict[
            str, List[str]
        ] = {}  # tag -> [group_name1, group_name2, ...]
        self._user_groups: Dict[str, SynonymGroup] = {}  # Track user-defined groups

        if load_builtin:
            self._load_builtin_groups()

    def _load_builtin_groups(self) -> None:
        """Load pre-built synonym groups for common financial concepts."""
        builtin_groups = _get_builtin_groups_cached()
        for group in builtin_groups:
            self._register_group_internal(group, is_user_defined=False)

    def _register_group_internal(
        self, group: SynonymGroup, is_user_defined: bool = False
    ) -> None:
        """
        Internal method to register a group and update indices.

        Tags can belong to multiple groups (multi-group membership). This allows
        concepts like DepreciationAndAmortization to appear in both income_statement
        and cash_flow contexts.

        Args:
            group: The SynonymGroup to register
            is_user_defined: Whether this is a user-defined group
        """
        self._groups[group.name] = group

        if is_user_defined:
            self._user_groups[group.name] = group

        # Update reverse index - append to list to support multi-group membership
        for tag in group.synonyms:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = []
            # Avoid duplicates if same group is re-registered
            if group.name not in self._tag_index[tag_lower]:
                self._tag_index[tag_lower].append(group.name)

    def get_group(self, name: str) -> Optional[SynonymGroup]:
        """
        Get a synonym group by name.

        Args:
            name: The canonical name of the concept (e.g., 'revenue', 'net_income')

        Returns:
            SynonymGroup if found, None otherwise

        Example:
        >>> synonyms = SynonymGroups()
        >>> revenue = synonyms.get_group('revenue')
        >>> print(revenue.synonyms[:2])
        ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues']
        """
        normalized = _normalize_name(name)
        return self._groups.get(normalized)

    def get_synonyms(self, name: str) -> List[str]:
        """
        Get the list of synonyms for a concept.

        Convenience method that returns just the synonym list.

        Args:
            name: The canonical name of the concept

        Returns:
            List of synonym tags, or empty list if not found

        Example:
        >>> synonyms = SynonymGroups()
        >>> tags = synonyms.get_synonyms('capex')
        >>> print(tags[:2])
        ['PaymentsToAcquirePropertyPlantAndEquipment', 'CapitalExpenditures']
        """
        group = self.get_group(name)
        return group.synonyms if group else []

    def identify_concept(
        self, tag: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[ConceptInfo]:
        """
        Identify which concept a tag belongs to (returns first match).

        Performs reverse lookup to find the canonical concept name
        for a given XBRL tag. If the tag belongs to multiple groups,
        returns the first one (order of registration).

        For tags that may belong to multiple concepts, use identify_concepts()
        to get all matches.

        Args:
            tag: XBRL tag to identify (with or without namespace prefix)
            context: Optional context for disambiguation (section, statement_type, etc.)

        Returns:
            ConceptInfo if tag is recognized, None otherwise

        Example:
        >>> synonyms = SynonymGroups()
        >>> info = synonyms.identify_concept('us-gaap:NetIncomeLoss')
        >>> print(info.name)
        'net_income'
        >>> print(info.description)
        'Net income/loss'
        """
        # Try reverse index first (if available)
        try:
            from financial4all.xbrl.standardization.reverse_index import (
                get_reverse_index,
            )

            reverse_index = get_reverse_index()
            standard_concept = reverse_index.get_standard_concept(tag, context)
            if standard_concept:
                # Find matching group
                normalized_concept = (
                    standard_concept.lower().replace(" ", "_").replace("-", "_")
                )
                group = self._groups.get(normalized_concept)
                if group:
                    return ConceptInfo(
                        name=normalized_concept,
                        tag=tag,
                        group=group,
                        match_type="exact",
                    )
        except (ImportError, AttributeError):
            pass

        # Fallback to existing logic
        # Normalize tag
        normalized = SynonymGroup._strip_namespace(tag).lower()

        # Look up in index - returns list of group names
        group_names = self._tag_index.get(normalized, [])
        if group_names:
            group_name = group_names[0]  # Return first match
            group = self._groups[group_name]
            return ConceptInfo(
                name=group_name, tag=tag, group=group, match_type="exact"
            )

        return None

    def identify_concepts(self, tag: str) -> List[ConceptInfo]:
        """
        Identify all concepts a tag belongs to.

        Performs reverse lookup to find all canonical concept names
        for a given XBRL tag. Tags can belong to multiple groups
        (multi-group membership) when they have different meanings
        in different contexts.

        Args:
            tag: XBRL tag to identify (with or without namespace prefix)

        Returns:
            List of ConceptInfo for all matching groups (empty if not recognized)
        """
        # Normalize tag
        normalized = SynonymGroup._strip_namespace(tag).lower()

        # Look up in index - returns list of group names
        group_names = self._tag_index.get(normalized, [])

        results = []
        for group_name in group_names:
            group = self._groups[group_name]
            results.append(
                ConceptInfo(name=group_name, tag=tag, group=group, match_type="exact")
            )

        return results

    def register_group(
        self,
        name: str,
        synonyms: List[str],
        description: str = "",
        namespace: str = "us-gaap",
        priority_order: str = "listed",
        category: str = "",
    ) -> SynonymGroup:
        """
        Register a custom synonym group.

        User-defined groups take precedence over built-in groups
        if there are naming conflicts.

        Args:
            name: Canonical name for the concept
            synonyms: List of XBRL tags that represent this concept
            description: Human-readable description
            namespace: Default namespace for tags
            priority_order: How to order synonyms ('listed', 'frequency', 'specificity')
            category: Financial statement category

        Returns:
            The registered SynonymGroup
        """
        group = SynonymGroup(
            name=name,
            synonyms=synonyms,
            description=description,
            namespace=namespace,
            priority_order=priority_order,
            category=category,
        )
        self._register_group_internal(group, is_user_defined=True)
        log.info(f"Registered custom synonym group: {group.name}")
        return group

    def unregister_group(self, name: str) -> bool:
        """
        Remove a user-defined synonym group.

        Only user-defined groups can be removed. Built-in groups
        cannot be unregistered.

        Args:
            name: Name of the group to remove

        Returns:
            True if group was removed, False if not found or is built-in
        """
        normalized = _normalize_name(name)

        if normalized not in self._user_groups:
            log.warning(f"Cannot unregister group '{name}': not a user-defined group")
            return False

        group = self._groups.pop(normalized, None)
        self._user_groups.pop(normalized, None)

        if group:
            # Remove from index - handle list-based index
            for tag in group.synonyms:
                tag_lower = tag.lower()
                if tag_lower in self._tag_index:
                    group_list = self._tag_index[tag_lower]
                    if normalized in group_list:
                        group_list.remove(normalized)
                    # Clean up empty lists
                    if not group_list:
                        del self._tag_index[tag_lower]
            return True

        return False

    def list_groups(self, category: Optional[str] = None) -> List[str]:
        """
        List all available synonym group names.

        Args:
            category: Optional filter by category (e.g., 'income_statement', 'balance_sheet')

        Returns:
            List of group names, sorted alphabetically
        """
        if category:
            return sorted(
                [
                    name
                    for name, group in self._groups.items()
                    if group.category == category
                ]
            )
        return sorted(self._groups.keys())

    def export_to_json(self, file_path: Union[str, Path]) -> None:
        """
        Export user-defined groups to JSON file.

        Args:
            file_path: Path to JSON file
        """
        data = {"groups": [group.to_dict() for group in self._user_groups.values()]}

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        log.info(f"Exported {len(self._user_groups)} groups to {file_path}")

    def import_from_json(self, file_path: Union[str, Path]) -> None:
        """
        Import user-defined groups from JSON file.

        Args:
            file_path: Path to JSON file
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        groups_imported = 0
        for group_dict in data.get("groups", []):
            group = SynonymGroup.from_dict(group_dict)
            self._register_group_internal(group, is_user_defined=True)
            groups_imported += 1

        log.info(f"Imported {groups_imported} groups from {file_path}")

    def get_standard_concept(
        self, tag: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Get standard concept name for a tag using reverse index (if available).

        Args:
            tag: XBRL tag to look up
            context: Optional context for disambiguation

        Returns:
            Standard concept name or None
        """
        try:
            from financial4all.xbrl.standardization.reverse_index import (
                get_reverse_index,
            )

            reverse_index = get_reverse_index()
            return reverse_index.get_standard_concept(tag, context)
        except (ImportError, AttributeError):
            # Fallback to identify_concept
            info = self.identify_concept(tag, context)
            return info.name if info else None

    def get_display_name(
        self, tag: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Get user-friendly display name for a tag.

        Args:
            tag: XBRL tag to look up
            context: Optional context for disambiguation

        Returns:
            Display name or None
        """
        try:
            from financial4all.xbrl.standardization.reverse_index import (
                get_reverse_index,
            )

            reverse_index = get_reverse_index()
            return reverse_index.get_display_name(tag, context)
        except (ImportError, AttributeError):
            # Fallback to concept name
            info = self.identify_concept(tag, context)
            if info:
                # Try to get display name from StandardConcept enum
                try:
                    from financial4all.xbrl.standardization.standard_concepts import (
                        StandardConcept,
                    )

                    for concept in StandardConcept:
                        if (
                            concept.value.lower().replace(" ", "_").replace("-", "_")
                            == info.name
                        ):
                            return concept.value
                except ImportError:
                    pass
                return info.name.replace("_", " ").title()
            return None

    def is_ambiguous(self, tag: str) -> bool:
        """
        Check if a tag is ambiguous (maps to multiple concepts).

        Args:
            tag: XBRL tag to check

        Returns:
            True if ambiguous, False otherwise
        """
        try:
            from financial4all.xbrl.standardization.reverse_index import (
                get_reverse_index,
            )

            reverse_index = get_reverse_index()
            return reverse_index.is_ambiguous(tag)
        except (ImportError, AttributeError):
            # Fallback: check if tag appears in multiple groups
            normalized = SynonymGroup._strip_namespace(tag).lower()
            group_names = self._tag_index.get(normalized, [])
            return len(group_names) > 1


def get_synonym_groups() -> SynonymGroups:
    """
    Get the default singleton SynonymGroups instance.

    Returns:
        Global SynonymGroups instance
    """
    global _default_instance
    if _default_instance is None:
        _default_instance = SynonymGroups()
    return _default_instance


# ═══════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY LAYER
# ═══════════════════════════════════════════════════════════════════


class StandardizationStore:
    """
    Legacy compatibility layer for StandardizationStore.

    This class maintains backward compatibility with existing code
    while delegating to the new SynonymGroups system.

    Maps standardized concept names to XBRL concept names and vice versa.
    """

    def __init__(self):
        """Initialize standardization store."""
        self._synonym_groups = get_synonym_groups()

    def add_mapping(self, standard_name: str, xbrl_concepts: List[str]) -> None:
        """
        Add a standardization mapping.

        Args:
            standard_name: Standardized concept name
            xbrl_concepts: List of XBRL concept names that map to this standard name
        """
        # Convert to normalized name
        normalized = _normalize_name(standard_name)

        # Check if group already exists
        existing_group = self._synonym_groups.get_group(normalized)
        if existing_group:
            # Merge synonyms
            merged_synonyms = list(set(existing_group.synonyms + xbrl_concepts))
            # Unregister old and register new
            self._synonym_groups.unregister_group(normalized)
            self._synonym_groups.register_group(
                name=normalized,
                synonyms=merged_synonyms,
                description=existing_group.description,
                category=existing_group.category,
            )
        else:
            # Create new group
            self._synonym_groups.register_group(
                name=normalized,
                synonyms=xbrl_concepts,
                description=f"Standardized mapping for {standard_name}",
                category="",
            )

    def get_standard_name(self, xbrl_concept: str) -> Optional[str]:
        """
        Get standardized name for an XBRL concept.

        Args:
            xbrl_concept: XBRL concept name

        Returns:
            Standardized name or None if not found
        """
        info = self._synonym_groups.identify_concept(xbrl_concept)
        return info.name if info else None

    def get_xbrl_concepts(self, standard_name: str) -> List[str]:
        """
        Get XBRL concept names for a standardized name.

        Args:
            standard_name: Standardized concept name

        Returns:
            List of XBRL concept names
        """
        return self._synonym_groups.get_synonyms(standard_name)


# Global standardization store instance (legacy)
_standardization_store: Optional[StandardizationStore] = None


def get_default_store() -> StandardizationStore:
    """
    Get the default standardization store instance (legacy compatibility).

    Returns:
        StandardizationStore instance
    """
    global _standardization_store
    if _standardization_store is None:
        _standardization_store = StandardizationStore()
    return _standardization_store
