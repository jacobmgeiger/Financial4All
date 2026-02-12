# financial4all/xbrl/facts.py
"""
XBRL fact extraction, storage, and querying.

Fact represents a single numeric or string fact (concept, value, unit, period,
dimensions, form). FactSet is a collection with indexes by concept, period,
and concept+period; it can be built from SEC company facts JSON (from_company_facts)
and supports filtering (annual, by form), get_by_concept, get_all_facts_for_concept
(including namespace/extension variants), and synonym discovery. Used by all
statement classes to resolve metrics from raw XBRL.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union, Tuple, Callable
from datetime import datetime, date
import re
from functools import lru_cache

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from financial4all.xbrl.periods import Period, PeriodType
from financial4all.xbrl.entity_info import (
    extract_dei_facts,
    extract_dei_facts_from_xbrl,
    build_entity_info,
    EntityInfo,
)
from financial4all.xbrl.standardization import get_synonym_groups
from financial4all.core import log


@dataclass
class Fact:
    """
    Represents a single XBRL fact.

    Attributes:
        concept: XBRL concept name (e.g., "us-gaap_Revenues")
        value: Fact value
        unit: Unit of measurement (e.g., "USD", "shares")
        period: Period for this fact
        dimensions: Dimensional information (segments, breakdowns)
        form: Form type (e.g., "10-K", "10-Q")
        frame: Frame identifier (e.g., "CY2023Q1")
        filed: Filing date
    """

    concept: str
    value: Any
    unit: str
    period: Period
    dimensions: Dict[str, Any] = field(default_factory=dict)
    form: Optional[str] = None
    frame: Optional[str] = None
    filed: Optional[datetime] = None

    def is_annual_10k(self) -> bool:
        """Check if fact is from annual 10-K filing."""
        return (
            self.form == "10-K"
            and self.period.period_type == PeriodType.DURATION
            and self.period.is_annual()
            and (not self.frame or "Q" not in self.frame)
        )

    def is_annual_10k_or_amended(self) -> bool:
        """
        Check if fact is from annual 10-K or 10-K/A (amended) filing.

        Amended filings often contain restated/corrected data and should be
        accepted for improved historical coverage (EdgarTools alignment).
        """
        return (
            self.form in ("10-K", "10-K/A")
            and self.period.period_type == PeriodType.DURATION
            and self.period.is_annual()
            and (not self.frame or "Q" not in self.frame)
        )

    def __repr__(self) -> str:
        """String representation of Fact."""
        return f"Fact(concept={self.concept}, value={self.value}, unit={self.unit}, period={self.period})"


class FactSet:
    """
    Collection of XBRL facts with filtering and query capabilities.

    Uses indexes for efficient O(1) lookups by concept, period, or concept-period combination.
    """

    def __init__(
        self,
        facts: Optional[List[Fact]] = None,
        entity_info: Optional[EntityInfo] = None,
    ):
        """
        Initialize FactSet.

        Args:
            facts: Optional list of facts to initialize with
            entity_info: Optional entity information extracted from DEI facts
        """
        self.facts: List[Fact] = facts or []
        self._entity_info: Optional[EntityInfo] = entity_info

        # Indexes for O(1) lookups
        self._concept_index: Dict[str, List[Fact]] = {}  # concept -> list of facts
        self._period_index: Dict[str, List[Fact]] = {}  # period_end -> list of facts
        self._concept_period_index: Dict[
            Tuple[str, str], List[Fact]
        ] = {}  # (concept, period_end) -> list of facts

        # Build indexes if facts are provided
        if self.facts:
            self._build_indexes()

    def _build_indexes(self) -> None:
        """Build indexes for efficient fact lookups."""
        self._concept_index.clear()
        self._period_index.clear()
        self._concept_period_index.clear()

        for fact in self.facts:
            # Index by concept
            concept = fact.concept
            if concept not in self._concept_index:
                self._concept_index[concept] = []
            self._concept_index[concept].append(fact)

            # Index by period end
            period_key = str(fact.period.end)
            if period_key not in self._period_index:
                self._period_index[period_key] = []
            self._period_index[period_key].append(fact)

            # Index by concept-period combination
            concept_period_key = (concept, period_key)
            if concept_period_key not in self._concept_period_index:
                self._concept_period_index[concept_period_key] = []
            self._concept_period_index[concept_period_key].append(fact)

    def add(self, fact: Fact) -> None:
        """Add a fact to the set and update indexes."""
        self.facts.append(fact)

        # Update indexes
        concept = fact.concept
        period_key = str(fact.period.end)

        # Update concept index
        if concept not in self._concept_index:
            self._concept_index[concept] = []
        self._concept_index[concept].append(fact)

        # Update period index
        if period_key not in self._period_index:
            self._period_index[period_key] = []
        self._period_index[period_key].append(fact)

        # Update concept-period index
        concept_period_key = (concept, period_key)
        if concept_period_key not in self._concept_period_index:
            self._concept_period_index[concept_period_key] = []
        self._concept_period_index[concept_period_key].append(fact)

    def filter_by_form(self, form: str) -> "FactSet":
        """Filter facts by form type."""
        if not form:
            return FactSet([], entity_info=self._entity_info)
        filtered = [f for f in self.facts if f.form == form]
        return FactSet(filtered, entity_info=self._entity_info)

    def filter_by_concept(self, concept: str) -> "FactSet":
        """Filter facts by concept name."""
        if not concept:
            return FactSet([], entity_info=self._entity_info)
        # Use index for O(1) lookup if available
        if concept in self._concept_index:
            filtered = self._concept_index[concept]
        else:
            filtered = [f for f in self.facts if f.concept == concept]
        return FactSet(filtered, entity_info=self._entity_info)

    def filter_annual_10k(self) -> "FactSet":
        """Filter to only annual 10-K facts."""
        filtered = [f for f in self.facts if f.is_annual_10k()]
        return FactSet(filtered, entity_info=self._entity_info)

    def filter_annual(self) -> "FactSet":
        """
        Filter to annual facts from any form type (10-K, 10-K/A, etc.).

        This is more inclusive than filter_annual_10k() and captures more historical data
        by including annual data from amended filings and other form types.
        """
        filtered = [
            f
            for f in self.facts
            if f.period.period_type == PeriodType.DURATION
            and f.period.is_annual()
            and (not f.frame or "Q" not in str(f.frame))  # Exclude quarterly frames
        ]
        return FactSet(filtered, entity_info=self._entity_info)

    @staticmethod
    def _dims_key(dimensions: Optional[Dict]) -> str:
        """Stable key for dimensions to distinguish consolidated vs segment facts."""
        if not dimensions:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(dimensions.items()))

    @classmethod
    def merge(
        cls,
        fact_sets: List["FactSet"],
        prefer_most_recent: bool = True,
        entity_info: Optional[EntityInfo] = None,
    ) -> "FactSet":
        """
        Merge multiple FactSets into one, preserving both consolidated and dimensional facts.

        Uses (concept, period_end, dimensions_key) as key so segment breakdowns
        (e.g. Revenue — Graphics) coexist with consolidated totals (Revenue).
        EdgarTools-style: keep all facts; resolution prefers consolidated at display time.

        Args:
            fact_sets: List of FactSet instances to merge (most recent first)
            prefer_most_recent: If True, for same (concept, period, dims),
                keep the fact from the earliest FactSet (most recent filing).
            entity_info: Optional EntityInfo; uses first non-None from fact_sets.

        Returns:
            Merged FactSet
        """
        if not fact_sets:
            return cls([], entity_info=entity_info)

        # Key includes dimensions so we keep both consolidated and segment facts
        seen: Dict[Tuple[str, str, str], Fact] = {}

        for fs in fact_sets:
            if not fs or not fs.facts:
                continue
            for fact in fs.facts:
                period_key = str(fact.period.end)
                dims_key = cls._dims_key(fact.dimensions)
                key = (fact.concept, period_key, dims_key)
                if key not in seen:
                    seen[key] = fact
                elif not prefer_most_recent:
                    seen[key] = fact

        merged_facts = list(seen.values())
        ei = entity_info
        if ei is None:
            for fs in fact_sets:
                if fs._entity_info is not None:
                    ei = fs._entity_info
                    break

        return cls(merged_facts, entity_info=ei)

    @staticmethod
    def _norm_concept_for_supplement(concept: str) -> str:
        """Normalize concept for supplement matching (us-gaap_Revenues == Revenues)."""
        if not concept:
            return concept
        c = concept.strip()
        for prefix in ("us-gaap_", "us-gaap:", "ifrs-full_", "ifrs-full:"):
            if c.lower().startswith(prefix.lower()):
                return c[len(prefix) :].lstrip()
        return c

    def supplement_from(
        self,
        other: "FactSet",
        filter_fn: Optional[Callable[[Fact], bool]] = None,
    ) -> int:
        """
        Add facts from another FactSet only for missing (concept, period, dimensions) keys.

        Never replaces existing facts—preserves dimensional (segment) facts for
        _extract_segment_breakdowns. Resolution prefers consolidated at display time.

        Args:
            other: FactSet to pull facts from (e.g. from company facts API)
            filter_fn: Optional predicate; only add facts passing this filter.

        Returns:
            Number of facts added
        """
        def key(f: Fact) -> Tuple[str, str, str]:
            return (
                self._norm_concept_for_supplement(f.concept),
                str(f.period.end),
                self._dims_key(f.dimensions),
            )

        existing_keys: Set[Tuple[str, str, str]] = {key(f) for f in self.facts}
        added = 0

        for fact in other.facts:
            if filter_fn is not None and not filter_fn(fact):
                continue
            nkey = key(fact)
            if nkey not in existing_keys:
                self.add(fact)
                existing_keys.add(nkey)
                added += 1

        if added:
            log.debug(f"Supplemented FactSet with {added} facts from company facts")
        return added

    def get_unique_concepts(self) -> Set[str]:
        """Get set of unique concept names."""
        return {f.concept for f in self.facts}

    def get_by_concept(self, concept: str) -> List[Fact]:
        """
        Get all facts for a specific concept.

        Supports fuzzy matching for namespace variations:
        - Tries exact match first (uses index for O(1) lookup)
        - Then tries with 'us-gaap_' prefix if not found
        - Then tries without 'us-gaap_' prefix if concept starts with it

        Args:
            concept: XBRL concept name (e.g., "us-gaap_Revenues" or "Revenues")

        Returns:
            List of matching facts (empty list if concept is invalid or not found)
        """
        if not concept or not isinstance(concept, str):
            log.warning(f"Invalid concept provided to get_by_concept: {concept}")
            return []

        concept = concept.strip()
        if not concept:
            return []

        # Try exact match first using index
        if concept in self._concept_index:
            return self._concept_index[concept].copy()

        # Try with namespace prefix if not already present
        if not concept.startswith("us-gaap_"):
            prefixed = f"us-gaap_{concept}"
            if prefixed in self._concept_index:
                return self._concept_index[prefixed].copy()

        # Try without namespace prefix if it's present
        if concept.startswith("us-gaap_"):
            unprefixed = concept.replace("us-gaap_", "", 1)
            if unprefixed in self._concept_index:
                return self._concept_index[unprefixed].copy()

        return []

    def get_concepts_by_pattern(self, pattern: str) -> Set[str]:
        """
        Get all concepts that match a pattern (case-insensitive substring match).

        Args:
            pattern: Pattern to search for (e.g., "Interest" to find all interest-related concepts)

        Returns:
            Set of matching concept names (empty set if pattern is invalid)
        """
        if not pattern or not isinstance(pattern, str):
            log.warning(
                f"Invalid pattern provided to get_concepts_by_pattern: {pattern}"
            )
            return set()

        pattern_lower = pattern.lower().strip()
        if not pattern_lower:
            return set()

        # Pre-compile lowercase concept set for O(1) lookup optimization
        return {f.concept for f in self.facts if pattern_lower in f.concept.lower()}

    def find_synonym_concepts(self, concept: str) -> Set[str]:
        """
        Find synonym concepts using the SynonymGroups standardization system.

        Helps discover concepts that might be named differently but represent
        the same financial metric. Uses the comprehensive SynonymGroups system
        for accurate synonym discovery.

        Args:
            concept: Base concept name to find synonyms for

        Returns:
            Set of synonym concept names found in this FactSet (empty set if concept is invalid)
        """
        if not concept or not isinstance(concept, str):
            log.warning(f"Invalid concept provided to find_synonym_concepts: {concept}")
            return set()

        synonyms = set()
        synonym_groups = get_synonym_groups()

        # Try to identify the concept using SynonymGroups
        concept_info = synonym_groups.identify_concept(concept)
        if concept_info:
            # Found a match - get all synonyms from the group
            synonym_tags = concept_info.synonyms

            # Check which synonyms actually exist in this FactSet
            all_concepts_in_factset = {f.concept for f in self.facts}

            for tag in synonym_tags:
                # Try with and without namespace prefix
                for variant in [tag, f"us-gaap_{tag}", f"us-gaap:{tag}"]:
                    if variant in all_concepts_in_factset:
                        synonyms.add(variant)
        else:
            # Fallback: Use pattern matching for concepts not in SynonymGroups
            # Remove namespace prefix for matching
            base_concept = (
                concept.replace("us-gaap_", "").replace("us-gaap:", "").lower()
            )

            # Try to find concepts with similar names using substring matching
            for fact in self.facts:
                fact_concept_lower = fact.concept.lower()
                # Check if concepts share significant substrings
                if (
                    base_concept in fact_concept_lower
                    or fact_concept_lower in base_concept
                    or self._concepts_similar(base_concept, fact_concept_lower)
                ):
                    synonyms.add(fact.concept)

        # Also do direct substring matching for similar concepts (fallback)
        if not synonyms:
            # Remove namespace prefix for matching
            base_concept = (
                concept.replace("us-gaap_", "").replace("us-gaap:", "").lower()
            )

            # Remove common suffixes/prefixes and match
            base_stem = base_concept
            for suffix in ["net", "loss", "expense", "income", "revenue"]:
                if base_stem.endswith(suffix):
                    base_stem = base_stem[: -len(suffix)]
                    break

            if base_stem:
                stem_matches = self.get_concepts_by_pattern(base_stem)
                synonyms.update(stem_matches)

        # Remove the original concept from synonyms
        synonyms.discard(concept)

        return synonyms

    @staticmethod
    def _concepts_similar(concept1: str, concept2: str, threshold: float = 0.7) -> bool:
        """
        Check if two concepts are similar using simple heuristics.

        Args:
            concept1: First concept (normalized)
            concept2: Second concept (normalized)
            threshold: Similarity threshold (not used in simple version)

        Returns:
            True if concepts appear similar
        """
        # Simple similarity check: shared significant words
        words1 = set(concept1.split())
        words2 = set(concept2.split())

        if not words1 or not words2:
            return False

        # Check for significant overlap
        common_words = words1.intersection(words2)
        if len(common_words) >= min(2, len(words1), len(words2)):
            return True

        return False

    def has_reported_data(self, concept: str) -> bool:
        """
        Check if a concept has any reported facts.

        Args:
            concept: XBRL concept name

        Returns:
            True if at least one fact exists for this concept, False otherwise
        """
        return len(self.get_by_concept(concept)) > 0

    def _create_normalized_fact_key(
        self, element_id: str, context_ref: str, instance_id: Optional[int] = None
    ) -> str:
        """
        Create a normalized fact key using underscore format.

        Normalizes namespace prefixes (colon vs underscore) and handles duplicate facts
        with instance IDs. Format: element_id_context_ref[_instance_id]

        Args:
            element_id: Element/concept identifier (may include namespace prefix)
            context_ref: Context reference (period identifier)
            instance_id: Optional instance ID for duplicate facts

        Returns:
            Normalized key string
        """
        # Normalize namespace prefix (colon to underscore)
        normalized_element_id = element_id
        if ":" in element_id:
            prefix, name = element_id.split(":", 1)
            normalized_element_id = f"{prefix}_{name}"

        # Create base key
        base_key = f"{normalized_element_id}_{context_ref}"

        # Add instance ID if provided
        if instance_id is not None:
            return f"{base_key}_{instance_id}"

        return base_key

    def get_all_facts_for_concept(
        self,
        concept: str,
        include_variants: bool = True,
        exclude_structural: bool = False,
    ) -> List[Fact]:
        """
        Get all facts for a concept with comprehensive namespace variant matching.

        Searches for concept with all namespace variations and returns ALL facts
        regardless of form/frame initially. This is more comprehensive than
        get_by_concept() which stops after first match. Includes custom extension
        taxonomies (e.g. nvda_PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets).

        Uses indexes for efficient lookup when possible.

        Args:
            concept: XBRL concept name (e.g., "Revenues" or "us-gaap_Revenues")
            include_variants: If True, tries multiple namespace variations
            exclude_structural: If True, filter out facts with structural concepts
                (Axis, Domain, Member, Table, Abstract) - EdgarTools-aligned

        Returns:
            List of all matching facts
        """
        if not concept:
            return []

        all_facts = []
        seen_facts = set()  # Track by normalized key to avoid duplicates

        # Generate all possible variations: exact, us-gaap, bare, and extension prefixes.
        # Filing XML stores extension elements as local name only (e.g. PaymentsToAcquire...
        # from nvda:PaymentsToAcquire...). EdgarTools alignment: ensure we find all variants.
        variations = [concept]
        if include_variants:
            if not concept.startswith("us-gaap_"):
                variations.append(f"us-gaap_{concept}")
            if concept.startswith("us-gaap_"):
                variations.append(concept.replace("us-gaap_", "", 1))
            # Local name for extension lookups (e.g. PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets)
            if concept.startswith("us-gaap_"):
                local_name = concept.replace("us-gaap_", "", 1)
            elif "_" in concept:
                local_name = concept.split("_", 1)[1]
            else:
                local_name = concept
            # Always add bare local name: XML parsing yields local-name-only for extension elements
            if local_name not in variations:
                variations.append(local_name)
            # Custom extension taxonomies: try prefix_localName for every prefix present in this FactSet
            extension_prefixes = set()
            for f in self.facts:
                if "_" in f.concept:
                    p = f.concept.split("_", 1)[0]
                    if p and p != "us-gaap":
                        extension_prefixes.add(p)
            for prefix in extension_prefixes:
                variant = f"{prefix}_{local_name}"
                if variant not in variations:
                    variations.append(variant)

        # Collect facts from all variations using indexes
        # Use local function references for performance
        seen_facts_add = seen_facts.add
        all_facts_append = all_facts.append
        create_key = self._create_normalized_fact_key

        for variant in variations:
            # Use index if available for O(1) lookup
            if variant in self._concept_index:
                facts = self._concept_index[variant]
            else:
                facts = [f for f in self.facts if f.concept == variant]

            for fact in facts:
                # Include dimensions in key so consolidated and segment facts both
                # surface (EdgarTools-style: segment extraction needs dimensional facts)
                period_key = str(fact.period.end)
                dims_key = self._dims_key(fact.dimensions)
                fact_key = create_key(fact.concept, period_key)
                if dims_key:
                    fact_key = f"{fact_key}_{dims_key}"

                if fact_key not in seen_facts:
                    # EdgarTools-aligned: optionally exclude structural elements
                    if exclude_structural:
                        from financial4all.xbrl.structural_filter import is_xbrl_structural_element
                        if is_xbrl_structural_element(
                            fact.concept,
                            getattr(fact, "label", None),
                        ):
                            continue
                    seen_facts_add(fact_key)
                    all_facts_append(fact)

        return all_facts

    def filter_by_period_range(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> "FactSet":
        """
        Filter facts by period range.

        Args:
            start_date: Optional start date (inclusive)
            end_date: Optional end date (inclusive)

        Returns:
            Filtered FactSet

        Raises:
            ValueError: If start_date > end_date
        """
        if start_date and end_date and start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must be <= end_date ({end_date})"
            )

        filtered = []
        for fact in self.facts:
            try:
                period_end = fact.period.end
                if isinstance(period_end, str):
                    try:
                        period_end = datetime.fromisoformat(
                            period_end.replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        log.debug(
                            f"Skipping fact with invalid period end: {period_end}"
                        )
                        continue
                elif isinstance(period_end, date) and not isinstance(
                    period_end, datetime
                ):
                    period_end = datetime.combine(period_end, datetime.min.time())

                if start_date and period_end < start_date:
                    continue
                if end_date and period_end > end_date:
                    continue

                filtered.append(fact)
            except Exception as e:
                log.debug(f"Error filtering fact by period range: {e}")
                continue

        return FactSet(filtered, entity_info=self._entity_info)

    def get_facts_by_period(self, period_end: Union[datetime, date, str]) -> List[Fact]:
        """
        Get all facts for a specific period end date.

        Uses period index for O(1) lookup when possible.

        Args:
            period_end: Period end date (datetime, date, or ISO string)

        Returns:
            List of facts matching the period (empty list if period is invalid)
        """
        # Normalize period_end to string key for index lookup
        period_key = None
        try:
            if isinstance(period_end, str):
                # Try to parse and normalize
                try:
                    dt = datetime.fromisoformat(period_end.replace("Z", "+00:00"))
                    period_key = str(dt.date())
                except (ValueError, AttributeError):
                    period_key = period_end  # Use as-is if parsing fails
            elif isinstance(period_end, (datetime, date)):
                if isinstance(period_end, datetime):
                    period_key = str(period_end.date())
                else:
                    period_key = str(period_end)
            else:
                log.warning(f"Invalid period_end type: {type(period_end)}")
                return []
        except Exception as e:
            log.debug(f"Error normalizing period_end: {e}")
            return []

        if not period_key:
            return []

        # Use index for O(1) lookup
        if period_key in self._period_index:
            return self._period_index[period_key].copy()

        # Fallback to linear search if not in index
        matching_facts = []
        for fact in self.facts:
            fact_end = fact.period.end
            if isinstance(fact_end, str):
                try:
                    fact_end = datetime.fromisoformat(
                        fact_end.replace("Z", "+00:00")
                    ).date()
                except (ValueError, AttributeError):
                    continue
            elif isinstance(fact_end, datetime):
                fact_end = fact_end.date()

            if fact_end == period_end:
                matching_facts.append(fact)

        return matching_facts

    def get_facts_by_concept_and_period(
        self, concept: str, period_end: Union[datetime, date, str]
    ) -> List[Fact]:
        """
        Get facts for a specific concept and period end date.

        Uses concept-period index for O(1) lookup.

        Args:
            concept: XBRL concept name
            period_end: Period end date (datetime, date, or ISO string)

        Returns:
            List of matching facts (empty list if not found)
        """
        if not concept:
            return []

        # Normalize period_end to string key
        period_key = None
        try:
            if isinstance(period_end, str):
                try:
                    dt = datetime.fromisoformat(period_end.replace("Z", "+00:00"))
                    period_key = str(dt.date())
                except (ValueError, AttributeError):
                    period_key = period_end
            elif isinstance(period_end, (datetime, date)):
                period_key = str(
                    period_end.date()
                    if isinstance(period_end, datetime)
                    else period_end
                )
            else:
                return []
        except Exception:
            return []

        if not period_key:
            return []

        # Try exact match first
        concept_period_key = (concept, period_key)
        if concept_period_key in self._concept_period_index:
            return self._concept_period_index[concept_period_key].copy()

        # Try namespace variations
        if not concept.startswith("us-gaap_"):
            prefixed_key = (f"us-gaap_{concept}", period_key)
            if prefixed_key in self._concept_period_index:
                return self._concept_period_index[prefixed_key].copy()

        if concept.startswith("us-gaap_"):
            unprefixed_key = (concept.replace("us-gaap_", "", 1), period_key)
            if unprefixed_key in self._concept_period_index:
                return self._concept_period_index[unprefixed_key].copy()

        return []

    def normalize_units(self, facts: List[Fact]) -> List[Fact]:
        """
        Normalize units in facts by converting to base units.

        Handles unit multipliers (thousands, millions, billions) and converts
        all values to base unit (USD, shares, etc.). Preserves original unit
        information in fact metadata.

        Args:
            facts: List of facts to normalize

        Returns:
            List of facts with normalized units and values
        """
        # Detect multipliers from unit strings
        multiplier_patterns = {
            "thousands": 1e3,
            "millions": 1e6,
            "billions": 1e9,
        }

        normalized_facts = []
        for fact in facts:
            normalized_fact = fact  # Start with original

            # Check if unit contains multiplier information
            unit_lower = fact.unit.lower()
            multiplier = 1.0

            for pattern, mult in multiplier_patterns.items():
                if pattern in unit_lower:
                    multiplier = mult
                    break

            # If multiplier found and value is numeric, apply it
            if multiplier != 1.0 and isinstance(fact.value, (int, float)):
                # Extract base unit (remove multiplier text)
                base_unit = fact.unit
                for pattern in multiplier_patterns.keys():
                    base_unit = base_unit.replace(pattern, "").replace(" ", "").strip()

                normalized_fact = Fact(
                    concept=fact.concept,
                    value=float(fact.value) * multiplier,
                    unit=base_unit if base_unit else fact.unit,
                    period=fact.period,
                    dimensions=fact.dimensions,
                    form=fact.form,
                    frame=fact.frame,
                    filed=fact.filed,
                )

            normalized_facts.append(normalized_fact)

        return normalized_facts

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert FactSet to dictionary format similar to SEC API format.

        Returns:
            Dictionary with concepts as keys and lists of fact entries as values
        """
        result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for fact in self.facts:
            if fact.concept not in result:
                result[fact.concept] = {"units": {}}

            if fact.unit not in result[fact.concept]["units"]:
                result[fact.concept]["units"][fact.unit] = []

            entry: Dict[str, Any] = {
                "val": fact.value,
                "end": fact.period.end.isoformat()
                if hasattr(fact.period.end, "isoformat")
                else str(fact.period.end),
            }

            if fact.period.start:
                entry["start"] = (
                    fact.period.start.isoformat()
                    if hasattr(fact.period.start, "isoformat")
                    else str(fact.period.start)
                )

            if fact.form:
                entry["form"] = fact.form

            if fact.frame:
                entry["frame"] = fact.frame

            if fact.filed:
                entry["filed"] = (
                    fact.filed.isoformat()
                    if hasattr(fact.filed, "isoformat")
                    else str(fact.filed)
                )

            if fact.dimensions:
                entry["dimensions"] = fact.dimensions

            result[fact.concept]["units"][fact.unit].append(entry)

        return result

    @classmethod
    def from_xbrl_instance(cls, xbrl: Any, cik: Optional[str] = None) -> "FactSet":
        """
        Create FactSet from a parsed XBRL instance (from XML/filing).

        Converts XBRL._facts (ModelFact) and contexts into Fact objects.
        Used for statement-centric extraction from filing-level XBRL.

        Args:
            xbrl: XBRL instance with _facts, contexts, units (from from_xml/from_filing)
            cik: Optional CIK for entity info extraction

        Returns:
            FactSet object with facts and entity information
        """
        # If XBRL was built from company facts, reuse its fact_set
        if hasattr(xbrl, "fact_set") and xbrl.fact_set is not None:
            return xbrl.fact_set

        facts: List[Fact] = []
        entity_info: Optional[EntityInfo] = None

        if not hasattr(xbrl, "_facts") or not xbrl._facts:
            return cls(facts, entity_info=entity_info)

        contexts = getattr(xbrl, "contexts", {}) or {}
        units = getattr(xbrl, "units", {}) or {}

        for fact_key, model_fact in xbrl._facts.items():
            context = contexts.get(model_fact.context_ref)
            if not context:
                continue

            period_dict = context.period or {}
            period_type = period_dict.get("type", "duration")

            if period_type == "instant":
                end_str = period_dict.get("instant")
                period_dict_for_period = {"end": end_str}
            else:
                period_dict_for_period = {
                    "start": period_dict.get("startDate"),
                    "end": period_dict.get("endDate"),
                }

            try:
                period = Period.from_xbrl_dict(period_dict_for_period)
            except (ValueError, TypeError):
                continue

            unit_str = "USD"
            if model_fact.unit_ref and model_fact.unit_ref in units:
                unit_info = units[model_fact.unit_ref]
                if isinstance(unit_info, dict):
                    unit_str = unit_info.get("measure", unit_str) or unit_str
                else:
                    unit_str = str(unit_info)
            elif model_fact.unit_ref:
                unit_str = model_fact.unit_ref
            # Normalize QName units (e.g. iso4217:USD -> USD)
            if ":" in unit_str:
                unit_str = unit_str.split(":")[-1]

            concept = model_fact.element_id
            value = (
                model_fact.numeric_value
                if model_fact.numeric_value is not None
                else model_fact.value
            )
            dimensions = getattr(context, "dimensions", {}) or {}

            fact = Fact(
                concept=concept,
                value=value,
                unit=unit_str,
                period=period,
                dimensions=dimensions,
                form=getattr(xbrl, "document_type", None) or "10-K",
                frame=None,
                filed=None,
            )
            facts.append(fact)

        dei_facts = (
            extract_dei_facts_from_xbrl(xbrl) if hasattr(xbrl, "contexts") else {}
        )
        if dei_facts:
            entity_info = build_entity_info(dei_facts, cik=cik)

        return cls(facts, entity_info=entity_info)

    @classmethod
    def from_company_facts(
        cls, company_facts: Dict[str, Any], cik: Optional[str] = None
    ) -> "FactSet":
        """
        Create FactSet from SEC company facts API response.

        Extracts US-GAAP facts, custom/extension taxonomy facts, and DEI facts.
        DEI facts are used to build entity information. Custom extension taxonomies
        (e.g. nvda for NVIDIA) are stored with a prefix so concepts like
        nvda:PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets are
        findable when resolving by local name (see get_all_facts_for_concept).

        Args:
            company_facts: Dictionary from SEC company facts API
            cik: Optional CIK to include in entity info

        Returns:
            FactSet object with facts and entity information
        """
        facts = []
        facts_by_taxonomy = company_facts.get("facts", {})

        # Iterate all taxonomies: us-gaap plus custom extensions (e.g. nvda, aapl)
        for taxonomy_key, taxonomy_data in facts_by_taxonomy.items():
            if not isinstance(taxonomy_data, dict):
                continue
            # DEI is used only for entity info extraction, not for financial facts
            if taxonomy_key.lower() == "dei":
                continue

            # Store us-gaap concepts as bare names; others as prefix_localname (e.g. nvda_ConceptName)
            concept_prefix = (
                "" if taxonomy_key == "us-gaap" else (taxonomy_key.rstrip("_-") + "_")
            )

            for concept, concept_data in taxonomy_data.items():
                if not isinstance(concept_data, dict):
                    continue
                units = concept_data.get("units", {})

                for unit, entries in units.items():
                    if not isinstance(entries, list):
                        continue
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        # Extract period information
                        period_dict = {}
                        if "end" in entry:
                            period_dict["end"] = entry["end"]
                        if "start" in entry:
                            period_dict["start"] = entry["start"]

                        if not period_dict:
                            continue

                        try:
                            period = Period.from_xbrl_dict(period_dict)
                        except (ValueError, TypeError) as e:
                            log.debug(f"Skipping fact with invalid period: {e}")
                            continue

                        try:
                            filed_date = None
                            if entry.get("filed"):
                                filed_date = datetime.fromisoformat(
                                    entry["filed"].replace("Z", "+00:00")
                                )
                        except (ValueError, AttributeError):
                            filed_date = None

                        # Concept name: bare for us-gaap, taxonomy_ConceptName for extensions
                        concept_name = (
                            concept
                            if not concept_prefix
                            else (concept_prefix + concept)
                        )
                        fact = Fact(
                            concept=concept_name,
                            value=entry.get("val"),
                            unit=unit,
                            period=period,
                            form=entry.get("form"),
                            frame=entry.get("frame"),
                            filed=filed_date,
                            dimensions=entry.get("dimensions", {}),
                        )
                        facts.append(fact)

        # Extract DEI facts and build entity info
        dei_facts = extract_dei_facts(company_facts)
        entity_info = build_entity_info(dei_facts, cik=cik) if dei_facts else None

        return cls(facts, entity_info=entity_info)

    @property
    def entity_info(self) -> Optional[EntityInfo]:
        """
        Get entity information extracted from DEI facts.

        Returns:
            EntityInfo object if available, None otherwise
        """
        return self._entity_info

    def __len__(self) -> int:
        """Return number of facts."""
        return len(self.facts)

    def __repr__(self) -> str:
        """String representation of FactSet."""
        return f"FactSet({len(self.facts)} facts)"


class FactQuery:
    """
    A query builder for XBRL facts that enables filtering by various attributes.

    This class provides a fluent interface for building queries against XBRL facts,
    allowing filtering by concept, value, period, dimensions, and other attributes.
    """

    def __init__(self, facts_view: "FactsView"):
        """
        Initialize a new fact query.

        Args:
            facts_view: The FactsView instance to query against
        """
        self._facts_view = facts_view
        self._filters = []
        self._transformations = []
        self._aggregations = []
        self._include_dimensions = False
        self._include_contexts = True
        self._include_element_info = True
        self._sort_by = None
        self._sort_ascending = True
        self._limit = None
        self._statement_type = None
        self._requested_dimension = None

    def by_concept(self, pattern: str, exact: bool = False) -> "FactQuery":
        """
        Filter facts by concept name.

        Args:
            pattern: Pattern to match against concept names
            exact: If True, require exact match; otherwise, use regex pattern matching

        Returns:
            Self for method chaining
        """
        pattern = pattern.replace(
            "_", ":"
        )  # Normalize underscores to colons for concept names
        if exact:
            self._filters.append(lambda f: f.get("concept") == pattern)
        else:
            regex = re.compile(pattern, re.IGNORECASE)
            self._filters.append(lambda f: bool(regex.search(f.get("concept", ""))))
        return self

    def by_label(self, pattern: str, exact: bool = False) -> "FactQuery":
        """
        Filter facts by element label.

        Args:
            pattern: Pattern to match against element labels
            exact: If True, require exact match; otherwise, use regex pattern matching

        Returns:
            Self for method chaining
        """
        if exact:
            self._filters.append(
                lambda f: (
                    ("label" in f and f["label"] == pattern)
                    or ("element_label" in f and f["element_label"] == pattern)
                    or ("original_label" in f and f["original_label"] == pattern)
                )
            )
        else:
            regex = re.compile(pattern, re.IGNORECASE)
            self._filters.append(
                lambda f: (
                    (
                        "label" in f
                        and f["label"] is not None
                        and bool(regex.search(str(f["label"])))
                    )
                    or (
                        "element_label" in f
                        and f["element_label"] is not None
                        and bool(regex.search(str(f["element_label"])))
                    )
                    or (
                        "original_label" in f
                        and f["original_label"] is not None
                        and bool(regex.search(str(f["original_label"])))
                    )
                )
            )
        return self

    def by_value(
        self, value_filter: Union[Callable, str, int, float, list, tuple]
    ) -> "FactQuery":
        """
        Filter facts by value.

        Args:
            value_filter: Can be:
                - A callable predicate that takes a value and returns bool
                - A specific value to match exactly
                - A tuple or list of (min, max) for range filtering

        Returns:
            Self for method chaining
        """
        if callable(value_filter):

            def numeric_value_filter(f):
                return (
                    "numeric_value" in f
                    and f["numeric_value"] is not None
                    and value_filter(f["numeric_value"])
                )

            self._filters.append(numeric_value_filter)
        elif isinstance(value_filter, (list, tuple)) and len(value_filter) == 2:
            min_val, max_val = value_filter

            def numeric_range_filter(f):
                return (
                    "numeric_value" in f
                    and f["numeric_value"] is not None
                    and min_val <= f["numeric_value"] <= max_val
                )

            self._filters.append(numeric_range_filter)
        else:

            def numeric_equality_filter(f):
                return (
                    "numeric_value" in f
                    and f["numeric_value"] is not None
                    and f["numeric_value"] == value_filter
                )

            self._filters.append(numeric_equality_filter)
        return self

    def by_period_type(self, period_type: str) -> "FactQuery":
        """
        Filter facts by period type ('instant' or 'duration').

        Args:
            period_type: Period type to filter by

        Returns:
            Self for method chaining
        """

        def period_type_filter(f):
            return "period_type" in f and f["period_type"] == period_type

        self._filters.append(period_type_filter)
        return self

    def by_period_key(self, period_key: str) -> "FactQuery":
        """
        Filter facts by a specific period key.

        Args:
            period_key: Period key to filter by (e.g., "instant_2023-12-31")

        Returns:
            Self for method chaining
        """
        self._filters.append(
            lambda f: "period_key" in f and f["period_key"] == period_key
        )
        return self

    def by_period_keys(self, period_keys: List[str]) -> "FactQuery":
        """
        Filter facts by a list of period keys.

        Args:
            period_keys: List of period keys to filter by

        Returns:
            Self for method chaining
        """
        self._filters.append(
            lambda f: "period_key" in f and f["period_key"] in period_keys
        )
        return self

    def by_dimension(
        self, dimension: Optional[str], value: Optional[str] = None
    ) -> "FactQuery":
        """
        Filter facts by dimension with flexible matching.

        Args:
            dimension: Dimension name (supports multiple formats), or None to filter for facts with no dimensions
            value: Optional dimension value to filter by (supports multiple formats)

        Returns:
            Self for method chaining
        """
        if dimension is None:
            # Filter for facts with no dimensions
            self._filters.append(
                lambda f: not any(key.startswith("dim_") for key in f.keys())
            )
            return self

        self._include_dimensions = True
        self._requested_dimension = dimension

        # Normalize the input dimension to match stored format
        normalized_dim = dimension.replace(":", "_")

        if value is not None:
            normalized_value = value.replace("_", ":")

            def dimension_filter_with_value(f):
                if (
                    f"dim_{normalized_dim}" in f
                    and f[f"dim_{normalized_dim}"] == normalized_value
                ):
                    return True
                for dim_key, dim_value in f.items():
                    if not dim_key.startswith("dim_"):
                        continue
                    if self._dimension_key_matches(dim_key, dimension):
                        if self._dimension_value_matches(dim_value, value):
                            return True
                return False

            self._filters.append(dimension_filter_with_value)
        else:

            def dimension_filter_exists(f):
                if f"dim_{normalized_dim}" in f:
                    return True
                for dim_key in f.keys():
                    if dim_key.startswith("dim_") and self._dimension_key_matches(
                        dim_key, dimension
                    ):
                        return True
                return False

            self._filters.append(dimension_filter_exists)

        return self

    def _dimension_key_matches(self, stored_key: str, query_key: str) -> bool:
        """Check if a stored dimension key matches a query key with flexible matching."""
        stored_clean = stored_key[4:] if stored_key.startswith("dim_") else stored_key
        stored_normalized = stored_clean.replace(":", "_").replace("-", "_")
        query_normalized = query_key.replace(":", "_").replace("-", "_")
        if stored_normalized == query_normalized:
            return True
        if "_" in stored_normalized:
            stored_local = stored_normalized.split("_")[-1]
            query_local = query_normalized.split("_")[-1]
            if stored_local == query_local:
                return True
        return False

    def _dimension_value_matches(self, stored_value: str, query_value: str) -> bool:
        """Check if a stored dimension value matches a query value with flexible matching."""
        if not stored_value or not query_value:
            return stored_value == query_value
        stored_normalized = stored_value.replace("_", ":").replace("-", "_")
        query_normalized = query_value.replace("_", ":").replace("-", "_")
        if stored_normalized == query_normalized:
            return True
        if ":" in stored_normalized:
            stored_local = stored_normalized.split(":")[-1]
            query_local = (
                query_normalized.split(":")[-1]
                if ":" in query_normalized
                else query_normalized
            )
            if stored_local == query_local:
                return True
        return False

    def by_statement_type(self, statement_type: str) -> "FactQuery":
        """
        Filter facts by statement type.

        Args:
            statement_type: Statement type ('BalanceSheet', 'IncomeStatement', etc.)

        Returns:
            Self for method chaining
        """
        self._filters.append(
            lambda f: "statement_type" in f and f["statement_type"] == statement_type
        )
        return self

    def by_text(self, pattern: str) -> "FactQuery":
        """
        Search across concept names, labels, and element names for a pattern.

        Args:
            pattern: Pattern to search for in various text fields

        Returns:
            Self for method chaining
        """
        regex = re.compile(pattern, re.IGNORECASE)

        def text_filter(f):
            if (
                "concept" in f
                and f["concept"] is not None
                and regex.search(str(f["concept"]))
            ):
                return True
            if (
                "label" in f
                and f["label"] is not None
                and regex.search(str(f["label"]))
            ):
                return True
            if (
                "element_label" in f
                and f["element_label"] is not None
                and regex.search(str(f["element_label"]))
            ):
                return True
            if (
                "element_name" in f
                and f["element_name"] is not None
                and regex.search(str(f["element_name"]))
            ):
                return True
            if (
                "original_label" in f
                and f["original_label"] is not None
                and regex.search(str(f["original_label"]))
            ):
                return True
            return False

        self._filters.append(text_filter)
        return self

    def with_dimensions(self) -> "FactQuery":
        """Include dimension axis and member columns in results."""
        self._include_dimensions = True
        return self

    def exclude_dimensions(self) -> "FactQuery":
        """Exclude dimension columns from results."""
        self._include_dimensions = False
        return self

    def exclude_contexts(self) -> "FactQuery":
        """Exclude context information from results."""
        self._include_contexts = False
        return self

    def exclude_element_info(self) -> "FactQuery":
        """Exclude element catalog information from results."""
        self._include_element_info = False
        return self

    def sort_by(self, column: str, ascending: bool = True) -> "FactQuery":
        """
        Set sorting for results.

        Args:
            column: Column name to sort by
            ascending: Sort order (True for ascending, False for descending)

        Returns:
            Self for method chaining
        """
        self._sort_by = column
        self._sort_ascending = ascending
        return self

    def limit(self, n: int) -> "FactQuery":
        """
        Limit the number of results.

        Args:
            n: Maximum number of results to return

        Returns:
            Self for method chaining
        """
        self._limit = n
        return self

    def execute(self) -> List[Dict[str, Any]]:
        """
        Execute the query and return matching facts.

        Returns:
            List of fact dictionaries
        """
        results = self._facts_view.get_facts()

        # Apply filters
        for filter_func in self._filters:
            results = [f for f in results if filter_func(f)]

        # Apply transformations
        for transform_fn in self._transformations:
            for fact in results:
                if "value" in fact and fact["value"] is not None:
                    fact["value"] = transform_fn(fact["value"])

        # Apply sorting if specified
        if results and self._sort_by and self._sort_by in results[0]:
            results.sort(
                key=lambda f: f.get(self._sort_by, ""), reverse=not self._sort_ascending
            )

        # Apply limit if specified
        if self._limit is not None:
            results = results[: self._limit]

        return results

    @lru_cache(maxsize=8)
    def to_dataframe(self, *columns) -> "pd.DataFrame":
        """
        Execute the query and return results as a DataFrame.

        Args:
            *columns: List of columns to include in the DataFrame

        Returns:
            pandas DataFrame with query results
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for to_dataframe() method")

        results = self.execute()

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Filter columns based on inclusion flags
        if not self._include_dimensions:
            dimension_cols = {
                "dimension",
                "member",
                "dimension_label",
                "member_label",
                "full_dimension_label",
                "dimension_axis",
                "dimension_member",
                "dimension_member_label",
            }
            df = df.loc[
                :,
                [
                    col
                    for col in df.columns
                    if (not col.startswith("dim_") and col not in dimension_cols)
                    or col == "is_dimensioned"
                ],
            ]

        if not self._include_contexts:
            context_cols = [
                "context_ref",
                "entity_identifier",
                "entity_scheme",
                "period_type",
            ]
            df = df.loc[:, [col for col in df.columns if col not in context_cols]]

        if not self._include_element_info:
            element_cols = [
                "element_id",
                "element_name",
                "element_type",
                "element_period_type",
                "element_balance",
                "element_label",
            ]
            df = df.loc[:, [col for col in df.columns if col not in element_cols]]

        # Drop empty columns
        df = df.dropna(axis=1, how="all")

        # Filter columns if specified
        if columns:
            columns = [col for col in columns if col in df.columns]
            df = df[list(columns)]

        # Order columns
        first_columns = [
            col
            for col in [
                "concept",
                "label",
                "balance",
                "preferred_sign",
                "weight",
                "value",
                "numeric_value",
                "period_key",
                "period_start",
                "period_end",
                "period_instant",
                "is_dimensioned",
                "decimals",
                "statement_type",
            ]
            if col in df.columns
        ]
        columns = first_columns + [
            col
            for col in df.columns
            if col not in first_columns and col not in ["fact_key", "original_label"]
        ]

        return df[columns] if columns else df


class FactsView:
    """
    A view over all facts in an XBRL instance, providing methods to query and analyze facts.
    """

    def __init__(self, xbrl):
        """
        Initialize the FactsView with an XBRL instance.

        Args:
            xbrl: XBRL instance containing facts, contexts, and elements
        """
        self.xbrl = xbrl
        self._facts_cache = None
        self._facts_df_cache = None

    def __len__(self):
        return len(self.get_facts())

    @property
    def entity_name(self):
        """Get entity name from XBRL instance."""
        return getattr(self.xbrl, "entity_name", "Unknown")

    @property
    def document_type(self):
        """Get document type from XBRL instance."""
        return getattr(self.xbrl, "document_type", "Unknown")

    def get_facts(self) -> List[Dict[str, Any]]:
        """
        Get all facts with enriched context and element information.

        Returns:
            List of enriched fact dictionaries
        """
        # Return cached facts if available
        if self._facts_cache is not None:
            return self._facts_cache

        # Build enriched facts from raw facts, contexts, and elements
        enriched_facts = []

        # Check if XBRL instance has fact_set (from company facts API)
        if hasattr(self.xbrl, "fact_set") and self.xbrl.fact_set:
            fact_set = self.xbrl.fact_set
            if isinstance(fact_set, FactSet):
                # Convert FactSet facts to enriched dictionaries
                for fact in fact_set.facts:
                    fact_dict = {
                        "concept": fact.concept,
                        "value": fact.value,
                        "numeric_value": fact.value
                        if isinstance(fact.value, (int, float))
                        else None,
                        "unit_ref": fact.unit,
                        "period_type": fact.period.period_type.value
                        if hasattr(fact.period, "period_type")
                        else None,
                        "period_start": str(fact.period.start)
                        if fact.period.start
                        else None,
                        "period_end": str(fact.period.end) if fact.period.end else None,
                        "period_instant": str(fact.period.end)
                        if not fact.period.start
                        else None,
                        "dimensions": fact.dimensions,
                        "form": fact.form,
                        "frame": fact.frame,
                        "filed": fact.filed.isoformat() if fact.filed else None,
                    }
                    enriched_facts.append(fact_dict)

        # Check if XBRL instance has _facts (from XML parsing)
        elif hasattr(self.xbrl, "_facts") and self.xbrl._facts:
            # Convert ModelFact objects to enriched dictionaries
            for fact_key, fact in self.xbrl._facts.items():
                # Get context information
                context = self.xbrl.contexts.get(fact.context_ref)
                period_info = {}
                dimensions = {}

                if context:
                    period = context.period
                    period_type = period.get("type", "")

                    if period_type == "instant":
                        period_info = {
                            "period_type": "instant",
                            "period_instant": period.get("instant", ""),
                            "period_start": None,
                            "period_end": None,
                        }
                    elif period_type == "duration":
                        period_info = {
                            "period_type": "duration",
                            "period_start": period.get("startDate", ""),
                            "period_end": period.get("endDate", ""),
                            "period_instant": None,
                        }

                    dimensions = context.dimensions or {}

                # Get unit information
                unit_info = (
                    self.xbrl.units.get(fact.unit_ref, {}) if fact.unit_ref else {}
                )
                unit_ref = (
                    unit_info.get("measure", fact.unit_ref) if fact.unit_ref else None
                )

                # Get element catalog information
                element_info = self.xbrl.element_catalog.get(fact.element_id)
                element_name = element_info.name if element_info else fact.element_id
                is_abstract = element_info.abstract if element_info else False

                fact_dict = {
                    "concept": fact.element_id,
                    "element_id": fact.element_id,
                    "element_name": element_name,
                    "is_abstract": is_abstract,
                    "value": fact.value,
                    "numeric_value": fact.numeric_value,
                    "unit_ref": unit_ref,
                    "decimals": fact.decimals,
                    "context_ref": fact.context_ref,
                    "fact_id": fact.fact_id,
                    "instance_id": fact.instance_id,
                    **period_info,
                    "dimensions": dimensions,
                }
                enriched_facts.append(fact_dict)

        self._facts_cache = enriched_facts
        return enriched_facts

    def query(self) -> FactQuery:
        """
        Start a new query for facts.

        Returns:
            FactQuery instance for building queries
        """
        return FactQuery(self)
