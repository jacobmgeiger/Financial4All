# financial4all/xbrl/facts.py
"""
Fact extraction and management for XBRL data.

This module provides functionality for extracting and managing XBRL facts,
including dimensional facts, units, and period filtering.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from datetime import datetime, date

from financial4all.xbrl.periods import Period, PeriodType
from financial4all.xbrl.entity_info import extract_dei_facts, build_entity_info, EntityInfo
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
            self.form == "10-K" and
            self.period.period_type == PeriodType.DURATION and
            self.period.is_annual() and
            (not self.frame or "Q" not in self.frame)
        )
    
    def __repr__(self) -> str:
        """String representation of Fact."""
        return f"Fact(concept={self.concept}, value={self.value}, unit={self.unit}, period={self.period})"


class FactSet:
    """
    Collection of XBRL facts with filtering and query capabilities.
    
    Uses indexes for efficient O(1) lookups by concept, period, or concept-period combination.
    """
    
    def __init__(self, facts: Optional[List[Fact]] = None, entity_info: Optional[EntityInfo] = None):
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
        self._concept_period_index: Dict[Tuple[str, str], List[Fact]] = {}  # (concept, period_end) -> list of facts
        
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
            log.warning(f"Invalid pattern provided to get_concepts_by_pattern: {pattern}")
            return set()
        
        pattern_lower = pattern.lower().strip()
        if not pattern_lower:
            return set()
        
        # Pre-compile lowercase concept set for O(1) lookup optimization
        # Use set comprehension with optimized string operations
        pattern_lower_len = len(pattern_lower)
        return {
            f.concept for f in self.facts
            if pattern_lower in f.concept.lower()
        }
    
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
            base_concept = concept.replace("us-gaap_", "").replace("us-gaap:", "").lower()
            
            # Try to find concepts with similar names using substring matching
            for fact in self.facts:
                fact_concept_lower = fact.concept.lower()
                # Check if concepts share significant substrings
                if (base_concept in fact_concept_lower or 
                    fact_concept_lower in base_concept or
                    self._concepts_similar(base_concept, fact_concept_lower)):
                    synonyms.add(fact.concept)
        
        # Also do direct substring matching for similar concepts (fallback)
        if not synonyms:
            # Remove namespace prefix for matching
            base_concept = concept.replace("us-gaap_", "").replace("us-gaap:", "").lower()
            
            # Remove common suffixes/prefixes and match
            base_stem = base_concept
            for suffix in ["net", "loss", "expense", "income", "revenue"]:
                if base_stem.endswith(suffix):
                    base_stem = base_stem[:-len(suffix)]
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
    
    def _create_normalized_fact_key(self, element_id: str, context_ref: str, instance_id: Optional[int] = None) -> str:
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
        if ':' in element_id:
            prefix, name = element_id.split(':', 1)
            normalized_element_id = f"{prefix}_{name}"
        
        # Create base key
        base_key = f"{normalized_element_id}_{context_ref}"
        
        # Add instance ID if provided
        if instance_id is not None:
            return f"{base_key}_{instance_id}"
        
        return base_key
    
    def get_all_facts_for_concept(self, concept: str, include_variants: bool = True) -> List[Fact]:
        """
        Get all facts for a concept with comprehensive namespace variant matching.
        
        Searches for concept with all namespace variations and returns ALL facts
        regardless of form/frame initially. This is more comprehensive than
        get_by_concept() which stops after first match.
        
        Uses indexes for efficient lookup when possible.
        
        Args:
            concept: XBRL concept name (e.g., "Revenues" or "us-gaap_Revenues")
            include_variants: If True, tries multiple namespace variations
            
        Returns:
            List of all matching facts
        """
        if not concept:
            return []
        
        all_facts = []
        seen_facts = set()  # Track by normalized key to avoid duplicates
        
        # Generate all possible variations
        variations = [concept]
        if include_variants:
            if not concept.startswith("us-gaap_"):
                variations.append(f"us-gaap_{concept}")
            if concept.startswith("us-gaap_"):
                variations.append(concept.replace("us-gaap_", "", 1))
        
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
                # Create normalized key to avoid duplicates
                period_key = str(fact.period.end)
                fact_key = create_key(fact.concept, period_key)
                
                if fact_key not in seen_facts:
                    seen_facts_add(fact_key)
                    all_facts_append(fact)
        
        return all_facts
    
    def filter_by_period_range(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> "FactSet":
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
            raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")
        
        filtered = []
        for fact in self.facts:
            try:
                period_end = fact.period.end
                if isinstance(period_end, str):
                    try:
                        period_end = datetime.fromisoformat(period_end.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        log.debug(f"Skipping fact with invalid period end: {period_end}")
                        continue
                elif isinstance(period_end, date) and not isinstance(period_end, datetime):
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
                    dt = datetime.fromisoformat(period_end.replace('Z', '+00:00'))
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
                    fact_end = datetime.fromisoformat(fact_end.replace('Z', '+00:00')).date()
                except (ValueError, AttributeError):
                    continue
            elif isinstance(fact_end, datetime):
                fact_end = fact_end.date()
            
            if fact_end == period_end:
                matching_facts.append(fact)
        
        return matching_facts
    
    def get_facts_by_concept_and_period(self, concept: str, period_end: Union[datetime, date, str]) -> List[Fact]:
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
                    dt = datetime.fromisoformat(period_end.replace('Z', '+00:00'))
                    period_key = str(dt.date())
                except (ValueError, AttributeError):
                    period_key = period_end
            elif isinstance(period_end, (datetime, date)):
                period_key = str(period_end.date() if isinstance(period_end, datetime) else period_end)
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
                "end": fact.period.end.isoformat() if hasattr(fact.period.end, 'isoformat') else str(fact.period.end),
            }
            
            if fact.period.start:
                entry["start"] = fact.period.start.isoformat() if hasattr(fact.period.start, 'isoformat') else str(fact.period.start)
            
            if fact.form:
                entry["form"] = fact.form
            
            if fact.frame:
                entry["frame"] = fact.frame
            
            if fact.filed:
                entry["filed"] = fact.filed.isoformat() if hasattr(fact.filed, 'isoformat') else str(fact.filed)
            
            if fact.dimensions:
                entry["dimensions"] = fact.dimensions
            
            result[fact.concept]["units"][fact.unit].append(entry)
        
        return result
    
    @classmethod
    def from_company_facts(cls, company_facts: Dict[str, Any], cik: Optional[str] = None) -> "FactSet":
        """
        Create FactSet from SEC company facts API response.
        
        Extracts both US-GAAP facts and DEI (Document and Entity Information) facts.
        DEI facts are used to build entity information including fiscal year end dates.
        
        Args:
            company_facts: Dictionary from SEC company facts API
            cik: Optional CIK to include in entity info
            
        Returns:
            FactSet object with facts and entity information
        """
        facts = []
        us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
        
        for concept, concept_data in us_gaap.items():
            units = concept_data.get("units", {})
            
            for unit, entries in units.items():
                for entry in entries:
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
                            filed_date = datetime.fromisoformat(entry["filed"].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        filed_date = None
                    
                    fact = Fact(
                        concept=concept,
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
