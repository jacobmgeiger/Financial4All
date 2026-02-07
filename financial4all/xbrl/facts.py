# financial4all/xbrl/facts.py
"""
Fact extraction and management for XBRL data.

This module provides functionality for extracting and managing XBRL facts,
including dimensional facts, units, and period filtering.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime, date

from financial4all.xbrl.periods import Period, PeriodType


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
    """
    
    def __init__(self, facts: Optional[List[Fact]] = None):
        """
        Initialize FactSet.
        
        Args:
            facts: Optional list of facts to initialize with
        """
        self.facts: List[Fact] = facts or []
    
    def add(self, fact: Fact) -> None:
        """Add a fact to the set."""
        self.facts.append(fact)
    
    def filter_by_form(self, form: str) -> "FactSet":
        """Filter facts by form type."""
        filtered = [f for f in self.facts if f.form == form]
        return FactSet(filtered)
    
    def filter_by_concept(self, concept: str) -> "FactSet":
        """Filter facts by concept name."""
        filtered = [f for f in self.facts if f.concept == concept]
        return FactSet(filtered)
    
    def filter_annual_10k(self) -> "FactSet":
        """Filter to only annual 10-K facts."""
        filtered = [f for f in self.facts if f.is_annual_10k()]
        return FactSet(filtered)
    
    def get_unique_concepts(self) -> Set[str]:
        """Get set of unique concept names."""
        return {f.concept for f in self.facts}
    
    def get_by_concept(self, concept: str) -> List[Fact]:
        """
        Get all facts for a specific concept.
        
        Supports fuzzy matching for namespace variations:
        - Tries exact match first
        - Then tries with 'us-gaap_' prefix if not found
        - Then tries without 'us-gaap_' prefix if concept starts with it
        
        Args:
            concept: XBRL concept name (e.g., "us-gaap_Revenues" or "Revenues")
            
        Returns:
            List of matching facts
        """
        # Try exact match first
        exact_matches = [f for f in self.facts if f.concept == concept]
        if exact_matches:
            return exact_matches
        
        # Try with namespace prefix if not already present
        if not concept.startswith("us-gaap_"):
            prefixed = f"us-gaap_{concept}"
            exact_matches = [f for f in self.facts if f.concept == prefixed]
            if exact_matches:
                return exact_matches
        
        # Try without namespace prefix if it's present
        if concept.startswith("us-gaap_"):
            unprefixed = concept.replace("us-gaap_", "", 1)
            exact_matches = [f for f in self.facts if f.concept == unprefixed]
            if exact_matches:
                return exact_matches
        
        return []
    
    def get_concepts_by_pattern(self, pattern: str) -> Set[str]:
        """
        Get all concepts that match a pattern (case-insensitive substring match).
        
        Args:
            pattern: Pattern to search for (e.g., "Interest" to find all interest-related concepts)
            
        Returns:
            Set of matching concept names
        """
        pattern_lower = pattern.lower()
        return {
            f.concept for f in self.facts
            if pattern_lower in f.concept.lower()
        }
    
    def find_synonym_concepts(self, concept: str) -> Set[str]:
        """
        Find synonym concepts using pattern matching and known synonyms.
        
        Helps discover concepts that might be named differently but represent
        the same financial metric. Uses concept patterns and known synonyms.
        
        Args:
            concept: Base concept name to find synonyms for
            
        Returns:
            Set of synonym concept names found in this FactSet
        """
        synonyms = set()
        
        # Remove namespace prefix for matching
        base_concept = concept.replace("us-gaap_", "").lower()
        
        # Common synonym patterns
        synonym_patterns = {
            "revenue": ["revenue", "revenues", "sales", "income"],
            "revenues": ["revenue", "revenues", "sales", "income"],
            "salesrevenuenet": ["revenue", "revenues", "salesrevenuenet", "revenuefromcontract"],
            "revenuefromcontractwithcustomer": ["revenue", "revenues", "salesrevenuenet", "revenuefromcontract"],
            "interestincome": ["interestincome", "interest", "investmentincome"],
            "interestexpense": ["interestexpense", "interest"],
            "operatingincome": ["operatingincome", "incomefromoperations", "operatingprofit"],
            "netincome": ["netincome", "netincomeloss", "profitloss", "earnings"],
        }
        
        # Check if base concept matches any known pattern
        for pattern_key, pattern_list in synonym_patterns.items():
            if pattern_key in base_concept or base_concept in pattern_key:
                # Search for concepts matching any synonym pattern
                for synonym_pattern in pattern_list:
                    matching_concepts = self.get_concepts_by_pattern(synonym_pattern)
                    synonyms.update(matching_concepts)
        
        # Also do direct substring matching for similar concepts
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
    
    def has_reported_data(self, concept: str) -> bool:
        """
        Check if a concept has any reported facts.
        
        Args:
            concept: XBRL concept name
            
        Returns:
            True if at least one fact exists for this concept, False otherwise
        """
        return len(self.get_by_concept(concept)) > 0
    
    def get_all_facts_for_concept(self, concept: str, include_variants: bool = True) -> List[Fact]:
        """
        Get all facts for a concept with comprehensive namespace variant matching.
        
        Searches for concept with all namespace variations and returns ALL facts
        regardless of form/frame initially. This is more comprehensive than
        get_by_concept() which stops after first match.
        
        Args:
            concept: XBRL concept name (e.g., "Revenues" or "us-gaap_Revenues")
            include_variants: If True, tries multiple namespace variations
            
        Returns:
            List of all matching facts
        """
        all_facts = []
        seen_facts = set()  # Track by (concept, period.end, unit) to avoid duplicates
        
        # Generate all possible variations
        variations = [concept]
        if include_variants:
            if not concept.startswith("us-gaap_"):
                variations.append(f"us-gaap_{concept}")
            if concept.startswith("us-gaap_"):
                variations.append(concept.replace("us-gaap_", "", 1))
        
        # Collect facts from all variations
        for variant in variations:
            facts = [f for f in self.facts if f.concept == variant]
            for fact in facts:
                # Use a unique key to avoid duplicates
                fact_key = (fact.concept, str(fact.period.end), fact.unit, fact.value)
                if fact_key not in seen_facts:
                    seen_facts.add(fact_key)
                    all_facts.append(fact)
        
        return all_facts
    
    def filter_by_period_range(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> "FactSet":
        """
        Filter facts by period range.
        
        Args:
            start_date: Optional start date (inclusive)
            end_date: Optional end date (inclusive)
            
        Returns:
            Filtered FactSet
        """
        filtered = []
        for fact in self.facts:
            period_end = fact.period.end
            if isinstance(period_end, str):
                try:
                    period_end = datetime.fromisoformat(period_end.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    continue
            
            if start_date and period_end < start_date:
                continue
            if end_date and period_end > end_date:
                continue
            
            filtered.append(fact)
        
        return FactSet(filtered)
    
    def get_facts_by_period(self, period_end: Union[datetime, date, str]) -> List[Fact]:
        """
        Get all facts for a specific period end date.
        
        Args:
            period_end: Period end date (datetime, date, or ISO string)
            
        Returns:
            List of facts matching the period
        """
        # Normalize period_end to date for comparison
        if isinstance(period_end, str):
            try:
                period_end = datetime.fromisoformat(period_end.replace('Z', '+00:00')).date()
            except (ValueError, AttributeError):
                return []
        elif isinstance(period_end, datetime):
            period_end = period_end.date()
        
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
    def from_company_facts(cls, company_facts: Dict[str, Any]) -> "FactSet":
        """
        Create FactSet from SEC company facts API response.
        
        Args:
            company_facts: Dictionary from SEC company facts API
            
        Returns:
            FactSet object
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
                    
                    period = Period.from_xbrl_dict(period_dict)
                    
                    fact = Fact(
                        concept=concept,
                        value=entry.get("val"),
                        unit=unit,
                        period=period,
                        form=entry.get("form"),
                        frame=entry.get("frame"),
                        filed=datetime.fromisoformat(entry["filed"]) if entry.get("filed") else None,
                        dimensions=entry.get("dimensions", {}),
                    )
                    facts.append(fact)
        
        return cls(facts)
    
    def __len__(self) -> int:
        """Return number of facts."""
        return len(self.facts)
    
    def __repr__(self) -> str:
        """String representation of FactSet."""
        return f"FactSet({len(self.facts)} facts)"
