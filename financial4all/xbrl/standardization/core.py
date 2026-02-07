# financial4all/xbrl/standardization/core.py
"""
Core standardization mapping store and concept mapper.

This module provides the MappingStore and ConceptMapper classes for
managing concept mappings with company-specific support and priority scoring.
"""

import json
import logging
import os
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from .standard_concepts import StandardConcept

logger = logging.getLogger(__name__)


class MappingStore:
    """
    Storage for mappings between company-specific concepts and standard concepts.
    
    Attributes:
        source (str): Path to the JSON file storing the mappings
        mappings (Dict[str, Set[str]]): Dictionary mapping standard concepts to sets of company concepts
        company_mappings (Dict[str, Dict]): Company-specific mappings loaded from company_mappings/
        merged_mappings (Dict[str, List[Tuple]]): Merged mappings with priority scoring
    """
    
    def __init__(self, source: Optional[str] = None, validate_with_enum: bool = False, read_only: bool = False):
        """
        Initialize the mapping store.
        
        Args:
            source: Path to the JSON file storing the mappings. If None, uses default location.
            validate_with_enum: Whether to validate JSON keys against StandardConcept enum
            read_only: If True, never save changes back to the file (used in testing)
        """
        self.read_only = read_only
        
        if source is None:
            # Default to a file in the same directory as this module
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.source = os.path.join(module_dir, "concept_mappings.json")
        else:
            self.source = source
        
        self.mappings = self._load_mappings()
        
        # Load company-specific mappings (always enabled)
        self.company_mappings = self._load_all_company_mappings()
        self.merged_mappings = self._create_merged_mappings()
        self.hierarchy_rules = self._load_hierarchy_rules()
        
        # Validate the loaded mappings against StandardConcept enum
        if validate_with_enum:
            self.validate_against_enum()
    
    def validate_against_enum(self) -> Tuple[bool, List[str]]:
        """
        Validate that all keys in the mappings exist in StandardConcept enum.
        
        Returns:
            Tuple of (is_valid, list_of_missing_keys)
        """
        standard_values = StandardConcept.get_all_values()
        json_keys = set(self.mappings.keys())
        
        # Find keys in JSON that aren't in enum
        missing_in_enum = json_keys - standard_values
        
        # Find enum values not in JSON (just for information)
        missing_in_json = standard_values - json_keys
        
        if missing_in_enum:
            logger.warning("Found %d keys in concept_mappings.json that don't exist in StandardConcept enum: %s", 
                         len(missing_in_enum), sorted(missing_in_enum))
        
        if missing_in_json:
            logger.info("Found %d StandardConcept values without mappings in concept_mappings.json: %s", 
                       len(missing_in_json), sorted(missing_in_json))
        
        return len(missing_in_enum) == 0, list(missing_in_enum)
    
    def _load_all_company_mappings(self) -> Dict[str, Dict]:
        """Load all company-specific mapping files from company_mappings/ directory."""
        mappings = {}
        company_dir = os.path.join(os.path.dirname(self.source or __file__), "company_mappings")
        
        if os.path.exists(company_dir):
            for file in os.listdir(company_dir):
                if file.endswith("_mappings.json"):
                    entity_id = file.replace("_mappings.json", "")
                    try:
                        with open(os.path.join(company_dir, file), 'r') as f:
                            company_data = json.load(f)
                            mappings[entity_id] = company_data
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        logger.warning("Failed to load %s: %s", file, e)
        
        return mappings
    
    def _create_merged_mappings(self) -> Dict[str, List[Tuple[str, str, int]]]:
        """Create merged mappings with priority scoring.
        
        Priority levels:
        1. Core mappings (lowest)
        2. Company mappings (higher)
        3. Company-specific matches (highest when company detected)
        
        Returns:
            Dict mapping standard concepts to list of (company_concept, source, priority) tuples
        """
        merged = {}
        
        # Add core mappings (priority 1 - lowest)
        for std_concept, company_concepts in self.mappings.items():
            merged[std_concept] = []
            for concept in company_concepts:
                merged[std_concept].append((concept, "core", 1))
        
        # Add company mappings (priority 2 - higher)
        for entity_id, company_data in self.company_mappings.items():
            concept_mappings = company_data.get("concept_mappings", {})
            priority_level = 2
            
            for std_concept, company_concepts in concept_mappings.items():
                if std_concept not in merged:
                    merged[std_concept] = []
                for concept in company_concepts:
                    merged[std_concept].append((concept, entity_id, priority_level))
        
        return merged
    
    def _load_hierarchy_rules(self) -> Dict[str, Dict]:
        """Load hierarchy rules from company mappings."""
        all_rules = {}
        
        # Add company hierarchy rules
        for _entity_id, company_data in self.company_mappings.items():
            hierarchy_rules = company_data.get("hierarchy_rules", {})
            all_rules.update(hierarchy_rules)
        
        return all_rules
    
    def _detect_entity_from_concept(self, concept: str) -> Optional[str]:
        """Detect entity identifier from concept name prefix."""
        if '_' in concept:
            prefix = concept.split('_')[0].lower()
            # Check if this prefix corresponds to a known company
            if prefix in self.company_mappings:
                return prefix
        return None
    
    def _load_mappings(self) -> Dict[str, Set[str]]:
        """
        Load mappings from the JSON file.
        
        Returns:
            Dictionary mapping standard concepts to sets of company concepts
        """
        try:
            with open(self.source, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, IOError, PermissionError) as e:
            logger.warning("Could not load concept_mappings.json: %s. Standardization will be limited.", e)
            return {}
        
        # If we have data, process it based on its structure
        if data:
            # Check if the structure is flat or nested
            if any(isinstance(value, dict) for value in data.values()):
                # Nested structure by statement type
                flattened = {}
                for _statement_type, concepts in data.items():
                    for standard_concept, company_concepts in concepts.items():
                        flattened[standard_concept] = set(company_concepts)
                return flattened
            else:
                # Flat structure
                return {k: set(v) for k, v in data.items()}
        
        return {}
    
    def _save_mappings(self) -> None:
        """Save mappings to the JSON file, unless in read_only mode."""
        # Skip saving if in read_only mode
        if self.read_only:
            return
        
        # Ensure directory exists
        directory = os.path.dirname(self.source)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Convert sets to lists for JSON serialization
        serializable_mappings = {k: list(v) for k, v in self.mappings.items()}
        
        with open(self.source, 'w') as f:
            json.dump(serializable_mappings, f, indent=2)
    
    def add(self, company_concept: str, standard_concept: str) -> None:
        """
        Add a mapping from a company concept to a standard concept.
        
        Args:
            company_concept: The company-specific concept
            standard_concept: The standard concept
        """
        if standard_concept not in self.mappings:
            self.mappings[standard_concept] = set()
        
        self.mappings[standard_concept].add(company_concept)
        self._save_mappings()
    
    def get_standard_concept(self, company_concept: str, context: Optional[Dict] = None) -> Optional[str]:
        """
        Get the standard concept for a given company concept with priority-based resolution.
        
        Uses a multi-tier lookup strategy:
        1. Reverse index (O(1) lookup, 2,067 GAAP tags) - NEW
        2. Company-specific mappings (priority-based)
        3. Core mappings fallback
        
        Args:
            company_concept: The company-specific concept
            context: Optional context information (for future disambiguation)
        
        Returns:
            The standard concept or None if not found
        """
        # Tier 1: Try the reverse index first (O(1) lookup, covers 95% of tags)
        try:
            from .reverse_index import get_reverse_index
            reverse_index = get_reverse_index()
            display_name = reverse_index.get_display_name(company_concept, context)
            if display_name:
                logger.debug("Reverse index match: %s -> %s", company_concept, display_name)
                return display_name
        except (ImportError, FileNotFoundError, json.JSONDecodeError) as e:
            # Only catch expected errors - let other exceptions propagate
            logger.debug("Reverse index lookup failed for %s: %s", company_concept, e)
        
        # Tier 2: Use merged mappings with priority-based resolution
        if self.merged_mappings:
            # Detect company from concept prefix (e.g., 'tsla:Revenue' -> 'tsla')
            detected_entity = self._detect_entity_from_concept(company_concept)
            
            # Search through merged mappings with priority
            candidates = []
            
            for std_concept, mapping_list in self.merged_mappings.items():
                for concept, source, priority in mapping_list:
                    if concept == company_concept:
                        # Boost priority if it matches detected entity
                        effective_priority = priority
                        if detected_entity and source == detected_entity:
                            effective_priority = 4  # Highest priority for exact company match
                        
                        candidates.append((std_concept, effective_priority, source))
            
            # Return highest priority match
            if candidates:
                best_match = max(candidates, key=lambda x: x[1])
                logger.debug("Merged mapping applied: %s -> %s (source: %s, priority: %s)", 
                           company_concept, best_match[0], best_match[2], best_match[1])
                return best_match[0]
        
        # Tier 3: Fallback to core mappings
        for standard_concept, company_concepts in self.mappings.items():
            if company_concept in company_concepts:
                logger.debug("Core mapping fallback: %s -> %s", company_concept, standard_concept)
                return standard_concept
        
        return None
    
    def get_company_concepts(self, standard_concept: str) -> Set[str]:
        """
        Get all company concepts mapped to a standard concept.
        
        Args:
            standard_concept: The standard concept
        
        Returns:
            Set of company concepts mapped to the standard concept
        """
        return self.mappings.get(standard_concept, set())


class ConceptMapper:
    """
    Maps company-specific concepts to standard concepts using various techniques.
    
    Attributes:
        mapping_store (MappingStore): Storage for concept mappings
        pending_mappings (Dict): Low-confidence mappings pending review
        _cache (Dict): In-memory cache of mapped concepts
    """
    
    def __init__(self, mapping_store: MappingStore):
        """
        Initialize the concept mapper.
        
        Args:
            mapping_store: Storage for concept mappings
        """
        self.mapping_store = mapping_store
        self.pending_mappings = {}
        # Cache for faster lookups of previously mapped concepts
        self._cache = {}
        # Precompute lowercased standard concept values for faster comparison
        self._std_concept_values = [(concept, concept.value.lower()) for concept in StandardConcept]
        
        # Statement-specific keyword sets for faster contextual matching
        self._bs_keywords = {'assets', 'liabilities', 'equity', 'cash', 'debt', 'inventory', 'receivable', 'payable'}
        self._is_keywords = {'revenue', 'sales', 'income', 'expense', 'profit', 'loss', 'tax', 'earnings'}
        self._cf_keywords = {'cash', 'operating', 'investing', 'financing', 'activities'}
    
    def map_concept(self, company_concept: str, label: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Map a company concept to a standard concept.
        
        Args:
            company_concept: The company-specific concept
            label: The label for the concept
            context: Additional context information (statement type, calculation relationships, etc.)
        
        Returns:
            The standard concept or None if no mapping found
        """
        # Use cache for faster lookups - include section for ambiguous tag disambiguation
        cache_key = (company_concept, context.get('statement_type', ''), context.get('section', ''))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Check if we already have a mapping in the store
        # Pass context for context-aware disambiguation of ambiguous tags
        standard_concept = self.mapping_store.get_standard_concept(company_concept, context)
        if standard_concept:
            self._cache[cache_key] = standard_concept
            return standard_concept
        
        # Cache negative results too to avoid repeated inference
        self._cache[cache_key] = None
        return None
    
    def _infer_mapping(self, company_concept: str, label: str, context: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """
        Infer a mapping between a company concept and a standard concept.
        
        Args:
            company_concept: The company-specific concept
            label: The label for the concept
            context: Additional context information
        
        Returns:
            Tuple of (standard_concept, confidence)
        """
        # Fast path for common patterns
        label_lower = label.lower()
        
        # Quick matching for common concepts without full sequence matching
        if "total assets" in label_lower:
            return StandardConcept.TOTAL_ASSETS.value, 0.95
        elif "revenue" in label_lower and len(label_lower) < 30:  # Only match short labels to avoid false positives
            return StandardConcept.REVENUE.value, 0.9
        elif "net income" in label_lower and "parent" not in label_lower:
            return StandardConcept.NET_INCOME.value, 0.9
        
        # Faster direct match checking with precomputed lowercase values
        for std_concept, std_value_lower in self._std_concept_values:
            if std_value_lower == label_lower:
                return std_concept.value, 1.0  # Perfect match
        
        # Fall back to sequence matching for similarity
        best_match = None
        best_score = 0
        
        # Only compute similarity if some relevant keywords are present to reduce workload
        statement_type = context.get("statement_type", "")
        
        # Statement type based filtering to reduce unnecessary comparisons
        limited_concepts = []
        if statement_type == "BalanceSheet":
            if any(kw in label_lower for kw in self._bs_keywords):
                # Filter to balance sheet concepts only
                limited_concepts = [c for c, v in self._std_concept_values
                                  if any(kw in v for kw in self._bs_keywords)]
        elif statement_type == "IncomeStatement":
            if any(kw in label_lower for kw in self._is_keywords):
                # Filter to income statement concepts only
                limited_concepts = [c for c, v in self._std_concept_values
                                  if any(kw in v for kw in self._is_keywords)]
        elif statement_type == "CashFlowStatement":
            if any(kw in label_lower for kw in self._cf_keywords):
                # Filter to cash flow concepts only
                limited_concepts = [c for c, v in self._std_concept_values
                                  if any(kw in v for kw in self._cf_keywords)]
        
        # Use limited concepts if available, otherwise use all
        concepts_to_check = limited_concepts if limited_concepts else [c for c, _ in self._std_concept_values]
        
        # Calculate similarities for candidate concepts
        for std_concept in concepts_to_check:
            # Calculate similarity between labels
            similarity = SequenceMatcher(None, label_lower, std_concept.value.lower()).ratio()
            
            # Check if this is the best match so far
            if similarity > best_score:
                best_score = similarity
                best_match = std_concept.value
        
        # Apply specific contextual rules based on statement type
        if statement_type == "BalanceSheet":
            if "assets" in label_lower and "total" in label_lower:
                if best_match == StandardConcept.TOTAL_ASSETS.value:
                    best_score = min(1.0, best_score + 0.2)
            elif "liabilities" in label_lower and "total" in label_lower:
                if best_match == StandardConcept.TOTAL_LIABILITIES.value:
                    best_score = min(1.0, best_score + 0.2)
            elif "equity" in label_lower and ("total" in label_lower or "stockholders" in label_lower):
                if best_match == StandardConcept.TOTAL_EQUITY.value:
                    best_score = min(1.0, best_score + 0.2)
        
        elif statement_type == "IncomeStatement":
            if any(term in label_lower for term in ["revenue", "sales"]):
                if best_match == StandardConcept.REVENUE.value:
                    best_score = min(1.0, best_score + 0.2)
            elif "net income" in label_lower:
                if best_match == StandardConcept.NET_INCOME.value:
                    best_score = min(1.0, best_score + 0.2)
        
        # Promote to 0.5 confidence if score close enough to help match
        # more items that are almost at threshold
        if 0.45 <= best_score < 0.5:
            best_score = 0.5
        
        # If confidence is too low, return None
        if best_score < 0.5:
            return None, 0.0
        
        return best_match, best_score
