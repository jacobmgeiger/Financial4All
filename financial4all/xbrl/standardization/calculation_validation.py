# financial4all/xbrl/standardization/calculation_validation.py
"""
Calculation-based validation for XBRL concept mappings.

This module provides functionality to validate concept mappings by checking
if they satisfy calculation relationships defined in XBRL calculation linkbases.
This helps catch mapping errors early, such as misclassified concepts that
don't satisfy expected financial relationships.

Example:
    >>> validator = CalculationValidator(calculation_engine)
    >>> is_valid, error = validator.validate_mapping(
    ...     mapped_concept="Interest Income",
    ...     calculation_parent="Operating Income",
    ...     available_facts={"Operating Income": 100000, "Interest Income": 95000}
    ... )
    >>> if not is_valid:
    ...     print(f"Mapping error: {error}")
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from financial4all.xbrl.calculations import CalculationEngine

from financial4all.core import log

logger = logging.getLogger(__name__)


class CalculationValidator:
    """
    Validates concept mappings using calculation relationships.
    
    This class checks if mapped concepts satisfy expected calculation
    relationships (e.g., Revenue - COGS = Gross Profit) to catch mapping errors.
    
    Attributes:
        calculation_engine: CalculationEngine instance for accessing formulas
        validation_rules: Dictionary of standard validation rules
        tolerance: Tolerance for floating-point comparisons (default: 0.01 = 1%)
    """
    
    def __init__(self, calculation_engine: Optional['CalculationEngine'] = None, tolerance: float = 0.01):
        """
        Initialize the calculation validator.
        
        Args:
            calculation_engine: Optional CalculationEngine instance. If None, creates new one lazily.
            tolerance: Tolerance for floating-point comparisons (default: 0.01 = 1%)
        """
        self._calculation_engine = calculation_engine
        self.tolerance = tolerance
        
        # Standard validation rules for common financial relationships
        # Format: {parent_concept: [(child_concept, weight, expected_relationship)]}
        self.validation_rules = self._build_standard_rules()
        
        logger.debug(
            "CalculationValidator initialized with %d validation rules",
            len(self.validation_rules)
        )
    
    @property
    def calculation_engine(self) -> 'CalculationEngine':
        """Lazy-load CalculationEngine to avoid circular imports."""
        if self._calculation_engine is None:
            from financial4all.xbrl.calculations import CalculationEngine
            self._calculation_engine = CalculationEngine()
        return self._calculation_engine
    
    def _build_standard_rules(self) -> Dict[str, List[Tuple[str, float, str]]]:
        """
        Build standard validation rules for common financial relationships.
        
        Returns:
            Dictionary mapping parent concepts to validation rules
        """
        rules = {
            # Income Statement relationships
            "Gross Profit": [
                ("Revenue", 1.0, "Revenue - Cost of Revenue = Gross Profit"),
                ("Cost of Revenue", -1.0, "Revenue - Cost of Revenue = Gross Profit"),
            ],
            "Operating Income": [
                ("Gross Profit", 1.0, "Gross Profit - Operating Expenses = Operating Income"),
                ("Operating Expenses", -1.0, "Gross Profit - Operating Expenses = Operating Income"),
            ],
            "Income Before Taxes": [
                ("Operating Income", 1.0, "Operating Income + Other income (expense), net = Income Before Taxes"),
                ("Other income (expense), net", 1.0, "Operating Income + Other income (expense), net = Income Before Taxes"),
            ],
            "Net Income": [
                ("Income Before Taxes", 1.0, "Income Before Taxes - Taxes = Net Income"),
                ("Taxes", -1.0, "Income Before Taxes - Taxes = Net Income"),
            ],
            "Other income (expense), net": [
                ("Interest Income", 1.0, "Interest Income + Interest Expense + Other, net = Other income (expense), net"),
                ("Interest Expense", 1.0, "Interest Income + Interest Expense + Other, net = Other income (expense), net"),
                ("Other, net", 1.0, "Interest Income + Interest Expense + Other, net = Other income (expense), net"),
            ],
            # Operating Expenses relationships
            "Operating Expenses": [
                ("Research and Development Expense", 1.0, "R&D + SG&A = Operating Expenses"),
                ("Selling, General and Administrative Expense", 1.0, "R&D + SG&A = Operating Expenses"),
            ],
        }
        return rules
    
    def validate_mapping(
        self,
        mapped_concept: str,
        calculation_parent: Optional[str] = None,
        calculation_children: Optional[List[Dict[str, Any]]] = None,
        available_facts: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a concept mapping using calculation relationships.
        
        Args:
            mapped_concept: The standard concept that was mapped
            calculation_parent: Optional parent concept from calculation linkbase
            calculation_children: Optional list of child concepts with weights
            available_facts: Optional dictionary of available fact values (concept -> value)
            context: Optional context information (statement type, period, etc.)
        
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if mapping appears valid, False if suspicious
            - error_message: Description of validation issue if not valid
        """
        if available_facts is None:
            available_facts = {}
        
        # Check standard validation rules first
        validation_result = self._check_standard_rules(mapped_concept, available_facts)
        if validation_result is not None:
            return validation_result
        
        # Check calculation linkbase relationships if provided
        if calculation_parent and calculation_children:
            validation_result = self._check_calculation_linkbase(
                mapped_concept, calculation_parent, calculation_children, available_facts
            )
            if validation_result is not None:
                return validation_result
        
        # If no validation issues found, mapping appears valid
        return True, None
    
    def _check_standard_rules(
        self,
        mapped_concept: str,
        available_facts: Dict[str, float]
    ) -> Optional[Tuple[bool, Optional[str]]]:
        """
        Check standard validation rules for a mapped concept.
        
        Args:
            mapped_concept: The standard concept that was mapped
            available_facts: Dictionary of available fact values
        
        Returns:
            Tuple of (is_valid, error_message) if validation issue found, None otherwise
        """
        # Check if this concept is a parent in any validation rule
        if mapped_concept in self.validation_rules:
            rules = self.validation_rules[mapped_concept]
            return self._validate_parent_concept(mapped_concept, rules, available_facts)
        
        # Check if this concept is a child in any validation rule
        for parent_concept, rules in self.validation_rules.items():
            for child_concept, weight, relationship in rules:
                if child_concept == mapped_concept:
                    return self._validate_child_concept(
                        mapped_concept, parent_concept, weight, relationship, available_facts
                    )
        
        return None
    
    def _validate_parent_concept(
        self,
        parent_concept: str,
        rules: List[Tuple[str, float, str]],
        available_facts: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a parent concept against its expected children.
        
        Args:
            parent_concept: The parent concept being validated
            rules: List of (child_concept, weight, relationship_description) tuples
            available_facts: Dictionary of available fact values
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if parent_concept not in available_facts:
            return True, None  # Can't validate without parent value
        
        parent_value = available_facts[parent_concept]
        
        # Calculate expected parent value from children
        calculated_parent = 0.0
        missing_children = []
        
        for child_concept, weight, relationship in rules:
            if child_concept in available_facts:
                calculated_parent += available_facts[child_concept] * weight
            else:
                missing_children.append(child_concept)
        
        # If we have all children, validate the relationship
        if not missing_children:
            diff = abs(parent_value - calculated_parent)
            relative_diff = diff / abs(parent_value) if parent_value != 0 else float('inf')
            
            if relative_diff > self.tolerance:
                error_msg = (
                    f"Calculation validation failed for {parent_concept}: "
                    f"expected {calculated_parent:,.0f} but got {parent_value:,.0f} "
                    f"(diff: {diff:,.0f}, {relative_diff:.1%}). "
                    f"Relationship: {rules[0][2] if rules else 'unknown'}"
                )
                return False, error_msg
        
        return True, None
    
    def _validate_child_concept(
        self,
        child_concept: str,
        parent_concept: str,
        weight: float,
        relationship: str,
        available_facts: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a child concept against its parent.
        
        This checks if a child concept's value is reasonable relative to its parent.
        For example, Interest Income should be much smaller than Operating Income.
        
        Args:
            child_concept: The child concept being validated
            parent_concept: The parent concept
            weight: Weight of child in calculation (1.0 for addition, -1.0 for subtraction)
            relationship: Description of the relationship
            available_facts: Dictionary of available fact values
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if child_concept not in available_facts or parent_concept not in available_facts:
            return True, None  # Can't validate without both values
        
        child_value = available_facts[child_concept]
        parent_value = available_facts[parent_concept]
        
        # Special validation for Interest Income vs Operating Income
        # Focus on detecting values suspiciously CLOSE to Operating Income (likely misclassified)
        # rather than just large values (which might be legitimate for cash-rich companies)
        if child_concept == "Interest Income" and parent_concept == "Operating Income":
            if abs(parent_value) > 0:
                ratio = abs(child_value) / abs(parent_value)
                diff_ratio = abs(child_value - parent_value) / abs(parent_value)
                abs_diff = abs(child_value - parent_value)
                
                # PRIMARY CHECK: If Interest Income is suspiciously CLOSE to Operating Income (< 1%)
                # This is the strongest indicator of misclassification
                # Values within 1% are almost certainly misclassified Operating Income
                if diff_ratio < 0.01:  # Within 1% - very suspicious, likely misclassified
                    error_msg = (
                        f"Interest Income ({child_value:,.0f}) is suspiciously close to "
                        f"Operating Income ({parent_value:,.0f}) (diff: {diff_ratio:.3%}). "
                        f"This is likely misclassified Operating Income."
                    )
                    return False, error_msg
                
                # SECONDARY CHECK: If Interest Income is moderately close (1-5%) AND large (> 50% ratio)
                # This catches cases where Interest Income is both close and large, which is very suspicious
                if 0.01 <= diff_ratio < 0.05 and ratio > 0.50:
                    error_msg = (
                        f"Interest Income ({child_value:,.0f}) is suspiciously close to "
                        f"Operating Income ({parent_value:,.0f}) (diff: {diff_ratio:.2%}, ratio: {ratio:.1%}). "
                        f"This may be misclassified Operating Income."
                    )
                    return False, error_msg
                
                # TERTIARY CHECK: Negative Interest Income is always suspicious
                if child_value < 0:
                    error_msg = (
                        f"Interest Income is negative ({child_value:,.0f}), which is unusual. "
                        f"This may be misclassified."
                    )
                    return False, error_msg
                
                # NOTE: We DON'T reject values that are just large (> 10% ratio) but NOT close
                # Cash-rich companies like AAPL can legitimately have Interest Income > 10% of Operating Income
                # The key indicator of misclassification is being CLOSE to Operating Income, not just large
        
        # For subtraction relationships (weight < 0), child should be smaller than parent
        if weight < 0:
            if abs(child_value) > abs(parent_value) * 1.5:  # Allow some tolerance
                error_msg = (
                    f"{child_concept} ({child_value:,.0f}) is larger than expected "
                    f"relative to {parent_concept} ({parent_value:,.0f}). "
                    f"Relationship: {relationship}"
                )
                return False, error_msg
        
        return True, None
    
    def _check_calculation_linkbase(
        self,
        mapped_concept: str,
        calculation_parent: str,
        calculation_children: List[Dict[str, Any]],
        available_facts: Dict[str, float]
    ) -> Optional[Tuple[bool, Optional[str]]]:
        """
        Check validation using calculation linkbase relationships.
        
        Args:
            mapped_concept: The standard concept that was mapped
            calculation_parent: Parent concept from calculation linkbase
            calculation_children: List of child concepts with weights
            available_facts: Dictionary of available fact values
        
        Returns:
            Tuple of (is_valid, error_message) if validation issue found, None otherwise
        """
        # Try to get formulas for the parent concept
        formulas = self.calculation_engine.get_formulas_for_concept(calculation_parent)
        
        if not formulas:
            return None  # No formulas available for validation
        
        # Check if mapped concept appears in calculation children
        for child_info in calculation_children:
            child_name = child_info.get("child", "")
            weight = child_info.get("weight", 1.0)
            
            # If this child matches our mapped concept, validate it
            if child_name == mapped_concept or mapped_concept in child_name:
                # Validate using the calculation relationship
                if calculation_parent in available_facts:
                    parent_value = available_facts[calculation_parent]
                    
                    # Calculate expected parent from children
                    calculated_parent = 0.0
                    for child_info2 in calculation_children:
                        child_name2 = child_info2.get("child", "")
                        weight2 = child_info2.get("weight", 1.0)
                        if child_name2 in available_facts:
                            calculated_parent += available_facts[child_name2] * weight2
                    
                    # Validate the relationship
                    if abs(calculated_parent - parent_value) > abs(parent_value) * self.tolerance:
                        error_msg = (
                            f"Calculation linkbase validation failed: "
                            f"{calculation_parent} expected {calculated_parent:,.0f} "
                            f"but got {parent_value:,.0f}"
                        )
                        return False, error_msg
        
        return None
    
    def validate_statement_mappings(
        self,
        mapped_concepts: Dict[str, str],
        available_facts: Dict[str, float],
        calculation_relationships: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Validate multiple concept mappings for a financial statement.
        
        Args:
            mapped_concepts: Dictionary mapping XBRL tags to standard concepts
            available_facts: Dictionary mapping standard concepts to values
            calculation_relationships: Optional dictionary mapping parent concepts to child relationships
        
        Returns:
            Dictionary mapping standard concepts to (is_valid, error_message) tuples
        """
        validation_results = {}
        
        for xbrl_tag, standard_concept in mapped_concepts.items():
            calculation_parent = None
            calculation_children = None
            
            if calculation_relationships and standard_concept in calculation_relationships:
                calculation_parent = standard_concept
                calculation_children = calculation_relationships[standard_concept]
            
            is_valid, error_msg = self.validate_mapping(
                mapped_concept=standard_concept,
                calculation_parent=calculation_parent,
                calculation_children=calculation_children,
                available_facts=available_facts,
                context={"xbrl_tag": xbrl_tag}
            )
            
            validation_results[standard_concept] = (is_valid, error_msg)
        
        return validation_results
