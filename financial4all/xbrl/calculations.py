# financial4all/xbrl/calculations.py
"""
Calculation linkbase processing and formula application.

This module provides functionality for processing XBRL calculation linkbases
and applying calculation formulas to derive missing values.
"""

import json
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
from pathlib import Path

from financial4all.core import resource_path, log
from financial4all.xbrl.parser import XBRLParser


class CalculationEngine:
    """
    Engine for applying XBRL calculation formulas.
    
    This class processes calculation linkbases and applies formulas
    to derive missing financial values.
    """
    
    def __init__(self, formulas_path: Optional[str] = None):
        """
        Initialize calculation engine.
        
        Args:
            formulas_path: Path to JSON file with calculation formulas
                          (defaults to income_statement_formulas.json)
        """
        self.parser = XBRLParser()
        self.formulas: Dict[str, List[Dict[str, Any]]] = {}
        
        if formulas_path:
            self.load_formulas(formulas_path)
        else:
            # Try to load default formulas
            default_path = resource_path("xbrl_prep/income_statement_formulas.json")
            if Path(default_path).exists():
                self.load_formulas(default_path)
    
    def load_formulas(self, file_path: str) -> None:
        """
        Load calculation formulas from JSON file.
        
        Args:
            file_path: Path to JSON file with formulas
        """
        with open(file_path, "r") as f:
            self.formulas = json.load(f)
        log.debug(f"Loaded {len(self.formulas)} calculation formulas")
    
    def get_formulas_for_concept(self, concept: str) -> List[Dict[str, Any]]:
        """
        Get all formulas for a given concept.
        
        Args:
            concept: XBRL concept name (e.g., "GrossProfit")
            
        Returns:
            List of formula dictionaries
        """
        # Remove namespace prefix if present
        concept_base = concept.replace("us-gaap_", "")
        return self.formulas.get(concept_base, [])
    
    def calculate_value(
        self,
        concept: str,
        child_values: Dict[str, float],
        formula: Dict[str, Any]
    ) -> Optional[float]:
        """
        Calculate a value using a formula and child values.
        
        Args:
            concept: Parent concept name
            child_values: Dictionary mapping child concept names to values
            formula: Formula dictionary with children and weights
            
        Returns:
            Calculated value or None if calculation not possible
        """
        children = formula.get("children", [])
        if not children:
            return None
        
        result = 0.0
        for child_info in children:
            child_name = child_info["child"]
            weight = child_info.get("weight", 1.0)
            
            if child_name not in child_values:
                return None  # Missing required child value
            
            result += child_values[child_name] * weight
        
        return result
    
    def find_calculation_path(
        self,
        target_concept: str,
        available_concepts: Set[str],
        visited: Optional[Set[str]] = None
    ) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
        """
        Find a calculation path to derive a target concept.
        
        Args:
            target_concept: Concept to derive
            available_concepts: Set of concepts with known values
            visited: Set of concepts already visited (to prevent cycles)
            
        Returns:
            List of (concept, formula) tuples representing calculation path,
            or None if no path found
        """
        if visited is None:
            visited = set()
        
        if target_concept in visited:
            return None  # Cycle detected
        
        visited.add(target_concept)
        
        # Check if already available
        if target_concept in available_concepts:
            return []
        
        # Try each formula for this concept
        formulas = self.get_formulas_for_concept(target_concept)
        
        for formula in formulas:
            children = formula.get("children", [])
            if not children:
                continue
            
            # Check if all children are available or derivable
            path = []
            all_children_available = True
            
            for child_info in children:
                child_name = child_info["child"]
                
                if child_name in available_concepts:
                    continue  # Child is available
                
                # Try to derive child
                child_path = self.find_calculation_path(
                    child_name,
                    available_concepts,
                    visited.copy()
                )
                
                if child_path is None:
                    all_children_available = False
                    break
                
                path.extend(child_path)
                path.append((child_name, {}))  # Child will be calculated
            
            if all_children_available:
                path.append((target_concept, formula))
                return path
        
        return None  # No path found
