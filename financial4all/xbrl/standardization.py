# financial4all/xbrl/standardization.py
"""
Cross-company standardization for XBRL concepts.

This module provides functionality for standardizing XBRL concept names
across different companies to enable cross-company comparisons.
"""

from typing import Dict, List, Set, Optional
from collections import defaultdict

from financial4all.core import log


class StandardizationStore:
    """
    Store for standardization mappings.
    
    Maps standardized concept names to XBRL concept names and vice versa.
    """
    
    def __init__(self):
        """Initialize standardization store."""
        # Maps standardized name -> list of XBRL concept names
        self.mappings: Dict[str, List[str]] = {}
        
        # Reverse mapping: XBRL concept -> standardized name
        self.reverse_mappings: Dict[str, str] = {}
        
        # Load default mappings
        self._load_default_mappings()
    
    def _load_default_mappings(self) -> None:
        """Load default standardization mappings."""
        # Income Statement mappings
        self.add_mapping("Revenue", [
            "SalesRevenueNet",
            "Revenues",
            "RevenueFromContractWithCustomer",
        ])
        
        self.add_mapping("Cost of Revenue", [
            "CostOfRevenue",
            "CostOfGoodsAndServicesSold",
        ])
        
        self.add_mapping("Gross Profit", ["GrossProfit"])
        self.add_mapping("Operating Income", ["OperatingIncomeLoss"])
        self.add_mapping("Net Income", [
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
        ])
        
        # Balance Sheet mappings
        self.add_mapping("Total Assets", ["Assets"])
        self.add_mapping("Total Liabilities", ["Liabilities"])
        self.add_mapping("Stockholders Equity", [
            "StockholdersEquity",
            "Equity",
        ])
        
        log.debug(f"Loaded {len(self.mappings)} standardization mappings")
    
    def add_mapping(self, standard_name: str, xbrl_concepts: List[str]) -> None:
        """
        Add a standardization mapping.
        
        Args:
            standard_name: Standardized concept name
            xbrl_concepts: List of XBRL concept names that map to this standard name
        """
        self.mappings[standard_name] = xbrl_concepts
        
        # Update reverse mapping
        for concept in xbrl_concepts:
            # Handle both with and without namespace
            self.reverse_mappings[concept] = standard_name
            self.reverse_mappings[f"us-gaap_{concept}"] = standard_name
    
    def get_standard_name(self, xbrl_concept: str) -> Optional[str]:
        """
        Get standardized name for an XBRL concept.
        
        Args:
            xbrl_concept: XBRL concept name
            
        Returns:
            Standardized name or None if not found
        """
        # Try exact match first
        if xbrl_concept in self.reverse_mappings:
            return self.reverse_mappings[xbrl_concept]
        
        # Try without namespace
        concept_base = xbrl_concept.replace("us-gaap_", "")
        if concept_base in self.reverse_mappings:
            return self.reverse_mappings[concept_base]
        
        return None
    
    def get_xbrl_concepts(self, standard_name: str) -> List[str]:
        """
        Get XBRL concept names for a standardized name.
        
        Args:
            standard_name: Standardized concept name
            
        Returns:
            List of XBRL concept names
        """
        return self.mappings.get(standard_name, [])


# Global standardization store instance
_standardization_store: Optional[StandardizationStore] = None


def get_default_store() -> StandardizationStore:
    """
    Get the default standardization store instance.
    
    Returns:
        StandardizationStore instance
    """
    global _standardization_store
    if _standardization_store is None:
        _standardization_store = StandardizationStore()
    return _standardization_store
