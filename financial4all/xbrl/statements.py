# financial4all/xbrl/statements.py
"""
Statement resolution for XBRL financial statements.

This module provides functionality for identifying and extracting
financial statements (Income Statement, Balance Sheet, Cash Flow)
from XBRL data.
"""

from typing import Dict, List, Optional, Set
from enum import Enum

from financial4all.xbrl.facts import FactSet


class StatementType(Enum):
    """Types of financial statements."""
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW_STATEMENT = "cash_flow_statement"
    STATEMENT_OF_EQUITY = "statement_of_equity"


class StatementResolver:
    """
    Resolves financial statements from XBRL facts.
    
    This class identifies and extracts financial statements using
    presentation linkbases and concept mappings.
    """
    
    # Common concept patterns for each statement type
    INCOME_STATEMENT_CONCEPTS = {
        "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomer",
        "GrossProfit", "OperatingIncomeLoss", "NetIncomeLoss",
        "CostOfRevenue", "OperatingExpenses",
    }
    
    BALANCE_SHEET_CONCEPTS = {
        "Assets", "Liabilities", "StockholdersEquity",
        "CurrentAssets", "CurrentLiabilities",
    }
    
    CASH_FLOW_CONCEPTS = {
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInFinancingActivities",
    }
    
    def __init__(self):
        """Initialize statement resolver."""
        pass
    
    def identify_statements(self, fact_set: FactSet) -> Dict[StatementType, FactSet]:
        """
        Identify financial statements from facts.
        
        Args:
            fact_set: FactSet to analyze
            
        Returns:
            Dictionary mapping statement types to filtered FactSets
        """
        results = {}
        
        # Identify income statement concepts
        income_concepts = self._find_matching_concepts(
            fact_set,
            self.INCOME_STATEMENT_CONCEPTS
        )
        if income_concepts:
            results[StatementType.INCOME_STATEMENT] = fact_set.filter_by_concept(
                list(income_concepts)[0]  # Use first match
            )
        
        # Identify balance sheet concepts
        balance_concepts = self._find_matching_concepts(
            fact_set,
            self.BALANCE_SHEET_CONCEPTS
        )
        if balance_concepts:
            results[StatementType.BALANCE_SHEET] = fact_set.filter_by_concept(
                list(balance_concepts)[0]
            )
        
        # Identify cash flow concepts
        cash_flow_concepts = self._find_matching_concepts(
            fact_set,
            self.CASH_FLOW_CONCEPTS
        )
        if cash_flow_concepts:
            results[StatementType.CASH_FLOW_STATEMENT] = fact_set.filter_by_concept(
                list(cash_flow_concepts)[0]
            )
        
        return results
    
    def _find_matching_concepts(
        self,
        fact_set: FactSet,
        target_concepts: Set[str]
    ) -> Set[str]:
        """
        Find concepts in fact set that match target concepts.
        
        Args:
            fact_set: FactSet to search
            target_concepts: Set of concept names to find
            
        Returns:
            Set of matching concept names found in fact set
        """
        available_concepts = fact_set.get_unique_concepts()
        matches = set()
        
        for concept in available_concepts:
            # Remove namespace prefix
            concept_base = concept.replace("us-gaap_", "")
            
            # Check for exact match or partial match
            for target in target_concepts:
                if concept_base == target or concept_base.endswith(target):
                    matches.add(concept)
                    break
        
        return matches
