# financial4all/xbrl/statement_resolver.py
"""
Enhanced statement resolver using presentation linkbases.

This module provides improved statement resolution that uses presentation
linkbases to identify and extract financial statements more accurately.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

from financial4all.xbrl.presentation import PresentationTree
from financial4all.xbrl.abstract_detection import is_abstract_concept


class StatementType(Enum):
    """Types of financial statements."""
    INCOME_STATEMENT = "IncomeStatement"
    BALANCE_SHEET = "BalanceSheet"
    CASH_FLOW_STATEMENT = "CashFlowStatement"
    STATEMENT_OF_EQUITY = "StatementOfEquity"
    COMPREHENSIVE_INCOME = "ComprehensiveIncome"


# Mapping of primary concepts to statement types
STATEMENT_CONCEPT_MAPPING = {
    'us-gaap_StatementOfIncomeAndComprehensiveIncomeAbstract': StatementType.INCOME_STATEMENT,
    'us-gaap_IncomeStatementAbstract': StatementType.INCOME_STATEMENT,
    'us-gaap_StatementOfFinancialPositionAbstract': StatementType.BALANCE_SHEET,
    'us-gaap_StatementOfCashFlowsAbstract': StatementType.CASH_FLOW_STATEMENT,
    'us-gaap_StatementOfStockholdersEquityAbstract': StatementType.STATEMENT_OF_EQUITY,
    'us-gaap_StatementOfComprehensiveIncomeAbstract': StatementType.COMPREHENSIVE_INCOME,
}


class StatementResolver:
    """
    Enhanced statement resolver using presentation linkbases.
    
    This class identifies and extracts financial statements using
    presentation linkbases for more accurate statement detection.
    """
    
    def __init__(self, presentation_trees: Optional[Dict[str, PresentationTree]] = None):
        """
        Initialize statement resolver.
        
        Args:
            presentation_trees: Optional dictionary of presentation trees by role URI
        """
        self.presentation_trees = presentation_trees or {}
    
    def find_statements(self) -> List[Dict[str, Any]]:
        """
        Find all financial statements from presentation trees.
        
        Returns:
            List of statement metadata dictionaries
        """
        statements = []
        
        for role_uri, tree in self.presentation_trees.items():
            # Get root element
            root_id = tree.root_element_id
            
            # Try to identify statement type from root element
            statement_type = self._identify_statement_type(root_id, tree.definition)
            
            if statement_type:
                statements.append({
                    'role_uri': role_uri,
                    'definition': tree.definition,
                    'statement_type': statement_type.value,
                    'root_element_id': root_id,
                    'element_count': len(tree.all_nodes),
                })
        
        return statements
    
    def _identify_statement_type(self, root_element_id: str, role_definition: str) -> Optional[StatementType]:
        """
        Identify statement type from root element and role definition.
        
        Args:
            root_element_id: Root element ID of the presentation tree
            role_definition: Role definition string
            
        Returns:
            StatementType if identified, None otherwise
        """
        # Check direct mapping
        if root_element_id in STATEMENT_CONCEPT_MAPPING:
            return STATEMENT_CONCEPT_MAPPING[root_element_id]
        
        # Check role definition for keywords
        role_lower = role_definition.lower()
        
        if 'income' in role_lower and 'comprehensive' not in role_lower:
            return StatementType.INCOME_STATEMENT
        elif 'balance' in role_lower or 'financial position' in role_lower:
            return StatementType.BALANCE_SHEET
        elif 'cash flow' in role_lower or 'cashflow' in role_lower:
            return StatementType.CASH_FLOW_STATEMENT
        elif 'equity' in role_lower or 'stockholders' in role_lower:
            return StatementType.STATEMENT_OF_EQUITY
        elif 'comprehensive income' in role_lower:
            return StatementType.COMPREHENSIVE_INCOME
        
        return None
    
    def get_statement_by_type(self, statement_type: str) -> Optional[Dict[str, Any]]:
        """
        Get statement by type name.
        
        Args:
            statement_type: Statement type name (e.g., 'BalanceSheet', 'IncomeStatement')
            
        Returns:
            Statement metadata if found, None otherwise
        """
        statements = self.find_statements()
        for stmt in statements:
            if stmt['statement_type'] == statement_type:
                return stmt
        return None
    
    def get_statement_by_role(self, role_uri: str) -> Optional[Dict[str, Any]]:
        """
        Get statement by role URI.
        
        Args:
            role_uri: Extended link role URI
            
        Returns:
            Statement metadata if found, None otherwise
        """
        if role_uri in self.presentation_trees:
            tree = self.presentation_trees[role_uri]
            statement_type = self._identify_statement_type(tree.root_element_id, tree.definition)
            
            return {
                'role_uri': role_uri,
                'definition': tree.definition,
                'statement_type': statement_type.value if statement_type else None,
                'root_element_id': tree.root_element_id,
                'element_count': len(tree.all_nodes),
            }
        return None
