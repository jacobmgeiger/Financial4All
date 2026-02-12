# financial4all/xbrl/presentation.py
"""
Presentation tree structures and sign normalization for XBRL financial statements.

This module provides:
- Presentation trees from presentation linkbases
- Preferred sign transformation (EdgarTools-aligned) for display consistency
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from collections import defaultdict

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class PresentationNode:
    """
    Represents a node in a presentation tree.
    
    Attributes:
        element_id: XBRL element/concept identifier
        parent: Parent element ID (None for root)
        children: List of child element IDs
        depth: Depth in the tree (0 for root)
        element_name: Human-readable element name
        standard_label: Standard label for the element
        labels: Dictionary of labels by role
        is_abstract: Whether this is an abstract element
        preferred_label: Preferred label role URI
        order: Order attribute value
    """
    
    element_id: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    depth: int = 0
    element_name: Optional[str] = None
    standard_label: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    is_abstract: bool = False
    preferred_label: Optional[str] = None
    order: float = 0.0
    is_total: bool = False  # True when preferred_label uses totalLabel role (EdgarTools parity)

    def __repr__(self) -> str:
        """String representation of PresentationNode."""
        return f"PresentationNode(element_id={self.element_id}, depth={self.depth}, children={len(self.children)})"


@dataclass
class PresentationTree:
    """
    Represents a presentation tree for a financial statement role.
    
    Attributes:
        role_uri: Extended link role URI
        definition: Role definition/description
        root_element_id: Root element ID
        all_nodes: Dictionary mapping element_id -> PresentationNode
    """
    
    role_uri: str
    definition: str
    root_element_id: str
    all_nodes: Dict[str, PresentationNode] = field(default_factory=dict)
    
    def get_node(self, element_id: str) -> Optional[PresentationNode]:
        """Get a node by element ID."""
        return self.all_nodes.get(element_id)
    
    def get_children(self, element_id: str) -> List[PresentationNode]:
        """Get child nodes for an element."""
        node = self.get_node(element_id)
        if not node:
            return []
        return [self.all_nodes.get(child_id) for child_id in node.children if child_id in self.all_nodes]
    
    def get_path_to_root(self, element_id: str) -> List[PresentationNode]:
        """Get path from element to root."""
        path = []
        current_id = element_id
        while current_id:
            node = self.get_node(current_id)
            if not node:
                break
            path.append(node)
            current_id = node.parent
        return path
    
    def __repr__(self) -> str:
        """String representation of PresentationTree."""
        return f"PresentationTree(role={self.role_uri}, nodes={len(self.all_nodes)})"


# --- Preferred Sign Transformation (EdgarTools-aligned) ---
# Income statement: expenses shown as positive (COGS, SG&A, R&D, etc.)
# Cash flow: outflows (CapEx, dividends) shown as negative; inflows positive

# Income statement columns that are expenses/costs (stored as credit/debit):
# If value is negative in instance, flip to positive for display
INCOME_STATEMENT_EXPENSE_COLUMNS = frozenset({
    "Cost of Revenue",
    "R&D Expenses",
    "SG&A Expenses",
    "General and Administrative Expense",
    "Selling and Marketing Expense",
    "Operating Expenses",
    "Restructuring and other charges",
    "Other Operating Expense",
    "Asset Impairment Charges",
    "Interest Expense",
    "Taxes",
})

# Cash flow columns that are outflows (payments):
# If value is positive (cash outflow), flip to negative for display
CASH_FLOW_OUTFLOW_COLUMNS = frozenset({
    "CapEx",
    "Capital Expenditures",
    "Payments for Property, Plant and Equipment",
    "Dividends Paid",
    "Share Repurchases",
    "Debt Repayment",
})


def apply_presentation(
    df: "pd.DataFrame",
    statement_type: str,
    expense_columns: Optional[Set[str]] = None,
    outflow_columns: Optional[Set[str]] = None,
) -> "pd.DataFrame":
    """
    Apply EdgarTools-style preferred sign transformation to statement DataFrame.

    Income statement: Ensure expenses (COGS, SG&A, etc.) display as positive.
    Cash flow: Ensure outflows (CapEx, dividends) display as negative.

    Args:
        df: Statement DataFrame with metric names as columns
        statement_type: "IncomeStatement", "CashFlowStatement", or "BalanceSheet"
        expense_columns: Override for income statement expense columns (default: INCOME_STATEMENT_EXPENSE_COLUMNS)
        outflow_columns: Override for cash flow outflow columns (default: CASH_FLOW_OUTFLOW_COLUMNS)

    Returns:
        DataFrame with sign-normalized values (modifies copy, not in place)
    """
    if not PANDAS_AVAILABLE or df is None or df.empty:
        return df

    result = df.copy()
    cols = set(result.columns)

    if statement_type == "IncomeStatement":
        exp_cols = expense_columns or INCOME_STATEMENT_EXPENSE_COLUMNS
        for col in exp_cols:
            if col in cols:
                # Flip negative expense values to positive (expenses shown as positive)
                mask = result[col].notna() & (result[col] < 0)
                result.loc[mask, col] = -result.loc[mask, col]

    elif statement_type == "CashFlowStatement":
        out_cols = outflow_columns or CASH_FLOW_OUTFLOW_COLUMNS
        for col in out_cols:
            if col in cols:
                # Flip positive outflow values to negative (outflows shown as negative)
                mask = result[col].notna() & (result[col] > 0)
                result.loc[mask, col] = -result.loc[mask, col]

    # BalanceSheet: no sign transformation

    return result
