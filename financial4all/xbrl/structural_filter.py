# financial4all/xbrl/structural_filter.py
"""
Structural element filtering for XBRL (EdgarTools-aligned).

Filters XBRL structural elements (axes, domains, members, tables, abstracts)
from fact resolution and statement output. These are internal XBRL constructs,
not actual financial data.
"""

from typing import Optional

# EdgarTools-aligned: Patterns in labels that indicate structural elements
STRUCTURAL_LABEL_PATTERNS = [
    "[Axis]",
    "[Domain]",
    "[Member]",
    "[Line Items]",
    "[Table]",
    "[Abstract]",
]

# EdgarTools-aligned: Concept suffixes indicating structural elements
STRUCTURAL_CONCEPT_SUFFIXES = ("Axis", "Domain", "Member", "LineItems", "Table")


def is_xbrl_structural_element(
    concept: str,
    label: Optional[str] = None,
) -> bool:
    """
    Check if an item is an XBRL structural element that should be filtered.

    XBRL structural elements include:
    - Axes: Dimensional axes like ProductOrServiceAxis
    - Domains: Domain members like ProductsAndServicesDomain
    - Tables: Hypercube tables like StatementTable
    - Line Items: Container elements like StatementLineItems
    - Root statement abstracts: Top-level abstract concepts with no proper label
      (e.g., StatementOfFinancialPositionAbstract where label equals concept)

    Args:
        concept: XBRL concept name (e.g., "us-gaap_StatementOfFinancialPositionAbstract")
        label: Optional element label. If None, only concept-based checks apply.

    Returns:
        True if this is a structural element that should be excluded from
        user-facing output; False for actual financial data.

    Example:
        >>> is_xbrl_structural_element("us-gaap_ProductOrServiceAxis", "[Axis]")
        True
        >>> is_xbrl_structural_element("us-gaap_Revenues", "Revenues")
        False
    """
    if not concept:
        return False

    # Check label for bracket patterns (e.g., "[Axis]", "[Table]")
    if label:
        for pattern in STRUCTURAL_LABEL_PATTERNS:
            if pattern in label:
                return True

    # Strip namespace prefix for suffix checks
    local_concept = concept.split("_")[-1] if "_" in concept else concept
    local_concept = local_concept.split(":")[-1] if ":" in local_concept else local_concept

    # Check concept name suffix (e.g., "ProductOrServiceAxis", "StatementTable")
    if local_concept.endswith(STRUCTURAL_CONCEPT_SUFFIXES):
        return True

    # Filter root statement abstracts where label equals concept
    # These are structural root nodes like "us-gaap_StatementOfFinancialPositionAbstract"
    # that have no proper label assigned.
    if local_concept.endswith("Abstract") and label is not None and label == concept:
        return True

    return False
