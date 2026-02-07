# financial4all/xbrl/models.py
"""
Data models for XBRL parsing.

This module defines the core data structures used throughout the XBRL parser.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from financial4all.xbrl.core import STANDARD_LABEL, TERSE_LABEL
from financial4all.xbrl.standardization import get_default_store

# Re-export from existing modules
from financial4all.xbrl.presentation import PresentationNode, PresentationTree
from financial4all.xbrl.dimensions import Axis, Domain, Table

__all__ = [
    'ElementCatalog',
    'Context',
    'Fact',
    'Footnote',
    'PresentationNode',
    'PresentationTree',
    'CalculationNode',
    'CalculationTree',
    'Axis',
    'Domain',
    'Table',
    'select_display_label',
    'XBRLProcessingError',
]


def select_display_label(
    labels: Dict[str, str],
    preferred_label: Optional[str] = None,
    standard_label: Optional[str] = None,
    element_id: Optional[str] = None,
    element_name: Optional[str] = None
) -> str:
    """
    Select the most appropriate label for display, following a consistent priority order.
    Includes standardization mapping to provide consistent labels across companies.

    Args:
        labels: Dictionary of available labels
        preferred_label: Role of the preferred label (if specified in presentation linkbase)
        standard_label: The standard label content (if available)
        element_id: Element ID (fallback)
        element_name: Element name (alternative fallback)

    Returns:
        The selected label according to priority rules, with standardization applied if available
    """
    # First, select the best available label using existing priority logic
    selected_label = None
    # Track if we used a company-specific label (preferred or terse) vs a generic fallback
    # Standardization should only override generic labels, not company-specific ones
    used_company_label = False

    # 1. Use preferred label if specified and available
    if preferred_label and labels and preferred_label in labels:
        selected_label = labels[preferred_label]
        used_company_label = True

    # 2. Use terse label if available (more user-friendly)
    elif labels and TERSE_LABEL in labels:
        selected_label = labels[TERSE_LABEL]
        used_company_label = True

    # 3. Fall back to standard label
    elif standard_label:
        selected_label = standard_label

    # 4. Try STANDARD_LABEL directly from labels dict
    elif labels and STANDARD_LABEL in labels:
        selected_label = labels[STANDARD_LABEL]

    # 5. Take any available label
    elif labels:
        selected_label = next(iter(labels.values()), "")

    # 6. Use element name if available
    elif element_name:
        selected_label = element_name

    # 7. Last resort: element ID
    else:
        selected_label = element_id or ""

    # Apply standardization only when using generic labels (not company-specific preferred/terse)
    # This preserves company-specific context like "Other intangible assets, net" instead of
    # generic "Intangible Assets"
    if element_id and selected_label and not used_company_label:
        try:
            # Try to get standardized concept using the singleton store
            standardized_label = get_default_store().get_standard_name(element_id)

            if standardized_label:
                return standardized_label

        except (ImportError, AttributeError):
            # Standardization not available, continue with selected label
            pass
        except Exception:
            # Any other error in standardization, continue with selected label
            pass

    return selected_label


@dataclass
class ElementCatalog:
    """
    A catalog of XBRL elements with their properties.

    This is the base data structure for element metadata.

    Attributes:
        name: The name of the element (e.g., "us-gaap_NetIncome")
        data_type: The data type of the element (e.g., "monetary", "string", etc.)
        period_type: The period type of the element (e.g., "instant", "duration")
        balance: The balance type of the element (e.g., "debit", "credit", or None)
        abstract: Whether the element is abstract (True/False)
        labels: A dictionary of labels for the element, keyed by role URI
    """

    name: str
    data_type: str
    period_type: str
    balance: Optional[str] = None
    abstract: bool = False
    labels: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.name


@dataclass
class Context:
    """
    An XBRL context defining entity, period, and dimensional information.

    This corresponds to the Context Registry in the design document.
    """
    context_id: str
    entity: Dict[str, Any] = field(default_factory=dict)
    period: Dict[str, Any] = field(default_factory=dict)
    dimensions: Dict[str, str] = field(default_factory=dict)

    @property
    def period_string(self) -> str:
        """Return a human-readable string representation of the period."""
        if self.period.get('type') == 'instant':
            return f"As of {self.period.get('instant')}"
        elif self.period.get('type') == 'duration':
            return f"From {self.period.get('startDate')} to {self.period.get('endDate')}"
        else:
            return "Forever"


@dataclass
class Fact:
    """
    An XBRL fact with value and references to context, unit, and element.

    This corresponds to the Fact Database in the design document.

    The instance_id field is used to differentiate between duplicate facts
    that share the same element_id and context_ref. When a fact has no
    duplicates, instance_id will be None.

    The fact_id field preserves the original id attribute from the XML element,
    enabling linkage with footnotes.
    """
    element_id: str
    context_ref: str
    value: str
    unit_ref: Optional[str] = None
    decimals: Optional[Union[int, str]] = None  # int or "INF"
    numeric_value: Optional[float] = None
    footnotes: List[str] = field(default_factory=list)
    instance_id: Optional[int] = None
    fact_id: Optional[str] = None  # Original id attribute from the XML


@dataclass
class Footnote:
    """
    Represents an XBRL footnote with its text content and related facts.

    Footnotes are linked to facts via footnoteArc elements that connect
    fact IDs to footnote IDs using xlink:from and xlink:to attributes.
    """
    footnote_id: str
    text: str
    lang: Optional[str] = "en-US"
    role: Optional[str] = None
    related_fact_ids: List[str] = field(default_factory=list)


@dataclass
class CalculationNode:
    """
    A node in the calculation hierarchy.

    This corresponds to the Calculation Node in the design document.
    """
    element_id: str
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    weight: float = 1.0
    order: float = 0.0

    # Information linked from schema
    balance_type: Optional[str] = None  # "debit", "credit", or None
    period_type: Optional[str] = None  # "instant" or "duration"


@dataclass
class CalculationTree:
    """
    A calculation tree for a specific role.

    This corresponds to the Calculation Network in the design document.
    """
    role_uri: str
    definition: str
    root_element_id: str
    all_nodes: Dict[str, CalculationNode] = field(default_factory=dict)


class XBRLProcessingError(Exception):
    """Exception raised for errors during XBRL processing."""
    pass
