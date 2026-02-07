# financial4all/xbrl/dimensions.py
"""
Dimensional fact structures for XBRL.

This module provides functionality for handling dimensional facts including
tables, axes, and domains from definition linkbases.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class Domain:
    """
    Represents a domain in XBRL dimensions.
    
    A domain is a set of possible values for a dimension (e.g., business segments,
    geographic regions).
    
    Attributes:
        element_id: Domain element identifier
        label: Human-readable label
        members: List of member element IDs
    """
    
    element_id: str
    label: Optional[str] = None
    members: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        """String representation of Domain."""
        return f"Domain(element_id={self.element_id}, members={len(self.members)})"


@dataclass
class Axis:
    """
    Represents an axis in XBRL dimensions.
    
    An axis represents a dimension (e.g., "Business Segment", "Geographic Region").
    
    Attributes:
        element_id: Axis element identifier
        label: Human-readable label
        domain_id: Associated domain ID
    """
    
    element_id: str
    label: Optional[str] = None
    domain_id: Optional[str] = None
    
    def __repr__(self) -> str:
        """String representation of Axis."""
        return f"Axis(element_id={self.element_id}, domain={self.domain_id})"


@dataclass
class Table:
    """
    Represents a table (hypercube) in XBRL dimensions.
    
    A table defines a multi-dimensional structure for facts.
    
    Attributes:
        element_id: Table/hypercube element identifier
        label: Human-readable label
        role_uri: Extended link role URI
        axes: List of axis element IDs
        line_items: List of line item element IDs
        closed: Whether the table is closed (all dimensions required)
    """
    
    element_id: str
    label: Optional[str] = None
    role_uri: Optional[str] = None
    axes: List[str] = field(default_factory=list)
    line_items: List[str] = field(default_factory=list)
    closed: bool = False
    
    def __repr__(self) -> str:
        """String representation of Table."""
        return f"Table(element_id={self.element_id}, axes={len(self.axes)}, line_items={len(self.line_items)})"
