# financial4all/xbrl/presentation.py
"""
Presentation tree structures for XBRL financial statements.

This module provides functionality for building and managing presentation trees
that represent the structure of financial statements from presentation linkbases.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from collections import defaultdict


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
