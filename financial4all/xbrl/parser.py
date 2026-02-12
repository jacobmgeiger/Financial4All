# financial4all/xbrl/parser.py
"""
XBRL document parsing.

This module provides functionality for parsing XBRL instance documents,
presentation linkbases, and calculation linkbases.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING

try:
    from lxml import etree as ET
    LXML_AVAILABLE = True
except ImportError:
    import xml.etree.ElementTree as ET
    LXML_AVAILABLE = False

from financial4all.xbrl.facts import FactSet, Fact
from financial4all.xbrl.periods import Period
from financial4all.core import log

if TYPE_CHECKING:
    from financial4all.xbrl.presentation import PresentationTree
    from financial4all.xbrl.dimensions import Table

# Forward reference for PresentationTree
if False:  # TYPE_CHECKING equivalent
    from financial4all.xbrl.presentation import PresentationTree


class XBRLParser:
    """
    Parser for XBRL documents.
    
    This class handles parsing of XBRL instance documents and linkbases.
    Uses lxml for optimized XML parsing when available.
    """
    
    XBRL_NAMESPACE = "http://www.xbrl.org/2003/instance"
    LINKBASE_NAMESPACE = "http://www.xbrl.org/2003/linkbase"
    XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
    XBRDLI_NAMESPACE = "http://xbrl.org/2006/xbrldi"

    def __init__(self):
        """Initialize XBRL parser."""
        self.namespaces = {
            "xbrl": self.XBRL_NAMESPACE,
            "xbrli": self.XBRL_NAMESPACE,
            "xbrldi": self.XBRDLI_NAMESPACE,
            "link": self.LINKBASE_NAMESPACE,
            "xlink": self.XLINK_NAMESPACE,
        }
        
        # Create optimized parser if lxml is available (edgartools-style: 10-30x faster)
        if LXML_AVAILABLE:
            self._parser = ET.XMLParser(
                remove_blank_text=True,
                recover=True,
                huge_tree=True,
                resolve_entities=False,  # Security and speed
            )
        else:
            self._parser = None
            log.warning("lxml not available, using slower xml.etree.ElementTree. Install lxml for better performance.")
    
    def _safe_parse_xml(self, content: Union[str, bytes]) -> ET.Element:
        """
        Safely parse XML content with optimized settings.
        
        Uses lxml with optimized parser settings when available, falling back
        to standard ElementTree if lxml is not installed.
        
        Args:
            content: XML content as string or bytes
            
        Returns:
            Parsed XML root element
            
        Raises:
            ValueError: If XML parsing fails
        """
        try:
            if LXML_AVAILABLE:
                # Convert to bytes for safer parsing
                if isinstance(content, str):
                    content_bytes = content.encode('utf-8')
                else:
                    content_bytes = content
                
                # Parse with optimized lxml parser
                return ET.XML(content_bytes, self._parser)
            else:
                # Fallback to standard ElementTree
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                return ET.fromstring(content)
        except ET.ParseError as e:
            raise ValueError(f"Error parsing XML content: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Unexpected error parsing XML: {str(e)}") from e
    
    def parse_instance_document(self, xml_content: Union[str, bytes]) -> Dict[str, Any]:
        """
        Parse an XBRL instance document.
        
        Args:
            xml_content: XML content as string or bytes
            
        Returns:
            Dictionary with parsed XBRL data
            
        Raises:
            ValueError: If XML parsing fails
        """
        root = self._safe_parse_xml(xml_content)
        
        # Extract facts
        facts = []
        for context in root.findall(".//xbrl:context", self.namespaces):
            context_id = context.get("id")
            # Extract period and entity information from context
            # This is simplified - full implementation would parse all context elements
            
        # Extract units
        units = {}
        for unit in root.findall(".//xbrl:unit", self.namespaces):
            unit_id = unit.get("id")
            # Extract unit information
        
        return {
            "facts": facts,
            "units": units,
        }
    
    def parse_calculation_linkbase(
        self, file_path_or_content: Union[str, Path]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse a calculation linkbase XML file or XML content string.

        This extracts calculation relationships (parent-child formulas)
        from XBRL calculation linkbase files.

        Args:
            file_path_or_content: Path to file or XML content string

        Returns:
            Dictionary mapping parent concepts to lists of formulas

        Raises:
            ValueError: If file cannot be read or parsed
        """
        from collections import defaultdict

        try:
            content = file_path_or_content
            if isinstance(file_path_or_content, str):
                stripped = file_path_or_content.strip()
                if not (stripped.startswith("<") or stripped.startswith("<?xml")):
                    path_val = Path(file_path_or_content)
                    if not path_val.exists():
                        raise ValueError(
                            f"Calculation linkbase file not found: {file_path_or_content}"
                        )
                    content = path_val.read_text(encoding="utf-8")
            elif isinstance(file_path_or_content, Path):
                if not file_path_or_content.exists():
                    raise ValueError(
                        f"Calculation linkbase file not found: {file_path_or_content}"
                    )
                content = file_path_or_content.read_text(encoding="utf-8")

            root = self._safe_parse_xml(content)
        except (FileNotFoundError, IOError) as e:
            raise ValueError(f"Error reading calculation linkbase file: {file_path}") from e
        except ValueError as e:
            raise ValueError(f"Error parsing calculation linkbase file: {file_path}") from e
        calculations = defaultdict(list)
        
        # Process each calculationLink
        # Use optimized XPath if lxml is available, otherwise use findall
        if LXML_AVAILABLE and hasattr(root, 'xpath'):
            calc_links = root.xpath('//link:calculationLink', namespaces=self.namespaces)
        else:
            calc_links = root.findall("link:calculationLink", self.namespaces)
        
        for calc_link in calc_links:
            # Get role URI - handle both lxml and ElementTree attribute access
            if LXML_AVAILABLE:
                role_uri = calc_link.get(f"{{{self.XLINK_NAMESPACE}}}role")
            else:
                role_uri = calc_link.get(f"{{{self.XLINK_NAMESPACE}}}role")
            
            if not role_uri:
                continue
            
            locators = {}
            
            # Build map of labels to concept names
            # Use optimized XPath if available
            if LXML_AVAILABLE and hasattr(calc_link, 'xpath'):
                locs = calc_link.xpath('.//link:loc', namespaces=self.namespaces)
            else:
                locs = calc_link.findall("link:loc", self.namespaces)
            
            for loc in locs:
                label = loc.get(f"{{{self.XLINK_NAMESPACE}}}label")
                href = loc.get(f"{{{self.XLINK_NAMESPACE}}}href")
                if href and label:
                    element_name = href.split("#")[-1].replace("us-gaap_", "")
                    locators[label] = element_name
            
            # Group calculation arcs by parent
            formulas_in_role = defaultdict(list)
            
            # Use optimized XPath if available
            if LXML_AVAILABLE and hasattr(calc_link, 'xpath'):
                calc_arcs = calc_link.xpath('.//link:calculationArc', namespaces=self.namespaces)
            else:
                calc_arcs = calc_link.findall("link:calculationArc", self.namespaces)
            
            for calc_arc in calc_arcs:
                parent_label = calc_arc.get(f"{{{self.XLINK_NAMESPACE}}}from")
                child_label = calc_arc.get(f"{{{self.XLINK_NAMESPACE}}}to")
                
                if not parent_label or not child_label:
                    continue
                
                parent_element = locators.get(parent_label)
                child_element = locators.get(child_label)
                
                if parent_element and child_element:
                    try:
                        weight = float(calc_arc.get("weight", 1.0))
                    except (ValueError, TypeError):
                        weight = 1.0
                    
                    formulas_in_role[parent_element].append({
                        "child": child_element,
                        "weight": weight,
                    })
            
            # Add formulas to main dictionary
            for parent, children in formulas_in_role.items():
                new_formula = {
                    "role": role_uri,
                    "children": sorted(children, key=lambda x: x["child"]),
                }
                
                # Check for duplicates
                new_children_set = frozenset(c["child"] for c in new_formula["children"])
                is_duplicate = any(
                    new_children_set == frozenset(c["child"] for c in existing_formula["children"])
                    for existing_formula in calculations[parent]
                )
                
                if not is_duplicate:
                    calculations[parent].append(new_formula)
        
        return dict(calculations)
    
    def parse_presentation_linkbase(
        self, file_path_or_content: Union[str, Path]
    ) -> Dict[str, "PresentationTree"]:
        """
        Parse a presentation linkbase XML file or XML content string.

        This extracts presentation relationships (statement structure)
        from XBRL presentation linkbase files and builds presentation trees.

        Args:
            file_path_or_content: Path to presentation linkbase XML file,
                or XML content as string (when fetching from SEC)

        Returns:
            Dictionary mapping role_uri -> PresentationTree

        Raises:
            ValueError: If file cannot be read or parsed
        """
        from financial4all.xbrl.presentation import PresentationTree, PresentationNode

        try:
            content = file_path_or_content
            if isinstance(file_path_or_content, str):
                stripped = file_path_or_content.strip()
                # Treat as XML content if it looks like XML (not a path)
                if stripped.startswith("<") or stripped.startswith("<?xml"):
                    content = file_path_or_content
                else:
                    path_val = Path(file_path_or_content)
                    if not path_val.exists():
                        raise ValueError(
                            f"Presentation linkbase file not found: {file_path_or_content}"
                        )
                    content = path_val.read_text(encoding="utf-8")
            elif isinstance(file_path_or_content, Path):
                if not file_path_or_content.exists():
                    raise ValueError(
                        f"Presentation linkbase file not found: {file_path_or_content}"
                    )
                content = file_path_or_content.read_text(encoding="utf-8")

            root = self._safe_parse_xml(content)
        except (FileNotFoundError, IOError) as e:
            raise ValueError(f"Error reading presentation linkbase file: {file_path}") from e
        except ValueError as e:
            raise ValueError(f"Error parsing presentation linkbase file: {file_path}") from e
        
        presentation_trees = {}
        
        # Extract presentation links
        # Use optimized XPath if lxml is available
        if LXML_AVAILABLE and hasattr(root, 'xpath'):
            presentation_links = root.xpath('//link:presentationLink', namespaces=self.namespaces)
        else:
            presentation_links = root.findall("link:presentationLink", self.namespaces)
        
        for link in presentation_links:
            role_uri = link.get(f"{{{self.XLINK_NAMESPACE}}}role")
            if not role_uri:
                continue
            
            # Store role information
            role_id = role_uri.split('/')[-1] if '/' in role_uri else role_uri
            role_def = role_id.replace('_', ' ')
            
            # Build locator map
            loc_map = {}
            if LXML_AVAILABLE and hasattr(link, 'xpath'):
                locs = link.xpath('.//link:loc', namespaces=self.namespaces)
            else:
                locs = link.findall("link:loc", self.namespaces)
            
            for loc in locs:
                label = loc.get(f"{{{self.XLINK_NAMESPACE}}}label")
                href = loc.get(f"{{{self.XLINK_NAMESPACE}}}href")
                if label and href:
                    # Extract element ID from href
                    element_id = href.split("#")[-1] if "#" in href else href
                    # Normalize namespace prefix
                    if ':' in element_id:
                        prefix, name = element_id.split(':', 1)
                        element_id = f"{prefix}_{name}"
                    loc_map[label] = element_id
            
            # Extract presentation arcs
            relationships = []
            if LXML_AVAILABLE and hasattr(link, 'xpath'):
                arcs = link.xpath('.//link:presentationArc', namespaces=self.namespaces)
            else:
                arcs = link.findall("link:presentationArc", self.namespaces)
            
            for arc in arcs:
                from_ref = arc.get(f"{{{self.XLINK_NAMESPACE}}}from")
                to_ref = arc.get(f"{{{self.XLINK_NAMESPACE}}}to")
                
                if not from_ref or not to_ref:
                    continue
                
                from_element = loc_map.get(from_ref)
                to_element = loc_map.get(to_ref)
                
                if not from_element or not to_element:
                    continue
                
                # Parse order attribute
                order = 0.0
                order_attr = arc.get(f"{{{self.XLINK_NAMESPACE}}}order") or arc.get("order")
                if order_attr:
                    try:
                        order = float(order_attr)
                    except (ValueError, TypeError):
                        order = 0.0
                
                preferred_label = arc.get("preferredLabel")
                
                relationships.append({
                    'from_element': from_element,
                    'to_element': to_element,
                    'order': order,
                    'preferred_label': preferred_label
                })
            
            # Build presentation tree if we have relationships
            if relationships:
                tree = self._build_presentation_tree(role_uri, role_def, relationships)
                if tree:
                    presentation_trees[role_uri] = tree
        
        return presentation_trees
    
    def parse_definition_linkbase(self, file_path: Union[str, Path]) -> Dict[str, List["Table"]]:
        """
        Parse a definition linkbase XML file.
        
        This extracts dimensional structures (tables, axes, domains) from
        XBRL definition linkbase files.
        
        Args:
            file_path: Path to definition linkbase XML file
            
        Returns:
            Dictionary mapping role_uri -> list of Table objects
            
        Raises:
            ValueError: If file cannot be read or parsed
        """
        from financial4all.xbrl.dimensions import Table, Axis, Domain
        
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise ValueError(f"Definition linkbase file not found: {file_path}")
            
            # Read and parse file
            content = file_path_obj.read_text(encoding='utf-8')
            root = self._safe_parse_xml(content)
        except (FileNotFoundError, IOError) as e:
            raise ValueError(f"Error reading definition linkbase file: {file_path}") from e
        except ValueError as e:
            raise ValueError(f"Error parsing definition linkbase file: {file_path}") from e
        
        # XBRL Dimensions arcrole URIs
        HYPERCUBE_DIMENSION = "http://xbrl.org/int/dim/arcrole/hypercube-dimension"
        DIMENSION_DOMAIN = "http://xbrl.org/int/dim/arcrole/dimension-domain"
        DOMAIN_MEMBER = "http://xbrl.org/int/dim/arcrole/domain-member"
        ALL = "http://xbrl.org/int/dim/arcrole/all"
        
        tables_by_role: Dict[str, List[Table]] = {}
        axes: Dict[str, Axis] = {}
        domains: Dict[str, Domain] = {}
        
        # Extract definition links
        if LXML_AVAILABLE and hasattr(root, 'xpath'):
            definition_links = root.xpath('//link:definitionLink', namespaces=self.namespaces)
        else:
            definition_links = root.findall("link:definitionLink", self.namespaces)
        
        for link in definition_links:
            role_uri = link.get(f"{{{self.XLINK_NAMESPACE}}}role")
            if not role_uri:
                continue
            
            # Build locator map
            loc_map = {}
            if LXML_AVAILABLE and hasattr(link, 'xpath'):
                locs = link.xpath('.//link:loc', namespaces=self.namespaces)
            else:
                locs = link.findall("link:loc", self.namespaces)
            
            for loc in locs:
                label = loc.get(f"{{{self.XLINK_NAMESPACE}}}label")
                href = loc.get(f"{{{self.XLINK_NAMESPACE}}}href")
                if label and href:
                    element_id = href.split("#")[-1] if "#" in href else href
                    if ':' in element_id:
                        prefix, name = element_id.split(':', 1)
                        element_id = f"{prefix}_{name}"
                    loc_map[label] = element_id
            
            # Extract arcs and group by arcrole
            grouped_rels = defaultdict(list)
            if LXML_AVAILABLE and hasattr(link, 'xpath'):
                arcs = link.xpath('.//link:definitionArc', namespaces=self.namespaces)
            else:
                arcs = link.findall("link:definitionArc", self.namespaces)
            
            for arc in arcs:
                from_ref = arc.get(f"{{{self.XLINK_NAMESPACE}}}from")
                to_ref = arc.get(f"{{{self.XLINK_NAMESPACE}}}to")
                arcrole = arc.get(f"{{{self.XLINK_NAMESPACE}}}arcrole")
                
                if not from_ref or not to_ref or not arcrole:
                    continue
                
                from_element = loc_map.get(from_ref)
                to_element = loc_map.get(to_ref)
                
                if not from_element or not to_element:
                    continue
                
                # Parse order attribute
                order = 0.0
                order_attr = arc.get(f"{{{self.XLINK_NAMESPACE}}}order") or arc.get("order")
                if order_attr:
                    try:
                        order = float(order_attr)
                    except (ValueError, TypeError):
                        order = 0.0
                
                grouped_rels[arcrole].append({
                    'from_element': from_element,
                    'to_element': to_element,
                    'order': order
                })
            
            # Process dimensional relationships
            hypercube_axes = defaultdict(list)
            
            # Process hypercube-dimension relationships
            if HYPERCUBE_DIMENSION in grouped_rels:
                for rel in grouped_rels[HYPERCUBE_DIMENSION]:
                    table_id = rel['from_element']
                    axis_id = rel['to_element']
                    hypercube_axes[table_id].append(axis_id)
                    
                    # Create or update axis
                    if axis_id not in axes:
                        axes[axis_id] = Axis(element_id=axis_id)
            
            # Process dimension-domain relationships
            if DIMENSION_DOMAIN in grouped_rels:
                for rel in grouped_rels[DIMENSION_DOMAIN]:
                    axis_id = rel['from_element']
                    domain_id = rel['to_element']
                    
                    if axis_id in axes:
                        axes[axis_id].domain_id = domain_id
                    
                    if domain_id not in domains:
                        domains[domain_id] = Domain(element_id=domain_id)
            
            # Process domain-member relationships
            if DOMAIN_MEMBER in grouped_rels:
                domain_members = defaultdict(list)
                for rel in grouped_rels[DOMAIN_MEMBER]:
                    domain_id = rel['from_element']
                    member_id = rel['to_element']
                    domain_members[domain_id].append(member_id)
                    
                    if domain_id not in domains:
                        domains[domain_id] = Domain(element_id=domain_id)
                
                for domain_id, members in domain_members.items():
                    domains[domain_id].members = members
            
            # Process 'all' relationships to build tables
            if ALL in grouped_rels:
                tables = []
                for rel in grouped_rels[ALL]:
                    line_items_id = rel['from_element']
                    hypercube_id = rel['to_element']
                    
                    # Only process if this hypercube has axes defined
                    if hypercube_id in hypercube_axes:
                        table = Table(
                            element_id=hypercube_id,
                            role_uri=role_uri,
                            axes=hypercube_axes[hypercube_id],
                            line_items=[line_items_id],
                            closed=False
                        )
                        tables.append(table)
                
                if tables:
                    tables_by_role[role_uri] = tables
        
        return tables_by_role
    
    def _build_presentation_tree(
        self,
        role_uri: str,
        role_def: str,
        relationships: List[Dict[str, Any]]
    ) -> Optional["PresentationTree"]:
        """
        Build a presentation tree from relationships.
        
        Args:
            role_uri: Extended link role URI
            role_def: Role definition
            relationships: List of relationships (from_element, to_element, order, preferred_label)
            
        Returns:
            PresentationTree object or None if no root elements found
        """
        from financial4all.xbrl.presentation import PresentationTree, PresentationNode
        
        # Group relationships by source element
        from_map = defaultdict(list)
        to_set = set()
        
        for rel in relationships:
            from_element = rel['from_element']
            to_element = rel['to_element']
            from_map[from_element].append(rel)
            to_set.add(to_element)
        
        # Find root elements (appear as 'from' but not as 'to')
        root_elements = sorted(set(from_map.keys()) - to_set)
        
        if not root_elements:
            return None
        
        # Create presentation tree
        tree = PresentationTree(
            role_uri=role_uri,
            definition=role_def,
            root_element_id=root_elements[0]
        )
        
        # Build tree recursively
        for root_id in root_elements:
            self._build_presentation_subtree(root_id, None, 0, from_map, tree.all_nodes)
        
        return tree
    
    def _build_presentation_subtree(
        self,
        element_id: str,
        parent_id: Optional[str],
        depth: int,
        from_map: Dict[str, List[Dict[str, Any]]],
        all_nodes: Dict[str, "PresentationNode"]
    ) -> None:
        """
        Recursively build a presentation subtree.
        
        Args:
            element_id: Current element ID
            parent_id: Parent element ID
            depth: Current depth in tree
            from_map: Map of relationships by source element
            all_nodes: Dictionary to store all nodes
        """
        from financial4all.xbrl.presentation import PresentationNode
        
        # Create node
        node = PresentationNode(
            element_id=element_id,
            parent=parent_id,
            children=[],
            depth=depth
        )
        
        # Add to collection
        all_nodes[element_id] = node
        
        # Process children
        if element_id in from_map:
            # Sort children by order
            children = sorted(from_map[element_id], key=lambda r: r['order'])
            
            for rel in children:
                child_id = rel['to_element']
                
                # Add child to parent's children list
                node.children.append(child_id)
                
                # Set preferred label
                preferred_label = rel.get('preferred_label')
                
                # Recursively build child subtree
                self._build_presentation_subtree(
                    child_id, element_id, depth + 1, from_map, all_nodes
                )
                
                # Update preferred label, order, and is_total after child is built (EdgarTools parity)
                if child_id in all_nodes:
                    if preferred_label:
                        all_nodes[child_id].preferred_label = preferred_label
                        # totalLabel role: filers typically name label ref with "totalLabel"
                        all_nodes[child_id].is_total = "totallabel" in preferred_label.lower()
                    all_nodes[child_id].order = rel['order']
