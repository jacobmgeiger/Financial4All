# financial4all/xbrl/parser.py
"""
XBRL document parsing.

This module provides functionality for parsing XBRL instance documents,
presentation linkbases, and calculation linkbases.
"""

from typing import Dict, List, Optional, Any
import xml.etree.ElementTree as ET
from pathlib import Path

from financial4all.xbrl.facts import FactSet, Fact
from financial4all.xbrl.periods import Period


class XBRLParser:
    """
    Parser for XBRL documents.
    
    This class handles parsing of XBRL instance documents and linkbases.
    """
    
    XBRL_NAMESPACE = "http://www.xbrl.org/2003/instance"
    LINKBASE_NAMESPACE = "http://www.xbrl.org/2003/linkbase"
    XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
    
    def __init__(self):
        """Initialize XBRL parser."""
        self.namespaces = {
            "xbrl": self.XBRL_NAMESPACE,
            "link": self.LINKBASE_NAMESPACE,
            "xlink": self.XLINK_NAMESPACE,
        }
    
    def parse_instance_document(self, xml_content: str) -> Dict[str, Any]:
        """
        Parse an XBRL instance document.
        
        Args:
            xml_content: XML content as string
            
        Returns:
            Dictionary with parsed XBRL data
        """
        root = ET.fromstring(xml_content)
        
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
    
    def parse_calculation_linkbase(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse a calculation linkbase XML file.
        
        This extracts calculation relationships (parent-child formulas)
        from XBRL calculation linkbase files.
        
        Args:
            file_path: Path to calculation linkbase XML file
            
        Returns:
            Dictionary mapping parent concepts to lists of formulas
        """
        from collections import defaultdict
        
        try:
            tree = ET.parse(file_path)
        except ET.ParseError as e:
            raise ValueError(f"Error parsing XML file: {file_path}") from e
        
        root = tree.getroot()
        calculations = defaultdict(list)
        
        # Process each calculationLink
        for calc_link in root.findall("link:calculationLink", self.namespaces):
            role_uri = calc_link.get(f"{{{self.XLINK_NAMESPACE}}}role")
            locators = {}
            
            # Build map of labels to concept names
            for loc in calc_link.findall("link:loc", self.namespaces):
                label = loc.get(f"{{{self.XLINK_NAMESPACE}}}label")
                href = loc.get(f"{{{self.XLINK_NAMESPACE}}}href")
                if href:
                    element_name = href.split("#")[-1].replace("us-gaap_", "")
                    locators[label] = element_name
            
            # Group calculation arcs by parent
            formulas_in_role = defaultdict(list)
            for calc_arc in calc_link.findall("link:calculationArc", self.namespaces):
                parent_label = calc_arc.get(f"{{{self.XLINK_NAMESPACE}}}from")
                child_label = calc_arc.get(f"{{{self.XLINK_NAMESPACE}}}to")
                
                parent_element = locators.get(parent_label)
                child_element = locators.get(child_label)
                
                if parent_element and child_element:
                    formulas_in_role[parent_element].append({
                        "child": child_element,
                        "weight": float(calc_arc.get("weight", 1.0)),
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
    
    def parse_presentation_linkbase(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a presentation linkbase XML file.
        
        This extracts presentation relationships (statement structure)
        from XBRL presentation linkbase files.
        
        Args:
            file_path: Path to presentation linkbase XML file
            
        Returns:
            Dictionary with presentation structure
        """
        # Implementation would parse presentation arcs
        # This is a placeholder for future enhancement
        return {}
