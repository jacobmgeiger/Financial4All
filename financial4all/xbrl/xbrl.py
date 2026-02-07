# financial4all/xbrl/xbrl.py
"""
XBRL Parser - Top-level integration module for XBRL parsing.

This module provides the XBRL class, which integrates all components of the XBRL parsing system:
- Instance Document Parser
- Presentation Linkbase Parser
- Calculation Linkbase Parser
- Definition Linkbase Parser

The XBRL class provides a unified interface for working with XBRL data,
organizing facts according to presentation hierarchies, validating calculations,
and handling dimensional qualifiers.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict

if TYPE_CHECKING:
    from financial4all.xbrl.facts import FactQuery

from financial4all.core import log
from financial4all.xbrl.parser import XBRLParser
from financial4all.xbrl.facts import FactSet, FactsView, Fact
from financial4all.xbrl.models import (
    Context, Fact as ModelFact, ElementCatalog, CalculationNode, CalculationTree
)
from financial4all.xbrl.presentation import PresentationTree, PresentationNode
from financial4all.xbrl.dimensions import Table, Axis, Domain
from financial4all.xbrl.entity_info import EntityInfo, extract_dei_facts, build_entity_info
from financial4all.xbrl.periods import Period, PeriodType
from financial4all.xbrl.period_selector import select_periods
from financial4all.xbrl.statement_resolver import StatementResolver, StatementType
from financial4all.xbrl.deduplication_strategy import RevenueDeduplicator
from financial4all.xbrl.abstract_detection import is_abstract_concept
from financial4all.xbrl.models import select_display_label
from financial4all.xbrl.core import STANDARD_LABEL, TERSE_LABEL, parse_date
from financial4all.xbrl.current_period import CurrentPeriodView
from financial4all.xbrl.currency import CurrencyConverter
from financial4all.xbrl.validation import validate_balance_sheet, ValidationLevel
from financial4all.xbrl.rendering import render_statement, RenderedStatement


class XBRLFilingWithNoXbrlData(Exception):
    """Exception raised when a filing does not contain XBRL data."""
    pass


class XBRL:
    """
    Main XBRL processing class that integrates all components.
    
    This class provides a unified interface for working with XBRL data,
    organizing facts according to presentation hierarchies, validating calculations,
    and handling dimensional qualifiers.
    
    Example:
        >>> from financial4all.xbrl import XBRL
        >>> xbrl = XBRL.from_xml(xml_content)
        >>> balance_sheet = xbrl.get_statement('BalanceSheet')
        >>> facts = xbrl.facts.query().by_concept('Revenue').to_dataframe()
    """
    
    def __init__(self):
        """Initialize XBRL instance."""
        # Core data structures
        self._facts: Dict[str, ModelFact] = {}  # fact_key -> Fact
        self.contexts: Dict[str, Context] = {}  # context_id -> Context
        self.units: Dict[str, Dict[str, Any]] = {}  # unit_id -> unit info
        self.footnotes: Dict[str, Any] = {}  # footnote_id -> Footnote
        
        # Linkbase structures
        self.presentation_trees: Dict[str, PresentationTree] = {}  # role_uri -> PresentationTree
        self.calculation_trees: Dict[str, CalculationTree] = {}  # role_uri -> CalculationTree
        self.definition_tables: Dict[str, List[Table]] = {}  # role_uri -> List[Table]
        
        # Element catalog
        self.element_catalog: Dict[str, ElementCatalog] = {}  # element_id -> ElementCatalog
        
        # Entity information
        self.entity_info: Dict[str, Any] = {}
        self.entity_name: str = ""
        self.document_type: str = ""
        self.period_of_report: Optional[str] = None
        
        # Reporting periods
        self.reporting_periods: List[Dict[str, Any]] = []
        
        # FactSet for backward compatibility
        self.fact_set: Optional[FactSet] = None
        
        # Context-period mapping for fact enrichment
        self.context_period_map: Dict[str, str] = {}  # context_id -> period_key
        
        # Caches
        self._all_statements_cached: Optional[List[Dict[str, Any]]] = None
        self._statement_indices: Dict[str, List[Dict[str, Any]]] = {}
        self._statement_by_standard_name: Dict[str, List[Dict[str, Any]]] = {}
        self._statement_by_role_uri: Dict[str, Dict[str, Any]] = {}
        
        # Parser
        self.parser = XBRLParser()
    
    @classmethod
    def from_xml(cls, xml_content: Union[str, bytes], 
                  presentation_linkbase: Optional[Union[str, Path]] = None,
                  calculation_linkbase: Optional[Union[str, Path]] = None,
                  definition_linkbase: Optional[Union[str, Path]] = None) -> 'XBRL':
        """
        Create XBRL instance from XML content.
        
        Args:
            xml_content: XBRL instance document XML content
            presentation_linkbase: Optional path to presentation linkbase file
            calculation_linkbase: Optional path to calculation linkbase file
            definition_linkbase: Optional path to definition linkbase file
            
        Returns:
            XBRL instance
            
        Raises:
            XBRLFilingWithNoXbrlData: If no XBRL data found
        """
        xbrl = cls()
        
        # Step 1: Parse instance document
        xbrl._parse_instance_document(xml_content)
        
        # Step 2: Parse presentation linkbase
        if presentation_linkbase:
            xbrl.presentation_trees = xbrl.parser.parse_presentation_linkbase(presentation_linkbase)
        
        # Step 3: Parse calculation linkbase
        if calculation_linkbase:
            calc_data = xbrl.parser.parse_calculation_linkbase(calculation_linkbase)
            xbrl._build_calculation_trees(calc_data)
        
        # Step 4: Parse definition linkbase
        if definition_linkbase:
            xbrl.definition_tables = xbrl.parser.parse_definition_linkbase(definition_linkbase)
        
        # Step 5: Build element catalog with abstract detection
        xbrl._build_element_catalog()
        
        # Step 6: Enrich facts with context and element information
        xbrl._enrich_facts()
        
        # Step 7: Build presentation trees with label selection
        xbrl._enhance_presentation_trees()
        
        # Step 8: Build reporting periods
        xbrl._build_reporting_periods()
        
        # Step 9: Extract entity information
        xbrl._extract_entity_info()
        
        return xbrl
    
    @classmethod
    def from_filing(cls, filing: Any) -> 'XBRL':
        """
        Create XBRL instance from a filing object.
        
        Args:
            filing: Filing object with XBRL data
            
        Returns:
            XBRL instance
        """
        # Try to get XML content from filing
        xml_content = None
        
        if hasattr(filing, 'xbrl_content'):
            xml_content = filing.xbrl_content
        elif hasattr(filing, 'get_xbrl_content'):
            xml_content = filing.get_xbrl_content()
        elif hasattr(filing, 'content'):
            xml_content = filing.content
        
        if not xml_content:
            raise XBRLFilingWithNoXbrlData("Filing does not contain XBRL data")
        
        # Try to get linkbase paths
        presentation_linkbase = getattr(filing, 'presentation_linkbase', None)
        calculation_linkbase = getattr(filing, 'calculation_linkbase', None)
        definition_linkbase = getattr(filing, 'definition_linkbase', None)
        
        return cls.from_xml(
            xml_content,
            presentation_linkbase,
            calculation_linkbase,
            definition_linkbase
        )
    
    @classmethod
    def from_company_facts(cls, company_facts: Dict[str, Any], cik: Optional[str] = None) -> 'XBRL':
        """
        Create XBRL instance from SEC company facts API response.
        
        Args:
            company_facts: Dictionary from SEC company facts API
            cik: Optional CIK
            
        Returns:
            XBRL instance
        """
        xbrl = cls()
        
        # Build FactSet from company facts
        fact_set = FactSet.from_company_facts(company_facts, cik)
        xbrl.fact_set = fact_set
        
        # Extract entity info
        xbrl.entity_info = fact_set.entity_info.__dict__ if fact_set.entity_info else {}
        xbrl.entity_name = xbrl.entity_info.get('entity_name', '')
        xbrl.document_type = xbrl.entity_info.get('document_type', '')
        xbrl.period_of_report = xbrl.entity_info.get('document_period_end_date')
        
        # Build reporting periods from facts
        xbrl._build_reporting_periods_from_facts(fact_set)
        
        return xbrl
    
    def _parse_instance_document(self, xml_content: Union[str, bytes]) -> None:
        """Parse XBRL instance document and extract facts, contexts, units."""
        try:
            root = self.parser._safe_parse_xml(xml_content)
        except Exception as e:
            raise XBRLFilingWithNoXbrlData(f"Failed to parse XBRL instance document: {e}") from e
        
        # Extract contexts
        for context_elem in root.findall(".//xbrl:context", self.parser.namespaces):
            context_id = context_elem.get("id")
            if not context_id:
                continue
            
            # Extract entity
            entity_elem = context_elem.find("xbrli:entity", self.parser.namespaces)
            entity = {}
            if entity_elem is not None:
                identifier_elem = entity_elem.find("xbrli:identifier", self.parser.namespaces)
                if identifier_elem is not None:
                    entity['identifier'] = identifier_elem.text
                    entity['scheme'] = identifier_elem.get("scheme", "")
            
            # Extract period
            period_elem = context_elem.find("xbrli:period", self.parser.namespaces)
            period = {}
            if period_elem is not None:
                instant_elem = period_elem.find("xbrli:instant", self.parser.namespaces)
                if instant_elem is not None:
                    period['type'] = 'instant'
                    period['instant'] = instant_elem.text
                else:
                    start_elem = period_elem.find("xbrli:startDate", self.parser.namespaces)
                    end_elem = period_elem.find("xbrli:endDate", self.parser.namespaces)
                    if start_elem is not None and end_elem is not None:
                        period['type'] = 'duration'
                        period['startDate'] = start_elem.text
                        period['endDate'] = end_elem.text
            
            # Extract dimensions
            dimensions = {}
            segment_elem = context_elem.find("xbrli:entity/xbrli:segment", self.parser.namespaces)
            if segment_elem is not None:
                for explicit_member in segment_elem.findall(".//xbrldi:explicitMember", self.parser.namespaces):
                    dimension = explicit_member.get("dimension", "")
                    member = explicit_member.text
                    if dimension and member:
                        dimensions[dimension] = member
            
            self.contexts[context_id] = Context(
                context_id=context_id,
                entity=entity,
                period=period,
                dimensions=dimensions
            )
        
        # Extract units
        for unit_elem in root.findall(".//xbrl:unit", self.parser.namespaces):
            unit_id = unit_elem.get("id")
            if not unit_id:
                continue
            
            measure_elem = unit_elem.find("xbrli:measure", self.parser.namespaces)
            unit_info = {'type': 'simple', 'measure': ''}
            if measure_elem is not None:
                unit_info['measure'] = measure_elem.text or ''
            
            self.units[unit_id] = unit_info
        
        # Extract facts
        fact_counter = defaultdict(int)  # Track duplicates
        
        for elem in root:
            # Skip non-fact elements (contexts, units, etc.)
            if elem.tag.endswith('}context') or elem.tag.endswith('}unit'):
                continue
            
            # Extract fact information
            element_id = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            context_ref = elem.get("contextRef", "")
            unit_ref = elem.get("unitRef", "")
            decimals = elem.get("decimals")
            fact_id = elem.get("id")
            
            if not context_ref:
                continue
            
            # Handle namespace prefixes
            if ':' in element_id:
                prefix, name = element_id.split(':', 1)
                element_id = f"{prefix}_{name}"
            
            # Create fact key and handle duplicates
            fact_key_base = f"{element_id}_{context_ref}"
            instance_id = fact_counter[fact_key_base]
            fact_counter[fact_key_base] += 1
            
            fact_key = f"{fact_key_base}_{instance_id}" if instance_id > 0 else fact_key_base
            
            # Extract value
            value = elem.text or ""
            numeric_value = None
            try:
                numeric_value = float(value) if value else None
            except (ValueError, TypeError):
                pass
            
            # Parse decimals
            decimals_value = None
            if decimals:
                if decimals == "INF":
                    decimals_value = "INF"
                else:
                    try:
                        decimals_value = int(decimals)
                    except (ValueError, TypeError):
                        pass
            
            fact = ModelFact(
                element_id=element_id,
                context_ref=context_ref,
                value=value,
                unit_ref=unit_ref,
                decimals=decimals_value,
                numeric_value=numeric_value,
                instance_id=instance_id if instance_id > 0 else None,
                fact_id=fact_id
            )
            
            self._facts[fact_key] = fact
    
    def _build_calculation_trees(self, calc_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Build calculation trees from calculation linkbase data."""
        # Group by role
        by_role = defaultdict(lambda: {'nodes': {}, 'roles': set()})
        
        for parent, formulas in calc_data.items():
            for formula in formulas:
                role_uri = formula.get('role', '')
                children = formula.get('children', [])
                
                if role_uri not in by_role:
                    by_role[role_uri] = {'nodes': {}, 'roles': set()}
                
                # Create or update calculation node
                if parent not in by_role[role_uri]['nodes']:
                    by_role[role_uri]['nodes'][parent] = CalculationNode(
                        element_id=parent,
                        children=[],
                        weight=1.0
                    )
                
                node = by_role[role_uri]['nodes'][parent]
                
                # Add children
                for child_info in children:
                    child_id = child_info.get('child')
                    weight = child_info.get('weight', 1.0)
                    
                    if child_id not in node.children:
                        node.children.append(child_id)
                    
                    # Create child node if not exists
                    if child_id not in by_role[role_uri]['nodes']:
                        by_role[role_uri]['nodes'][child_id] = CalculationNode(
                            element_id=child_id,
                            parent=parent,
                            weight=weight
                        )
                    else:
                        # Update parent reference
                        by_role[role_uri]['nodes'][child_id].parent = parent
                        by_role[role_uri]['nodes'][child_id].weight = weight
        
        # Build calculation trees
        for role_uri, data in by_role.items():
            # Find root (node without parent)
            root_id = None
            for element_id, node in data['nodes'].items():
                if node.parent is None:
                    root_id = element_id
                    break
            
            if not root_id and data['nodes']:
                root_id = list(data['nodes'].keys())[0]
            
            if root_id:
                self.calculation_trees[role_uri] = CalculationTree(
                    role_uri=role_uri,
                    definition=role_uri.split('/')[-1] if '/' in role_uri else role_uri,
                    root_element_id=root_id,
                    all_nodes=data['nodes']
                )
    
    def _build_element_catalog(self) -> None:
        """Build element catalog with abstract detection."""
        # Collect all unique elements from facts and presentation trees
        all_elements = set()
        
        # From facts
        for fact in self._facts.values():
            all_elements.add(fact.element_id)
        
        # From presentation trees
        for tree in self.presentation_trees.values():
            for node in tree.all_nodes.values():
                all_elements.add(node.element_id)
        
        # Build catalog entries
        for element_id in all_elements:
            # Determine if abstract
            has_children = False
            has_values = element_id in [f.element_id for f in self._facts.values()]
            
            # Check presentation trees for children
            for tree in self.presentation_trees.values():
                if element_id in tree.all_nodes:
                    node = tree.all_nodes[element_id]
                    if node.children:
                        has_children = True
                        break
            
            # Use abstract detection
            abstract = is_abstract_concept(
                element_id,
                schema_abstract=False,  # We don't have schema info
                has_children=has_children,
                has_values=has_values
            )
            
            # Create element catalog entry
            self.element_catalog[element_id] = ElementCatalog(
                name=element_id,
                data_type='monetary',  # Default, could be enhanced
                period_type='duration',  # Default, could be enhanced
                abstract=abstract
            )
    
    def _enrich_facts(self) -> None:
        """Enrich facts with context and element information."""
        # Build context-period mapping
        for context_id, context in self.contexts.items():
            period = context.period
            if period.get('type') == 'instant':
                period_key = f"instant_{period.get('instant', '')}"
            elif period.get('type') == 'duration':
                period_key = f"duration_{period.get('startDate', '')}_{period.get('endDate', '')}"
            else:
                continue
            
            self.context_period_map[context_id] = period_key
    
    def _enhance_presentation_trees(self) -> None:
        """Enhance presentation trees with label selection and abstract detection."""
        for role_uri, tree in self.presentation_trees.items():
            for element_id, node in tree.all_nodes.items():
                # Enhance node with element catalog info
                if element_id in self.element_catalog:
                    element = self.element_catalog[element_id]
                    node.is_abstract = element.abstract
                    node.element_name = element.name
                
                # Use select_display_label for label selection
                if not node.standard_label and node.labels:
                    node.standard_label = select_display_label(
                        node.labels,
                        preferred_label=node.preferred_label,
                        element_id=element_id,
                        element_name=node.element_name
                    )
    
    def _build_reporting_periods(self) -> None:
        """Build reporting periods from contexts."""
        periods_dict = {}
        
        for context_id, context in self.contexts.items():
            period = context.period
            period_type = period.get('type')
            
            if period_type == 'instant':
                date_str = period.get('instant', '')
                if date_str:
                    period_key = f"instant_{date_str}"
                    if period_key not in periods_dict:
                        periods_dict[period_key] = {
                            'key': period_key,
                            'type': 'instant',
                            'date': date_str,
                            'label': date_str
                        }
            
            elif period_type == 'duration':
                start_str = period.get('startDate', '')
                end_str = period.get('endDate', '')
                if start_str and end_str:
                    period_key = f"duration_{start_str}_{end_str}"
                    if period_key not in periods_dict:
                        periods_dict[period_key] = {
                            'key': period_key,
                            'type': 'duration',
                            'start_date': start_str,
                            'end_date': end_str,
                            'label': f"{start_str} to {end_str}"
                        }
        
        self.reporting_periods = sorted(
            periods_dict.values(),
            key=lambda p: p.get('end_date', p.get('date', '')),
            reverse=True
        )
    
    def _build_reporting_periods_from_facts(self, fact_set: FactSet) -> None:
        """Build reporting periods from FactSet."""
        periods_dict = {}
        
        for fact in fact_set.facts:
            period = fact.period
            if period.period_type == PeriodType.INSTANT:
                date_str = str(period.end)
                period_key = f"instant_{date_str}"
                if period_key not in periods_dict:
                    periods_dict[period_key] = {
                        'key': period_key,
                        'type': 'instant',
                        'date': date_str,
                        'label': date_str
                    }
            elif period.period_type == PeriodType.DURATION:
                start_str = str(period.start) if period.start else ''
                end_str = str(period.end)
                period_key = f"duration_{start_str}_{end_str}"
                if period_key not in periods_dict:
                    periods_dict[period_key] = {
                        'key': period_key,
                        'type': 'duration',
                        'start_date': start_str,
                        'end_date': end_str,
                        'label': f"{start_str} to {end_str}" if start_str else end_str
                    }
        
        self.reporting_periods = sorted(
            periods_dict.values(),
            key=lambda p: p.get('end_date', p.get('date', '')),
            reverse=True
        )
    
    def _extract_entity_info(self) -> None:
        """Extract entity information from facts or contexts."""
        # Try to get entity info from fact_set if available
        if self.fact_set and self.fact_set.entity_info:
            self.entity_info = self.fact_set.entity_info.__dict__
            self.entity_name = self.entity_info.get('entity_name', '')
            self.document_type = self.entity_info.get('document_type', '')
            self.period_of_report = self.entity_info.get('document_period_end_date')
    
    @property
    def facts(self) -> FactsView:
        """Get FactsView for querying facts."""
        return FactsView(self)
    
    @property
    def current_period(self) -> CurrentPeriodView:
        """Get CurrentPeriodView for current period access."""
        return CurrentPeriodView(self)
    
    def query(self, include_dimensions: bool = False,
              include_contexts: bool = False,
              include_element_info: bool = False) -> 'FactQuery':
        """
        Start a new query for XBRL facts.
        
        Args:
            include_dimensions: Whether to include dimensions in results
            include_contexts: Whether to include context information
            include_element_info: Whether to include element catalog information
            
        Returns:
            FactQuery instance
        """
        fact_query = self.facts.query()
        fact_query._include_dimensions = include_dimensions
        if not include_contexts:
            fact_query = fact_query.exclude_contexts()
        if not include_element_info:
            fact_query = fact_query.exclude_element_info()
        return fact_query
    
    def get_all_statements(self) -> List[Dict[str, Any]]:
        """
        Get all available financial statements.
        
        Returns:
            List of statement metadata (role, definition, element count)
        """
        if self._all_statements_cached is not None:
            return self._all_statements_cached
        
        resolver = StatementResolver(self.presentation_trees)
        statements = resolver.find_statements()
        
        self._all_statements_cached = statements
        return statements
    
    def find_statement(self, role_or_type: str) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
        """
        Find statement by role URI, statement type, or statement short name.
        
        Args:
            role_or_type: Can be role URI, statement type, or short name
            
        Returns:
            Tuple of (matching_statements, found_role, actual_statement_type)
        """
        resolver = StatementResolver(self.presentation_trees)
        
        # Try direct role match
        if role_or_type in self.presentation_trees:
            stmt = resolver.get_statement_by_role(role_or_type)
            if stmt:
                return [stmt], role_or_type, stmt.get('statement_type')
        
        # Try statement type match
        stmt = resolver.get_statement_by_type(role_or_type)
        if stmt:
            return [stmt], stmt.get('role_uri'), stmt.get('statement_type')
        
        # Try partial match on role definition
        all_statements = resolver.find_statements()
        matching = []
        role_lower = role_or_type.lower()
        
        for stmt in all_statements:
            definition = stmt.get('definition', '').lower()
            role_name = stmt.get('role_uri', '').lower()
            
            if role_lower in definition or role_lower in role_name:
                matching.append(stmt)
        
        if matching:
            return matching, matching[0].get('role_uri'), matching[0].get('statement_type')
        
        return [], None, None
    
    def get_statement(self, role_or_type: str,
                     period_filter: Optional[str] = None,
                     should_display_dimensions: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Get a financial statement by role URI, statement type, or statement short name.
        
        Args:
            role_or_type: Can be role URI, statement type, or short name
            period_filter: Optional period key to filter facts
            should_display_dimensions: Whether to display dimensions
            
        Returns:
            List of line items with values
        """
        matching_statements, found_role, actual_statement_type = self.find_statement(role_or_type)
        
        if not found_role or found_role not in self.presentation_trees:
            return []
        
        tree = self.presentation_trees[found_role]
        root_id = tree.root_element_id
        
        if should_display_dimensions is None:
            should_display_dimensions = True
        
        # Generate line items (simplified - full implementation would be more complex)
        line_items = []
        self._generate_line_items(root_id, tree.all_nodes, line_items, period_filter, None, should_display_dimensions)
        
        # Apply deduplication for income statements
        if actual_statement_type == 'IncomeStatement':
            line_items = RevenueDeduplicator.deduplicate_statement_items(line_items)
        
        return line_items
    
    def _generate_line_items(self, element_id: str, nodes: Dict[str, PresentationNode],
                            result: List[Dict[str, Any]], period_filter: Optional[str] = None,
                            path: Optional[List[str]] = None, should_display_dimensions: bool = False) -> None:
        """Recursively generate line items for a statement."""
        if element_id not in nodes:
            return
        
        if path is None:
            path = []
        
        current_path = path + [element_id]
        node = nodes[element_id]
        
        # Get label
        label = node.standard_label or node.element_name or element_id
        
        # Get values from facts
        values = {}
        decimals = {}
        units = {}
        
        # Find facts for this element
        matching_facts = [f for f in self._facts.values() if f.element_id == element_id]
        
        for fact in matching_facts:
            context_id = fact.context_ref
            period_key = self.context_period_map.get(context_id)
            
            if period_filter and period_key != period_filter:
                continue
            
            if period_key:
                values[period_key] = fact.numeric_value if fact.numeric_value is not None else fact.value
                if fact.decimals:
                    decimals[period_key] = fact.decimals
                if fact.unit_ref:
                    units[period_key] = fact.unit_ref
        
        # Create line item
        line_item = {
            'concept': element_id,
            'name': node.element_name or element_id,
            'label': label,
            'values': values,
            'decimals': decimals,
            'units': units,
            'level': node.depth,
            'is_abstract': node.is_abstract,
            'has_values': len(values) > 0,
            'children': node.children
        }
        
        result.append(line_item)
        
        # Process children
        for child_id in node.children:
            self._generate_line_items(child_id, nodes, result, period_filter, current_path, should_display_dimensions)
    
    def render_statement(self, statement_type: str, period_view: Optional[str] = None) -> RenderedStatement:
        """
        Render a statement with formatting.
        
        Args:
            statement_type: Type of statement to render
            period_view: Optional period view name
            
        Returns:
            RenderedStatement object
        """
        from financial4all.xbrl.periods import determine_periods_to_display
        
        # Get statement data
        statement_data = self.get_statement(statement_type)
        
        # Determine periods to display
        periods_to_display = determine_periods_to_display(self, statement_type, period_view=period_view)
        
        # Render
        statement_title = f"{statement_type.replace('_', ' ').title()}"
        return render_statement(statement_data, statement_title, periods_to_display)
    
    def validate(self, statement_type: str = 'BalanceSheet', level: ValidationLevel = ValidationLevel.FUNDAMENTAL) -> Any:
        """
        Validate a statement.
        
        Args:
            statement_type: Type of statement to validate
            level: Validation level
            
        Returns:
            ValidationResult
        """
        statement_data = self.get_statement(statement_type)
        
        if statement_type == 'BalanceSheet':
            # Convert to DataFrame for validation
            try:
                import pandas as pd
                df = pd.DataFrame(statement_data)
                return validate_balance_sheet(df, level=level)
            except ImportError:
                log.warning("pandas required for validation")
                return None
        
        return None
    
    def __repr__(self) -> str:
        """String representation."""
        return f"XBRL(entity={self.entity_name}, facts={len(self._facts)}, statements={len(self.presentation_trees)})"
