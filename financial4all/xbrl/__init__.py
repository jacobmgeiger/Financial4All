# financial4all/xbrl/__init__.py
"""
XBRL parsing and processing module.

This module provides functionality for parsing XBRL documents, extracting facts,
resolving financial statements, and handling XBRL-specific features like
dimensions, periods, and calculations.
"""

from financial4all.xbrl.parser import XBRLParser
from financial4all.xbrl.facts import Fact, FactSet, FactsView, FactQuery
from financial4all.xbrl.statements import StatementResolver as BaseStatementResolver
from financial4all.xbrl.statement_resolver import StatementResolver, StatementType
from financial4all.xbrl.periods import (
    Period, PeriodType, classify_fiscal_period, classify_duration,
    calculate_fiscal_alignment_score, filter_periods_by_document_end_date, sort_periods,
    get_period_views, determine_periods_to_display
)
from financial4all.xbrl.period_selector import select_periods
from financial4all.xbrl.entity_info import EntityInfo, extract_dei_facts, build_entity_info
from financial4all.xbrl.presentation import PresentationTree, PresentationNode
from financial4all.xbrl.dimensions import Table, Axis, Domain
# Import from parent standardization.py module (not the standardization package)
from financial4all.xbrl import standardization
SynonymGroup = standardization.SynonymGroup
SynonymGroups = standardization.SynonymGroups
ConceptInfo = standardization.ConceptInfo
get_synonym_groups = standardization.get_synonym_groups
StandardizationStore = standardization.StandardizationStore
get_default_store = standardization.get_default_store

# Import new standardization package features
from financial4all.xbrl.standardization import (
    StandardConcept,
    ReverseIndex, MappingResult, get_reverse_index,
    SectionMembership, get_section_membership,
    UnmappedTagLogger, get_unmapped_logger,
    MappingStore, ConceptMapper,
    should_exclude, EXCLUDED_TAGS,
)
from financial4all.xbrl.core import (
    parse_date, format_date, format_value, determine_dominant_scale,
    get_currency_symbol, get_unit_display_name, is_point_in_time,
    STANDARD_LABEL, TERSE_LABEL, PERIOD_START_LABEL, PERIOD_END_LABEL, TOTAL_LABEL,
    NAMESPACES
)
from financial4all.xbrl.models import (
    ElementCatalog, Context, Fact as ModelFact, Footnote,
    CalculationNode, CalculationTree, XBRLProcessingError, select_display_label
)
from financial4all.xbrl.abstract_detection import (
    is_abstract_concept, add_known_abstract_concept, add_abstract_pattern,
    get_known_abstract_concepts, get_abstract_patterns
)
from financial4all.xbrl.validation import (
    ValidationLevel, ValidationSeverity, ValidationIssue, ValidationResult,
    validate_balance_sheet, validate_statement
)
from financial4all.xbrl.currency import CurrencyConverter, ExchangeRate
from financial4all.xbrl.deduplication_strategy import RevenueDeduplicator
from financial4all.xbrl.rendering import RenderedStatement, render_statement
from financial4all.xbrl.xbrl import XBRL, XBRLFilingWithNoXbrlData
from financial4all.xbrl.current_period import CurrentPeriodView
from financial4all.xbrl.period_data_check import (
    check_period_data, validate_period_ranges, detect_period_anomalies, validate_fiscal_alignment
)
from financial4all.xbrl.stitching import (
    StatementStitcher, XBRLS, StitchedStatements, stitch_statements,
    render_stitched_statement, to_pandas as stitch_to_pandas
)

__all__ = [
    # Core parsing
    "XBRLParser",
    
    # Facts
    "Fact",
    "FactSet",
    "FactsView",
    "FactQuery",
    
    # Statements
    "StatementResolver",
    "BaseStatementResolver",
    "StatementType",
    
    # Periods
    "Period",
    "PeriodType",
    "classify_fiscal_period",
    "classify_duration",
    "calculate_fiscal_alignment_score",
    "filter_periods_by_document_end_date",
    "sort_periods",
    "select_periods",
    "get_period_views",
    "determine_periods_to_display",
    
    # Entity info
    "EntityInfo",
    "extract_dei_facts",
    "build_entity_info",
    
    # Presentation
    "PresentationTree",
    "PresentationNode",
    
    # Dimensions
    "Table",
    "Axis",
    "Domain",
    
    # Standardization (parent module)
    "SynonymGroup",
    "SynonymGroups",
    "ConceptInfo",
    "get_synonym_groups",
    "StandardizationStore",
    "get_default_store",
    
    # Standardization (new package features)
    "StandardConcept",
    "ReverseIndex",
    "MappingResult",
    "get_reverse_index",
    "SectionMembership",
    "get_section_membership",
    "UnmappedTagLogger",
    "get_unmapped_logger",
    "MappingStore",
    "ConceptMapper",
    "should_exclude",
    "EXCLUDED_TAGS",
    
    # Core utilities
    "parse_date",
    "format_date",
    "format_value",
    "determine_dominant_scale",
    "get_currency_symbol",
    "get_unit_display_name",
    "is_point_in_time",
    "STANDARD_LABEL",
    "TERSE_LABEL",
    "PERIOD_START_LABEL",
    "PERIOD_END_LABEL",
    "TOTAL_LABEL",
    "NAMESPACES",
    
    # Models
    "ElementCatalog",
    "Context",
    "ModelFact",
    "Footnote",
    "CalculationNode",
    "CalculationTree",
    "XBRLProcessingError",
    "select_display_label",
    
    # Abstract detection
    "is_abstract_concept",
    "add_known_abstract_concept",
    "add_abstract_pattern",
    "get_known_abstract_concepts",
    "get_abstract_patterns",
    
    # Validation
    "ValidationLevel",
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "validate_balance_sheet",
    "validate_statement",
    
    # Currency
    "CurrencyConverter",
    "ExchangeRate",
    
    # Deduplication
    "RevenueDeduplicator",
    
    # Rendering
    "RenderedStatement",
    "render_statement",
    
    # Main XBRL class
    "XBRL",
    "XBRLFilingWithNoXbrlData",
    
    # Current period
    "CurrentPeriodView",
    
    # Period data check
    "check_period_data",
    "validate_period_ranges",
    "detect_period_anomalies",
    "validate_fiscal_alignment",
    
    # Stitching
    "StatementStitcher",
    "XBRLS",
    "StitchedStatements",
    "stitch_statements",
    "render_stitched_statement",
    "stitch_to_pandas",
]
