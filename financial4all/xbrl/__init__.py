# financial4all/xbrl/__init__.py
"""
XBRL parsing and processing module.

This module provides functionality for parsing XBRL documents, extracting facts,
resolving financial statements, and handling XBRL-specific features like
dimensions, periods, and calculations.
"""

from financial4all.xbrl.parser import XBRLParser
from financial4all.xbrl.facts import Fact, FactSet
from financial4all.xbrl.statements import StatementResolver
from financial4all.xbrl.periods import Period, PeriodType, classify_fiscal_period, classify_duration
from financial4all.xbrl.entity_info import EntityInfo, extract_dei_facts, build_entity_info
from financial4all.xbrl.presentation import PresentationTree, PresentationNode
from financial4all.xbrl.dimensions import Table, Axis, Domain
from financial4all.xbrl.standardization import (
    SynonymGroup,
    SynonymGroups,
    ConceptInfo,
    get_synonym_groups,
    StandardizationStore,
    get_default_store,
)

__all__ = [
    "XBRLParser",
    "Fact",
    "FactSet",
    "StatementResolver",
    "Period",
    "PeriodType",
    "classify_fiscal_period",
    "classify_duration",
    "EntityInfo",
    "extract_dei_facts",
    "build_entity_info",
    "PresentationTree",
    "PresentationNode",
    "Table",
    "Axis",
    "Domain",
    "SynonymGroup",
    "SynonymGroups",
    "ConceptInfo",
    "get_synonym_groups",
    "StandardizationStore",
    "get_default_store",
]
