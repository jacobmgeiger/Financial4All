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
from financial4all.xbrl.periods import Period, PeriodType

__all__ = [
    "XBRLParser",
    "Fact",
    "FactSet",
    "StatementResolver",
    "Period",
    "PeriodType",
]
