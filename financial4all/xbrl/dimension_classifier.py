# financial4all/xbrl/dimension_classifier.py
"""
XBRL dimension classification: face vs breakdown.

EdgarTools-inspired model:
- Face dimensions: Shown on face of statement (e.g., ProductOrServiceAxis for product lines).
- Breakdown dimensions: Notes disclosure (e.g., StatementGeographicalAxis, segment by geography).
Both types add segment rows in DETAILED view; FACE only in STANDARD.
"""

import re
from typing import Set, List

# -----------------------------------------------------------------------------
# EXPLICIT AXIS LISTS (EdgarTools-aligned)
# -----------------------------------------------------------------------------

# Axes shown on face of statement (product lines, contract types, etc.)
# These surface segment rows in both STANDARD and DETAILED views per EdgarTools.
FACE_AXES: Set[str] = {
    "ProductOrServiceAxis",
    "srt:ProductOrServiceAxis",
    "RelatedPartyTransactionsByRelatedPartyAxis",
    "PropertyPlantAndEquipmentByTypeAxis",
    "LongtermDebtTypeAxis",
    "ShortTermDebtTypeAxis",
    "StatementClassOfStockAxis",
    "ContracttypeAxis",
    "MajorProgramsAxis",
}

# Axes that are breakdown/notes disclosure (geography, business segments, etc.)
# These surface segment rows in DETAILED view only.
BREAKDOWN_AXES: Set[str] = {
    "StatementGeographicalAxis",
    "srt:StatementGeographicalAxis",
    "GeographicDistributionAxis",
    "StatementBusinessSegmentsAxis",
    "srt:StatementBusinessSegmentsAxis",
    "ConsolidationItemsAxis",
    "srt:ConsolidatedEntitiesAxis",
    "ConsolidatedEntitiesAxis",
    "SegmentReportingGeographicAxis",
    "BusinessAcquisitionAxis",
    "LegalEntityAxis",
    "dei:LegalEntityAxis",
    "ReportingUnitAxis",
}

# Regex patterns for breakdown dimensions (notes disclosure)
BREAKDOWN_PATTERNS: List[str] = [
    r"FairValue.*Axis",
    r".*HierarchyLevelAxis",
    r".*SegmentAxis",
    r".*Geographic.*Axis",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in BREAKDOWN_PATTERNS]


def _normalize_axis(dim_key: str) -> str:
    """Extract axis name for lookup (strip namespace prefix)."""
    if ":" in dim_key:
        return dim_key.split(":", 1)[1]
    return dim_key


def _matches_breakdown_pattern(axis_name: str) -> bool:
    """Return True if axis matches any breakdown pattern."""
    return any(p.search(axis_name) for p in _COMPILED)


def is_breakdown_dimension(dim_key: str) -> bool:
    """
    Return True if the dimension adds segment/breakdown rows (face or notes disclosure).

    For Income Statement, both ProductOrServiceAxis (face) and
    StatementBusinessSegmentsAxis (breakdown) surface segment rows in DETAILED view.
    Checks FACE_AXES first, then BREAKDOWN_AXES, then patterns.

    Args:
        dim_key: Dimension axis name (e.g., ProductOrServiceAxis,
                 StatementBusinessSegmentsAxis). May include namespace prefix.

    Returns:
        True if dimension adds segment rows; False for structural (e.g. equity) only.

    Example:
        >>> is_breakdown_dimension("StatementBusinessSegmentsAxis")
        True
        >>> is_breakdown_dimension("ProductOrServiceAxis")
        True
        >>> is_breakdown_dimension("StatementEquityComponentsAxis")
        False
    """
    axis = _normalize_axis(dim_key)
    # 1. FACE axes: product/service, etc. – add segment rows
    if axis in FACE_AXES or dim_key in FACE_AXES:
        return True
    # 2. BREAKDOWN axes: geography, segments, etc.
    if axis in BREAKDOWN_AXES or dim_key in BREAKDOWN_AXES:
        return True
    # 3. Pattern fallback (segment, geographic, product, region)
    local_lower = axis.lower()
    if any(p in local_lower for p in ("segment", "geographic", "product", "region", "geography")):
        return True
    if _matches_breakdown_pattern(axis):
        return True
    return False


def is_face_dimension(dim_key: str) -> bool:
    """
    Return True if the dimension is a face (classification) dimension.

    Face dimensions are shown on the statement face (e.g., ProductOrServiceAxis).
    Structural axes like equity components also qualify.

    Args:
        dim_key: Dimension axis name.

    Returns:
        True if dimension is a face type; False otherwise.
    """
    axis = _normalize_axis(dim_key)
    if axis in FACE_AXES or dim_key in FACE_AXES:
        return True
    local_lower = axis.lower()
    face_patterns = ("equity", "comprehensiveincome", "statement")
    return any(p in local_lower for p in face_patterns)
