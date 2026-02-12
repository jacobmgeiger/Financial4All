# financial4all/xbrl/fact_resolution.py
"""
EdgarTools-aligned duplicate fact resolution for XBRL.

When multiple facts exist for the same (concept, period), this module provides
a single, documented resolution order to select the best fact.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

# Resolution priority (lower sort key = higher priority):
# 1. Non-dimensional (consolidated) over dimensional (segment-specific)
# 2. Form: 10-K > 10-K/A > 10-Q > other
# 3. Concept priority (from standard mapping order; passed as concept_idx)
# 4. Unit match (USD for USD metrics, shares for shares)
# 5. Filing date: more recent preferred


def fact_resolution_sort_key(
    item: Tuple[int, Any],
    is_valid_unit_fn: Callable[[str, str], bool],
    std_name: str,
    exclude_amended: bool = False,
) -> Tuple[int, int, int, int, float]:
    """
    Build sort key for fact resolution (EdgarTools-aligned priority).

    Priority order (ascending - lower wins):
    1. has_dimensions: 0 = no dimensions (prefer), 1 = has dimensions
    2. concept_idx: lower = higher concept priority from mapping
    3. form_priority: 0 = 10-K, 1 = 10-K/A, 2 = 10-Q, 3 = other (when exclude_amended,
       use 99 for 10-K/A to deprioritize)
    4. unit_valid: 0 = valid, 1 = invalid
    5. -filed_ts: negated so more recent = lower = better

    Args:
        item: Tuple of (concept_idx, fact)
        is_valid_unit_fn: Function(unit, std_name) -> bool
        std_name: Standard metric name (for unit check)
        exclude_amended: If True, strongly deprioritize 10-K/A when 10-K exists

    Returns:
        Sort key tuple for sorted()
    """
    concept_idx, fact = item
    form = getattr(fact, "form", None) or ""
    unit_valid = 1 - int(is_valid_unit_fn(fact.unit, std_name)) if hasattr(fact, "unit") else 0
    has_dims = 1 if (getattr(fact, "dimensions", None) or {}) else 0

    # Form priority: 10-K=0, 10-K/A=1, 10-Q=2, other=3
    if form == "10-K":
        form_priority = 0
    elif form == "10-K/A":
        form_priority = 99 if exclude_amended else 1
    elif form and "10-Q" in form:
        form_priority = 2
    else:
        form_priority = 3

    filed = getattr(fact, "filed", None)
    filed_ts = float(filed.timestamp()) if filed else float("-inf")

    return (has_dims, concept_idx, form_priority, unit_valid, -filed_ts)


def sort_fact_candidates_by_priority(
    candidates: List[Tuple[int, Any]],
    is_valid_unit_fn: Callable[[str, str], bool],
    std_name: str,
    exclude_amended: bool = False,
) -> List[Tuple[int, Any]]:
    """
    Sort fact candidates by EdgarTools-aligned resolution priority.

    Use with period_facts_map: for each period, pass the list of (concept_idx, fact)
    tuples. Returns sorted list (best first).

    Args:
        candidates: List of (concept_idx, fact) tuples
        is_valid_unit_fn: Function(unit, std_name) -> bool
        std_name: Standard metric name
        exclude_amended: If True, deprioritize 10-K/A facts

    Returns:
        Sorted list of (concept_idx, fact) with best candidate first
    """
    return sorted(
        candidates,
        key=lambda x: fact_resolution_sort_key(x, is_valid_unit_fn, std_name, exclude_amended),
    )
