# financial4all/xbrl/rendering.py
"""
Rendering utilities for XBRL financial statements.

This module provides functionality for rendering financial statements
with proper formatting, scaling, and display options.
"""

from typing import Any, Dict, List, Optional, Union

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from rich.table import Table as RichTable
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from financial4all.xbrl.core import format_value, determine_dominant_scale, get_currency_symbol


class RenderedStatement:
    """
    A rendered financial statement with multiple output formats.
    """

    def __init__(self, statement_data: List[Dict[str, Any]], 
                 statement_title: str,
                 periods_to_display: List[tuple],
                 scale: Optional[int] = None,
                 currency_symbol: Optional[str] = None):
        """
        Initialize a rendered statement.

        Args:
            statement_data: List of statement line items
            statement_title: Title of the statement
            periods_to_display: List of (period_key, period_label) tuples (most recent first)
            scale: Scale factor (-3 for thousands, -6 for millions, -9 for billions)
            currency_symbol: Currency symbol to use
        """
        self.statement_data = statement_data
        self.statement_title = statement_title
        # periods_to_display should already be sorted newest first from sort_periods
        # Keep it as-is so newest appears on the left when iterating
        self.periods_to_display = periods_to_display
        self.scale = scale or determine_dominant_scale(statement_data, periods_to_display)
        self.currency_symbol = currency_symbol or '$'

    def to_dataframe(self, standard: bool = True) -> 'pd.DataFrame':
        """
        Convert statement to pandas DataFrame.

        Args:
            standard: Whether to use standardized labels

        Returns:
            pandas DataFrame
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for to_dataframe() method")

        rows = []
        for item in self.statement_data:
            row = {
                'label': item.get('label', ''),
                'concept': item.get('concept', ''),
                'level': item.get('level', 0),
            }

            # Add values for each period
            for period_key, period_label in self.periods_to_display:
                value = item.get('values', {}).get(period_key)
                if value is not None:
                    row[period_label] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        """
        Convert statement to markdown format.

        Returns:
            Markdown string
        """
        lines = [f"# {self.statement_title}", ""]

        # Create header
        header = ["Label"]
        for _, period_label in self.periods_to_display:
            header.append(period_label)
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        # Add rows
        for item in self.statement_data:
            if not item.get('has_values', False):
                continue

            label = item.get('label', '')
            level = item.get('level', 0)
            indent = "  " * level
            row = [f"{indent}{label}"]

            for period_key, _ in self.periods_to_display:
                value = item.get('values', {}).get(period_key)
                if value is not None:
                    decimals = item.get('decimals', {}).get(period_key)
                    formatted = format_value(
                        value, True, self.scale, decimals, self.currency_symbol
                    )
                    row.append(formatted)
                else:
                    row.append("—")

            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"RenderedStatement(title={self.statement_title}, items={len(self.statement_data)})"


def render_statement(
    statement_data: List[Dict[str, Any]],
    statement_title: str,
    periods_to_display: List[tuple],
    scale: Optional[int] = None,
    currency_symbol: Optional[str] = None
) -> RenderedStatement:
    """
    Render a financial statement.

    Args:
        statement_data: List of statement line items
        statement_title: Title of the statement
        periods_to_display: List of (period_key, period_label) tuples
        scale: Scale factor (-3 for thousands, -6 for millions, -9 for billions)
        currency_symbol: Currency symbol to use

    Returns:
        RenderedStatement object
    """
    return RenderedStatement(
        statement_data, statement_title, periods_to_display, scale, currency_symbol
    )
