# financial4all/utils/formatters.py
"""
Display formatting for numbers and currency.

Used by UI and export layers to show amounts (e.g. $1,234.56), plain numbers
with decimals, and compact large numbers with K/M/B/T suffixes. None inputs
return "N/A".
"""

from typing import Union, Optional


def format_currency(value: Union[int, float], currency: str = "USD") -> str:
    """
    Format a numeric value as currency with thousands separators.

    Args:
        value: Amount to format (None -> "N/A").
        currency: Currency code; "USD" yields "$X,XXX.XX", others "X,XXX.XX CODE".

    Returns:
        Formatted string (e.g. "$1,234.56" or "1,234.56 EUR").
    """
    if value is None:
        return "N/A"
    
    if currency == "USD":
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"


def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """
    Format a number with thousands separators and fixed decimal places.

    Args:
        value: Number to format (None -> "N/A").
        decimals: Number of decimal places.

    Returns:
        Formatted string (e.g. "1,234.56").
    """
    if value is None:
        return "N/A"
    
    return f"{value:,.{decimals}f}"


def format_large_number(value: Union[int, float]) -> str:
    """
    Format a number with K/M/B/T suffix (e.g. 1.23M, 4.56B).

    Uses 2 decimal places. Preserves sign. None returns "N/A".

    Args:
        value: Number to format.

    Returns:
        String like "1.23M", "-4.56B", or "N/A".
    """
    if value is None:
        return "N/A"
    
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_value >= 1e12:
        return f"{sign}{abs_value/1e12:.2f}T"
    elif abs_value >= 1e9:
        return f"{sign}{abs_value/1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"{sign}{abs_value/1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"{sign}{abs_value/1e3:.2f}K"
    else:
        return f"{sign}{abs_value:.2f}"
