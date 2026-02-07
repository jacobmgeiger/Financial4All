# financial4all/utils/formatters.py
"""
Data formatting utilities.

This module provides formatting functions for currency, numbers, and other data types.
"""

from typing import Union, Optional


def format_currency(value: Union[int, float], currency: str = "USD") -> str:
    """
    Format a numeric value as currency.
    
    Args:
        value: Numeric value to format
        currency: Currency code (default: "USD")
        
    Returns:
        Formatted currency string
    """
    if value is None:
        return "N/A"
    
    if currency == "USD":
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"


def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """
    Format a numeric value with specified decimal places.
    
    Args:
        value: Numeric value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if value is None:
        return "N/A"
    
    return f"{value:,.{decimals}f}"


def format_large_number(value: Union[int, float]) -> str:
    """
    Format a large number with appropriate suffix (K, M, B, T).
    
    Args:
        value: Numeric value to format
        
    Returns:
        Formatted number string with suffix
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
