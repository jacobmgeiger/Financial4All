# financial4all/utils/validators.py
"""
Data validation utilities.

This module provides validation functions for tickers, CIKs, and other data types.
"""

import re
from typing import Optional


def validate_ticker(ticker: str) -> bool:
    """
    Validate a stock ticker symbol.
    
    Args:
        ticker: Ticker symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Tickers are typically 1-5 uppercase letters
    pattern = r'^[A-Z]{1,5}$'
    return bool(re.match(pattern, ticker.upper()))


def validate_cik(cik: str) -> bool:
    """
    Validate a CIK (Central Index Key).
    
    Args:
        cik: CIK to validate (can be string or int)
        
    Returns:
        True if valid, False otherwise
    """
    if cik is None:
        return False
    
    # Convert to string if needed
    cik_str = str(cik).strip()
    
    # CIKs are 10-digit numbers (may have leading zeros)
    pattern = r'^\d{10}$'
    return bool(re.match(pattern, cik_str))


def normalize_cik(cik: str) -> str:
    """
    Normalize a CIK to 10-digit format with leading zeros.
    
    Args:
        cik: CIK to normalize
        
    Returns:
        Normalized CIK as 10-digit string
    """
    cik_str = str(cik).strip()
    # Remove any non-digit characters
    cik_str = re.sub(r'\D', '', cik_str)
    # Pad with leading zeros to 10 digits
    return cik_str.zfill(10)
