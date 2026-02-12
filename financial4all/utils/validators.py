# financial4all/utils/validators.py
"""
Validation helpers for tickers and CIKs.

Used by Company and SEC client to ensure ticker format (1–5 letters) and CIK
format (10 digits, optionally zero-padded). normalize_cik() strips non-digits
and left-pads to 10 characters for SEC API URLs.
"""

import re
from typing import Optional


def validate_ticker(ticker: str) -> bool:
    """
    Return True if the string is a valid ticker (1–5 letters, case-insensitive).

    Args:
        ticker: Candidate ticker symbol.

    Returns:
        True if valid, False if empty, not a string, or doesn't match [A-Z]{1,5}.
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Tickers are typically 1-5 uppercase letters
    pattern = r'^[A-Z]{1,5}$'
    return bool(re.match(pattern, ticker.upper()))


def validate_cik(cik: str) -> bool:
    """
    Return True if the value is a valid 10-digit CIK (digits only, no leading zeros required).

    Args:
        cik: CIK as string or int (will be stringified).

    Returns:
        True if exactly 10 digits after conversion, False otherwise.
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
    Normalize a CIK to a 10-character string of digits (leading zeros as needed).

    Strips non-digits, then left-pads with zeros. Use for SEC API paths and
    consistency (e.g. CIK 1045810 -> "0001045810").

    Args:
        cik: CIK in any common form (e.g. 1045810, "1045810", "0001045810").

    Returns:
        Exactly 10-digit string.
    """
    cik_str = str(cik).strip()
    # Remove any non-digit characters
    cik_str = re.sub(r'\D', '', cik_str)
    # Pad with leading zeros to 10 digits
    return cik_str.zfill(10)
