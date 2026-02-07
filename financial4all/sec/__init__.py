# financial4all/sec/__init__.py
"""
SEC EDGAR API integration module.

This module provides functionality for interacting with the SEC EDGAR database,
including company lookups, filing retrieval, and API client management.
"""

from financial4all.sec.company import Company
from financial4all.sec.client import SECClient
from financial4all.sec.filings import Filing, get_filings

__all__ = ["Company", "SECClient", "Filing", "get_filings"]
