# financial4all/exceptions.py
"""
Custom exceptions for Financial4All.

Defines a single base (Financial4AllError) and domain-specific subclasses
for SEC API failures, company/ticker/CIK validation, XBRL parsing, missing
data, and calculation errors. Use these for consistent error handling and
to avoid catching broad Exception.
"""


class Financial4AllError(Exception):
    """Base exception for all Financial4All errors. Subclass for domain-specific errors."""

    pass


class SECAPIError(Financial4AllError):
    """Raised when an SEC EDGAR/data.sec.gov request fails (HTTP errors, timeouts, invalid response)."""

    pass


class CompanyNotFoundError(Financial4AllError):
    """Raised when a company cannot be resolved (e.g. ticker or CIK not found in SEC data)."""

    pass


class InvalidTickerError(Financial4AllError):
    """Raised when a ticker symbol is invalid or unsupported."""

    pass


class InvalidCIKError(Financial4AllError):
    """Raised when a Central Index Key (CIK) is malformed or invalid."""

    pass


class XBRLError(Financial4AllError):
    """Raised when XBRL parsing or fact extraction fails."""

    pass


class NoDataError(Financial4AllError):
    """Raised when requested financial data is not available (e.g. no facts for a concept)."""

    pass


class CalculationError(Financial4AllError):
    """Raised when a financial calculation (e.g. statement articulation) fails or is inconsistent."""

    pass
