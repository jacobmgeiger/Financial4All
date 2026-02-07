# financial4all/exceptions.py
"""
Custom exceptions for Financial4All.

This module defines custom exception classes for better error handling.
"""


class Financial4AllError(Exception):
    """Base exception for all Financial4All errors."""
    pass


class SECAPIError(Financial4AllError):
    """Exception raised for SEC API errors."""
    pass


class CompanyNotFoundError(Financial4AllError):
    """Exception raised when a company is not found."""
    pass


class InvalidTickerError(Financial4AllError):
    """Exception raised for invalid ticker symbols."""
    pass


class InvalidCIKError(Financial4AllError):
    """Exception raised for invalid CIK values."""
    pass


class XBRLError(Financial4AllError):
    """Exception raised for XBRL parsing errors."""
    pass


class NoDataError(Financial4AllError):
    """Exception raised when no data is available."""
    pass


class CalculationError(Financial4AllError):
    """Exception raised for calculation errors."""
    pass
