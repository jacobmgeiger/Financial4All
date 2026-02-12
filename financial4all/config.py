# financial4all/config.py
"""
Configuration management for Financial4All.

This module provides centralized configuration for SEC API identity and rate
limiting, caching, XBRL taxonomy paths, and logging. The SEC requires a
User-Agent header with contact information; use set_identity() before making
requests. Environment variables (F4A_SEC_EMAIL, F4A_RATE_LIMIT_DELAY, F4A_LOG_LEVEL)
override defaults when set.
"""

import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Central configuration for the Financial4All package.

    Attributes:
        sec_email: Email used in the SEC User-Agent header (required by SEC).
        sec_user_agent: Full User-Agent string; defaults to "Financial4All {sec_email}".
        rate_limit_delay: Seconds to wait between SEC API requests (SEC suggests ≤10/sec).
        cache_enabled: Whether to use caching for API responses.
        cache_ttl: Cache time-to-live in seconds.
        xbrl_taxonomy_path: Optional path to local US-GAAP/other taxonomy files.
        log_level: Logging level (e.g. "INFO", "DEBUG").
    """

    # SEC API Configuration
    sec_email: str = "your_email@example.com"
    sec_user_agent: Optional[str] = None

    # Rate Limiting (SEC recommends max 10 requests per second)
    rate_limit_delay: float = 0.1

    # Cache Settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds

    # Local XBRL cache (edgartools-style): when set, store fetched XBRL to disk
    cache_dir: Optional[str] = None

    # XBRL Settings
    xbrl_taxonomy_path: Optional[str] = None

    # Statement-centric extraction: use filing-level XBRL (vs company facts API)
    use_filing_xbrl: bool = True

    # Hybrid gap fill: when using filing-based extraction, supplement with company
    # facts API for (concept, period) gaps to improve historical coverage (EdgarTools)
    supplement_with_company_facts: bool = True

    # 10-Q summation: when annual 10-K fact is missing, sum 4 consecutive 10-Q facts
    # to fill the gap. Enables CapEx for companies (e.g. NVDA 2013–2021) that report
    # only quarterly in 10-Q with no 10-K annual fact.
    fill_gaps_from_10q: bool = True

    # EdgarTools-aligned validation and cleanup
    exclude_amended_filings: bool = False  # When True, prefer 10-K over 10-K/A in fact resolution
    run_datapoint_validation: bool = True  # Run post-extraction validation (YoY, margins)
    apply_presentation_signs: bool = False  # Apply preferred sign transformation (expenses positive, outflows negative)
    exclude_structural_elements: bool = True  # Filter Axis, Domain, Member, Table, Abstract from fact resolution

    # Statement DataFrame caching: when True, bypass cache and recompute to_dataframe() each call
    disable_statement_cache: bool = True  # Default True during testing; set False for production perf

    # Logging
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Set user agent from email if not explicitly set."""
        if self.sec_user_agent is None:
            self.sec_user_agent = f"Financial4All {self.sec_email}"


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config: The global configuration instance
    """
    global _config
    if _config is None:
        _config = Config()
        # Load from environment variables if available
        if os.getenv("F4A_SEC_EMAIL"):
            _config.sec_email = os.getenv("F4A_SEC_EMAIL")
        if os.getenv("F4A_RATE_LIMIT_DELAY"):
            _config.rate_limit_delay = float(os.getenv("F4A_RATE_LIMIT_DELAY"))
        if os.getenv("F4A_LOG_LEVEL"):
            _config.log_level = os.getenv("F4A_LOG_LEVEL")
        if os.getenv("F4A_USE_FILING_XBRL"):
            _config.use_filing_xbrl = os.getenv("F4A_USE_FILING_XBRL", "").lower() in ("1", "true", "yes")
        env_supp = os.getenv("F4A_SUPPLEMENT_WITH_COMPANY_FACTS", "").lower()
        if env_supp in ("1", "true", "yes"):
            _config.supplement_with_company_facts = True
        elif env_supp in ("0", "false", "no"):
            _config.supplement_with_company_facts = False
        env_10q = os.getenv("F4A_FILL_GAPS_FROM_10Q", "").lower()
        if env_10q in ("1", "true", "yes"):
            _config.fill_gaps_from_10q = True
        elif env_10q in ("0", "false", "no"):
            _config.fill_gaps_from_10q = False
        if os.getenv("F4A_CACHE_DIR"):
            _config.cache_dir = os.getenv("F4A_CACHE_DIR")
        env_exclude = os.getenv("F4A_EXCLUDE_AMENDED_FILINGS", "").lower()
        if env_exclude in ("1", "true", "yes"):
            _config.exclude_amended_filings = True
        elif env_exclude in ("0", "false", "no"):
            _config.exclude_amended_filings = False
        env_validation = os.getenv("F4A_RUN_DATAPOINT_VALIDATION", "").lower()
        if env_validation in ("0", "false", "no"):
            _config.run_datapoint_validation = False
        env_presentation = os.getenv("F4A_APPLY_PRESENTATION_SIGNS", "").lower()
        if env_presentation in ("1", "true", "yes"):
            _config.apply_presentation_signs = True
        env_structural = os.getenv("F4A_EXCLUDE_STRUCTURAL_ELEMENTS", "").lower()
        if env_structural in ("0", "false", "no"):
            _config.exclude_structural_elements = False
        env_cache = os.getenv("F4A_DISABLE_STATEMENT_CACHE", "").lower()
        if env_cache in ("1", "true", "yes"):
            _config.disable_statement_cache = True
        elif env_cache in ("0", "false", "no"):
            _config.disable_statement_cache = False
    return _config


def set_identity(email: str) -> None:
    """
    Set the identity (email) for SEC API requests.
    
    The SEC requires a User-Agent header with contact information for all requests.
    
    Args:
        email: Email address to use in User-Agent header
        
    Example:
        >>> from financial4all import set_identity
        >>> set_identity("your.name@example.com")
    """
    config = get_config()
    config.sec_email = email
    config.sec_user_agent = f"Financial4All {email}"


def set_rate_limit(delay: float) -> None:
    """
    Set the rate limit delay between SEC API requests.
    
    Args:
        delay: Delay in seconds between requests (default: 0.1 for ~10 req/sec)
    """
    config = get_config()
    config.rate_limit_delay = delay
