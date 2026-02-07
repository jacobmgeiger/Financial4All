# financial4all/config.py
"""
Configuration management for Financial4All.

This module handles centralized configuration including SEC API settings,
rate limiting, and cache configuration.
"""

import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration settings for Financial4All."""
    
    # SEC API Configuration
    sec_email: str = "your_email@example.com"
    sec_user_agent: Optional[str] = None
    
    # Rate Limiting
    rate_limit_delay: float = 0.1  # Seconds between requests (SEC recommends 10 requests/second max)
    
    # Cache Settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    
    # XBRL Settings
    xbrl_taxonomy_path: Optional[str] = None  # Path to XBRL taxonomy files
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
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
