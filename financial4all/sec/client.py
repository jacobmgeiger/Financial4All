# financial4all/sec/client.py
"""
SEC API client with rate limiting and error handling.

This module provides a robust HTTP client for interacting with the SEC EDGAR API,
including rate limiting, retry logic, and proper error handling.
"""

import time
import logging
from typing import Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from financial4all.config import get_config
from financial4all.core import log
from financial4all.exceptions import SECAPIError

logger = logging.getLogger(__name__)


class SECClient:
    """
    HTTP client for SEC EDGAR API requests.
    
    Features:
    - Rate limiting (respects SEC guidelines)
    - Automatic retry with exponential backoff
    - Proper User-Agent header management
    - Error handling for HTTP errors
    """
    
    BASE_URL = "https://data.sec.gov"
    
    def __init__(self, email: Optional[str] = None):
        """
        Initialize SEC client.
        
        Args:
            email: Email address for User-Agent header (uses config default if not provided)
        """
        self.config = get_config()
        if email:
            self.config.sec_email = email
            self.config.sec_user_agent = f"Financial4All {email}"
        
        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            "User-Agent": self.config.sec_user_agent or f"Financial4All {self.config.sec_email}",
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        })
        
        self._last_request_time = 0.0
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        """
        Make a GET request to the SEC API.
        
        Args:
            endpoint: API endpoint (relative to BASE_URL)
            params: Query parameters
            
        Returns:
            Response object
            
        Raises:
            requests.HTTPError: For HTTP error responses
        """
        self._rate_limit()
        
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"SEC API request failed: {url} - {e}")
            raise SECAPIError(f"SEC API request failed: {url}") from e
    
    def get_json(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request and return JSON response.
        
        Args:
            endpoint: API endpoint (relative to BASE_URL)
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        response = self.get(endpoint, params)
        return response.json()
    
    def get_company_facts(self, cik: str) -> Dict[str, Any]:
        """
        Get company facts (XBRL data) for a given CIK.
        
        Args:
            cik: Central Index Key (10-digit string)
            
        Returns:
            Company facts dictionary
        """
        # Ensure CIK is properly formatted
        cik_normalized = str(cik).strip().zfill(10)
        endpoint = f"api/xbrl/companyfacts/CIK{cik_normalized}.json"
        return self.get_json(endpoint)
    
    def get_submissions(self, cik: str) -> Dict[str, Any]:
        """
        Get company submissions (filings list) for a given CIK.
        
        Args:
            cik: Central Index Key (10-digit string)
            
        Returns:
            Submissions dictionary
        """
        cik_normalized = str(cik).strip().zfill(10)
        endpoint = f"submissions/CIK{cik_normalized}.json"
        return self.get_json(endpoint)
