# financial4all/sec/client.py
"""
SEC EDGAR API client with rate limiting and error handling.

Provides SECClient for HTTPS requests to data.sec.gov: company facts (XBRL),
submissions (filings list), and other endpoints. Also supports fetching
filing documents from SEC Archives (www.sec.gov). Enforces rate limiting
between calls, uses a requests Session with retry/backoff for 429/5xx, and
sets the required User-Agent. All methods raise SECAPIError or ValueError on
failure.
"""

import re
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from financial4all.config import get_config
from financial4all.core import log
from financial4all.exceptions import SECAPIError

logger = logging.getLogger(__name__)

# SEC Archives base URL for filing documents (distinct from data.sec.gov API)
SEC_ARCHIVE_URL = "https://www.sec.gov"


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
        
        # Set default headers (Host varies by endpoint; SEC accepts requests without
        # explicit Host for both data.sec.gov and www.sec.gov)
        self.session.headers.update({
            "User-Agent": self.config.sec_user_agent or f"Financial4All {self.config.sec_email}",
            "Accept-Encoding": "gzip, deflate",
        })
        
        self._last_request_time = 0.0
    
    def _rate_limit(self) -> None:
        """Sleep if needed so that requests do not exceed config.rate_limit_delay spacing."""
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
            SECAPIError: For HTTP error responses or network errors
        """
        if not endpoint:
            raise ValueError("Endpoint cannot be empty")
        
        self._rate_limit()
        
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout as e:
            logger.error(f"SEC API request timed out: {url}")
            raise SECAPIError(f"SEC API request timed out: {url}") from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            logger.error(f"SEC API HTTP error {status_code}: {url} - {e}")
            raise SECAPIError(f"SEC API HTTP error {status_code}: {url}") from e
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
            
        Raises:
            SECAPIError: If request fails
            ValueError: If CIK is invalid
        """
        if not cik:
            raise ValueError("CIK cannot be empty")
        
        try:
            # Ensure CIK is properly formatted
            cik_normalized = str(cik).strip().zfill(10)
            if not cik_normalized.isdigit() or len(cik_normalized) != 10:
                raise ValueError(f"Invalid CIK format: {cik}")
            
            endpoint = f"api/xbrl/companyfacts/CIK{cik_normalized}.json"
            result = self.get_json(endpoint)
            
            # Validate response structure
            if not isinstance(result, dict):
                raise SECAPIError(f"Invalid response format from SEC API for CIK {cik_normalized}")
            
            return result
        except ValueError:
            raise
        except SECAPIError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching company facts for CIK {cik}: {e}")
            raise SECAPIError(f"Error fetching company facts: {str(e)}") from e
    
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

    def get_submissions_file(self, file_name: str) -> Dict[str, Any]:
        """
        Get older submissions from a file in the filings.files array.
        Used when "recent" has fewer than needed (e.g. only ~6 10-Ks for NVDA).
        
        Args:
            file_name: e.g. "CIK0001045810-submissions-001.json"
            
        Returns:
            Submissions dictionary with same structure as recent
        """
        if not file_name or not file_name.endswith(".json"):
            raise ValueError("Invalid submissions file name")
        endpoint = f"submissions/{file_name}"
        return self.get_json(endpoint)
    
    def get_url(self, url: str) -> requests.Response:
        """
        Make a GET request to an arbitrary URL (e.g. SEC Archives).
        Uses same rate limiting and error handling as get().
        
        Args:
            url: Full URL to fetch
            
        Returns:
            Response object
            
        Raises:
            SECAPIError: For HTTP error responses or network errors
        """
        if not url:
            raise ValueError("URL cannot be empty")
        
        self._rate_limit()
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout as e:
            logger.error(f"SEC request timed out: {url}")
            raise SECAPIError(f"SEC request timed out: {url}") from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            logger.error(f"SEC HTTP error {status_code}: {url} - {e}")
            raise SECAPIError(f"SEC HTTP error {status_code}: {url}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"SEC request failed: {url} - {e}")
            raise SECAPIError(f"SEC request failed: {url}") from e
    
    def get_filing_index(
        self, cik: str, accession_number: str
    ) -> str:
        """
        Fetch the filing index HTML for a given accession number.
        The index lists all documents in the filing (e.g. 10-K .htm, XBRL .xml).
        
        Args:
            cik: Central Index Key (10-digit string)
            accession_number: SEC accession number (e.g. 0001045810-25-000023)
            
        Returns:
            Raw HTML content of the filing index page
            
        Raises:
            SECAPIError: If request fails
            ValueError: If parameters are invalid
        """
        if not cik or not accession_number:
            raise ValueError("CIK and accession_number are required")
        
        cik_normalized = str(cik).strip().zfill(10)
        acc_no_clean = accession_number.replace("-", "")
        
        url = (
            f"{SEC_ARCHIVE_URL}/Archives/edgar/data/{cik_normalized}/"
            f"{acc_no_clean}/{accession_number}-index.htm"
        )
        
        response = self.get_url(url)
        return response.text
    
    def try_get_url(self, url: str) -> Optional[requests.Response]:
        """
        Try to fetch a URL; return None on 404, raise on other errors.

        Used for direct XBRL URL guess - avoid parsing index when URL works.

        Args:
            url: Full URL to fetch

        Returns:
            Response object if 2xx, None if 404
        """
        if not url:
            raise ValueError("URL cannot be empty")
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError:
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"SEC request timed out: {url}")
            raise SECAPIError(f"SEC request timed out: {url}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"SEC request failed: {url} - {e}")
            raise SECAPIError(f"SEC request failed: {url}") from e

    def get_direct_xbrl_url(self, cik: str, accession_number: str) -> str:
        """
        Build common SEC URL for extracted XBRL: {base}/{acc_no_clean}_htm.xml.

        Many filers use this pattern; if 404, caller falls back to index parse.
        """
        cik_normalized = str(cik).strip().zfill(10)
        acc_no_clean = accession_number.replace("-", "")
        return (
            f"{SEC_ARCHIVE_URL}/Archives/edgar/data/{cik_normalized}/"
            f"{acc_no_clean}/{acc_no_clean}_htm.xml"
        )

    def get_filing_document(self, url: str) -> str:
        """
        Fetch a specific filing document (e.g. main 10-K .htm or XBRL .xml).
        
        Args:
            url: Full URL to the document (e.g. from filing index)
            
        Returns:
            Raw content of the document (HTML or XML string)
            
        Raises:
            SECAPIError: If request fails
        """
        response = self.get_url(url)
        return response.text
    
    def get_filing_full_submission(
        self, cik: str, accession_number: str
    ) -> str:
        """
        Fetch the full submission .txt file (SGML-like format).
        Contains the complete filing package.
        
        Args:
            cik: Central Index Key (10-digit string)
            accession_number: SEC accession number
            
        Returns:
            Raw content of the full submission file
            
        Raises:
            SECAPIError: If request fails
        """
        if not cik or not accession_number:
            raise ValueError("CIK and accession_number are required")
        
        cik_normalized = str(cik).strip().zfill(10)
        acc_no_clean = accession_number.replace("-", "")
        
        url = (
            f"{SEC_ARCHIVE_URL}/Archives/edgar/data/{cik_normalized}/"
            f"{acc_no_clean}/{accession_number}.txt"
        )
        
        response = self.get_url(url)
        return response.text
