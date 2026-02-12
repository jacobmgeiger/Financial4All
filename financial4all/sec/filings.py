# financial4all/sec/filings.py
"""
SEC filing retrieval and metadata.

Provides the Filing dataclass (CIK, accession number, form type, dates) and
get_filings(cik, form=..., client=...) to fetch the list of recent filings
from the submissions API. Used to discover 10-K/10-Q filings and build
links to SEC EDGAR. Filing supports XBRL content retrieval via get_xbrl_content()
for statement-centric financial extraction.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

from financial4all.sec.client import SECClient, SEC_ARCHIVE_URL
from financial4all.utils.validators import normalize_cik

if TYPE_CHECKING:
    pass


def _write_xbrl_cache(
    config: Any,
    cik: str,
    accession_number: str,
    content: str,
    ext: str,
) -> None:
    """Write XBRL content to local cache when cache_dir is set."""
    if not config or not config.cache_dir:
        return
    try:
        safe_acc = re.sub(r"[^\w\-]", "_", accession_number)
        cache_path = Path(config.cache_dir) / cik / f"{safe_acc}{ext}"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding="utf-8")
    except Exception:
        pass


# Document types that indicate XBRL content in SEC filing index
XBRL_DOCUMENT_TYPES = [
    "EXTRACTED XBRL INSTANCE DOCUMENT",
    "XBRL INSTANCE DOCUMENT",
    "XBRL INSTANCE FILE",
]


@dataclass
class Filing:
    """
    Represents a SEC filing.
    
    Attributes:
        cik: Company CIK
        accession_number: Filing accession number
        filing_date: Filing date
        form: Form type (e.g., "10-K", "10-Q")
        report_date: Report date
        document_count: Number of documents in filing
    """
    
    cik: str
    accession_number: str
    filing_date: str
    form: str
    report_date: Optional[str] = None
    document_count: Optional[int] = None
    
    # Cache for XBRL content (lazy-loaded)
    _xbrl_content: Optional[str] = field(default=None, repr=False)
    # Cache for linkbase content (lazy-loaded when get_xbrl_content fetches index)
    _linkbase_urls: Optional[Dict[str, str]] = field(default=None, repr=False)
    _presentation_content: Optional[str] = field(default=None, repr=False)
    _calculation_content: Optional[str] = field(default=None, repr=False)
    
    @property
    def filing_date_obj(self) -> datetime:
        """Get filing date as datetime object."""
        return datetime.strptime(self.filing_date, "%Y-%m-%d")
    
    @property
    def url(self) -> str:
        """Get URL to filing on SEC website."""
        acc_no_clean = self.accession_number.replace("-", "")
        return f"{SEC_ARCHIVE_URL}/Archives/edgar/data/{self.cik}/{acc_no_clean}/{self.accession_number}.txt"
    
    @property
    def document_index_url(self) -> str:
        """
        Get URL to the filing document index page.
        The index lists all documents in the filing (10-K .htm, XBRL .xml, etc.).
        """
        acc_no_clean = self.accession_number.replace("-", "")
        return (
            f"{SEC_ARCHIVE_URL}/Archives/edgar/data/{self.cik}/{acc_no_clean}/"
            f"{self.accession_number}-index.htm"
        )
    
    def _parse_index_for_xbrl_documents(
        self, index_html: str
    ) -> List[Tuple[str, str]]:
        """
        Parse the filing index HTML to find XBRL-related documents.
        
        SEC index format: tables with rows containing Description|Document|Type.
        - EXTRACTED XBRL INSTANCE DOCUMENT -> .xml (standalone, preferred)
        - 10-K -> .htm (Inline XBRL, fallback)
        
        Returns:
            List of (document_type, full_url) tuples
        """
        found: List[Tuple[str, str]] = []
        base_url = self.document_index_url.rsplit("/", 1)[0] + "/"
        
        def to_full_url(href: str) -> str:
            """Convert href to full URL."""
            href = href.strip()
            if href.startswith("http"):
                return href
            if href.startswith("/"):
                return SEC_ARCHIVE_URL + href
            return base_url + href
        
        # 1. Find EXTRACTED XBRL INSTANCE DOCUMENT (*_htm.xml) - preferred
        # Pattern: href or markdown link to *_htm.xml
        xml_match = re.search(
            r'(?:href=|]\()["\']?([^"\')\s]+_htm\.xml)["\']?',
            index_html,
            re.IGNORECASE,
        )
        if xml_match:
            found.append((
                "EXTRACTED XBRL INSTANCE DOCUMENT",
                to_full_url(xml_match.group(1)),
            ))
        
        # 2. Fallback: any .xml in Data Files section
        if not found:
            xml_match = re.search(
                r'href=["\']([^"\']+\.xml)["\']',
                index_html,
                re.IGNORECASE,
            )
            if xml_match:
                found.append((
                    "EXTRACTED XBRL INSTANCE DOCUMENT",
                    to_full_url(xml_match.group(1)),
                ))
        
        # 3. Find main 10-K .htm (Inline XBRL)
        # SEC may use ix?doc=/Archives/edgar/data/.../file.htm - extract path
        # Escape ) in character class to avoid terminating outer group
        ix_doc_match = re.search(
            r'ix\?doc=([^"&\)\s]+\.(?:htm|html))',
            index_html,
            re.IGNORECASE,
        )
        if ix_doc_match:
            path = ix_doc_match.group(1)
            if path.startswith("/"):
                full_url = SEC_ARCHIVE_URL + path
            else:
                full_url = path
            # Main 10-K is typically first in list (e.g. nvda-20250126.htm)
            if "ex" not in path.lower().split("/")[-1][:5]:
                found.append(("10-K", full_url))
        
        if not any(t == "10-K" for t, _ in found):
            # Direct link to .htm
            htm_match = re.search(
                r'href=["\']([^"\']+\.(?:htm|html))["\']',
                index_html,
                re.IGNORECASE,
            )
            if htm_match:
                href = htm_match.group(1)
                if "ex" not in href.lower().split("/")[-1][:5]:
                    found.append(("10-K", to_full_url(href)))
        
        return found

    def _parse_index_for_linkbase_urls(
        self, index_html: str
    ) -> Dict[str, str]:
        """
        Parse the filing index to find XBRL linkbase document URLs.

        SEC Data Files table lists EX-101.PRE (presentation), EX-101.CAL
        (calculation), EX-101.DEF (definition). Returns dict mapping
        'pre'|'cal'|'def' to full URL.

        Returns:
            Dict with keys 'pre', 'cal', 'def' mapping to URLs (or absent if not found)
        """
        base_url = self.document_index_url.rsplit("/", 1)[0] + "/"

        def to_full_url(href: str) -> str:
            href = href.strip()
            if href.startswith("http"):
                return href
            if href.startswith("/"):
                return SEC_ARCHIVE_URL + href
            return base_url + href

        result: Dict[str, str] = {}
        # SEC index: Data Files table has rows with Document (link) and Type (EX-101.PRE etc)
        # Split by <tr> and find rows containing the type, then extract href from that row
        for lb_type, ext in [("pre", "EX-101.PRE"), ("cal", "EX-101.CAL"), ("def", "EX-101.DEF")]:
            # Find table rows - look for tr that contains both href to .xml and the type
            row_pattern = rf'<tr[^>]*>(.*?)</tr>'
            for row_match in re.finditer(row_pattern, index_html, re.IGNORECASE | re.DOTALL):
                row_content = row_match.group(1)
                if ext not in row_content:
                    continue
                href_match = re.search(r'href=["\']([^"\']+\.xml)["\']', row_content, re.IGNORECASE)
                if href_match:
                    result[lb_type] = to_full_url(href_match.group(1))
                    break
        return result

    def get_presentation_linkbase_content(
        self, client: Optional[SECClient] = None
    ) -> Optional[str]:
        """
        Fetch presentation linkbase (EX-101.PRE) from this filing.

        Returns:
            XML content of presentation linkbase, or None if not found
        """
        if self._presentation_content is not None:
            return self._presentation_content
        if client is None:
            client = SECClient()
        if self._linkbase_urls is None:
            try:
                index_html = client.get_filing_index(self.cik, self.accession_number)
                self._linkbase_urls = self._parse_index_for_linkbase_urls(index_html)
            except Exception:
                self._linkbase_urls = {}
        url = self._linkbase_urls.get("pre")
        if not url:
            return None
        try:
            resp = client.get_url(url)
            self._presentation_content = resp.text
            return self._presentation_content
        except Exception:
            return None

    def get_calculation_linkbase_content(
        self, client: Optional[SECClient] = None
    ) -> Optional[str]:
        """
        Fetch calculation linkbase (EX-101.CAL) from this filing.

        Returns:
            XML content of calculation linkbase, or None if not found
        """
        if self._calculation_content is not None:
            return self._calculation_content
        if client is None:
            client = SECClient()
        if self._linkbase_urls is None:
            try:
                index_html = client.get_filing_index(self.cik, self.accession_number)
                self._linkbase_urls = self._parse_index_for_linkbase_urls(index_html)
            except Exception:
                self._linkbase_urls = {}
        url = self._linkbase_urls.get("cal")
        if not url:
            return None
        try:
            resp = client.get_url(url)
            self._calculation_content = resp.text
            return self._calculation_content
        except Exception:
            return None

    def get_xbrl_content(
        self, client: Optional[SECClient] = None
    ) -> Optional[str]:
        """
        Fetch and return XBRL content from this filing.
        Tries direct URL guess first ({acc_no_clean}_htm.xml) to save 1 request;
        falls back to index parse. Prefers standalone .xml over .htm (Inline XBRL).
        
        Args:
            client: Optional SECClient; if None, creates a new one.
            
        Returns:
            XBRL content as string (XML or HTML/iXBRL), or None if not found.
        """
        if self._xbrl_content is not None:
            return self._xbrl_content
        
        if client is None:
            client = SECClient()
        
        # Phase 3: Check local cache if cache_dir is set
        from financial4all.config import get_config
        config = get_config()
        if config.cache_dir:
            safe_acc = re.sub(r"[^\w\-]", "_", self.accession_number)
            cache_path = Path(config.cache_dir) / self.cik / f"{safe_acc}.xml"
            if cache_path.exists():
                try:
                    self._xbrl_content = cache_path.read_text(encoding="utf-8")
                    return self._xbrl_content
                except Exception:
                    pass
            cache_path_htm = Path(config.cache_dir) / self.cik / f"{safe_acc}.htm"
            if cache_path_htm.exists():
                try:
                    self._xbrl_content = cache_path_htm.read_text(encoding="utf-8")
                    return self._xbrl_content
                except Exception:
                    pass
        
        # Phase 2: Try direct XBRL URL before parsing index (saves 1 request)
        direct_url = client.get_direct_xbrl_url(self.cik, self.accession_number)
        try:
            resp = client.try_get_url(direct_url)
            if resp is not None:
                content = resp.text
                if content and ("xbrl" in content.lower() or "xbrl:" in content or "xmlns" in content):
                    self._xbrl_content = content
                    _write_xbrl_cache(config, self.cik, self.accession_number, content, ".xml")
                    return content
        except Exception:
            pass
        
        try:
            index_html = client.get_filing_index(self.cik, self.accession_number)
        except Exception:
            return None
        
        documents = self._parse_index_for_xbrl_documents(index_html)
        
        # Prefer .xml (standalone XBRL) over .htm (Inline XBRL)
        xml_docs = [(t, u) for t, u in documents if u.lower().endswith(".xml")]
        htm_docs = [(t, u) for t, u in documents if u.lower().endswith((".htm", ".html"))]
        
        if xml_docs:
            # Use first EXTRACTED XBRL or standalone .xml
            _, url = xml_docs[0]
            try:
                content = client.get_filing_document(url)
                self._xbrl_content = content
                _write_xbrl_cache(config, self.cik, self.accession_number, content, ".xml")
                return content
            except Exception:
                pass
        
        if htm_docs:
            # Fall back to main .htm (Inline XBRL)
            _, url = htm_docs[0]
            try:
                content = client.get_filing_document(url)
                self._xbrl_content = content
                _write_xbrl_cache(config, self.cik, self.accession_number, content, ".htm")
                return content
            except Exception:
                pass
        
        return None
    
    @property
    def xbrl_content(self) -> Optional[str]:
        """
        Lazy-loaded XBRL content from this filing.
        Calls get_xbrl_content() with default client on first access.
        """
        if self._xbrl_content is None:
            self.get_xbrl_content()
        return self._xbrl_content
    
    def __repr__(self) -> str:
        """String representation of Filing."""
        return f"Filing(cik='{self.cik}', form='{self.form}', date='{self.filing_date}')"


def _parse_filings_from_data(
    filings_data: Dict[str, Any],
    cik_normalized: str,
    form: Optional[str] = None,
) -> List[Filing]:
    """Parse filings from SEC submissions columnar data (recent or files)."""
    filings = []
    acc_nums = filings_data.get("accessionNumber", [])
    for idx in range(len(acc_nums)):
        filing_form = filings_data["form"][idx]
        if form and filing_form != form:
            continue
        filing = Filing(
            cik=cik_normalized,
            accession_number=acc_nums[idx],
            filing_date=filings_data["filingDate"][idx],
            form=filing_form,
            report_date=filings_data.get("reportDate", [None] * len(filings_data["form"]))[idx],
            document_count=filings_data.get("documentCount", [None] * len(filings_data["form"]))[idx],
        )
        filings.append(filing)
    return filings


def get_filings(
    cik: str,
    form: Optional[str] = None,
    client: Optional[SECClient] = None,
) -> List[Filing]:
    """
    Get list of filings for a company.

    Uses SEC submissions "recent" first, then fetches "files" (older submissions)
    when available to ensure full history. Needed because "recent" has only
    ~1000 filings—e.g. NVDA has only 6 10-Ks in recent; files contain 1998–2019.

    Args:
        cik: Company CIK
        form: Optional form type filter (e.g., "10-K", "10-Q")
        client: Optional SECClient instance

    Returns:
        List of Filing objects (most recent first)
    """
    if client is None:
        client = SECClient()

    cik_normalized = normalize_cik(cik)
    submissions = client.get_submissions(cik_normalized)
    filings_container = submissions.get("filings", {})

    # Start with "recent" (newest ~1000 filings)
    filings_data = filings_container.get("recent", {})
    filings = _parse_filings_from_data(filings_data, cik_normalized, form)

    # Fetch "files" (older submissions) to get full history when form filter is set
    files_list = filings_container.get("files", [])
    if form and files_list:
        for file_entry in files_list:
            file_name = file_entry.get("name") if isinstance(file_entry, dict) else None
            if not file_name:
                continue
            try:
                older = client.get_submissions_file(file_name)
                # Files have columnar data at top level (accessionNumber, form, etc.)
                older_filings = _parse_filings_from_data(older, cik_normalized, form)
                filings.extend(older_filings)
            except Exception:
                continue

    return filings


def fetch_xbrl_parallel(
    filings: List[Filing],
    client: Optional[SECClient] = None,
    max_workers: int = 5,
) -> None:
    """
    Fetch XBRL content for all filings in parallel (EdgarTools-style optimization).

    Populates Filing._xbrl_content for each filing in-place. Subsequent
    from_filing/from_filings calls will use cached content. Respects SEC
    10 req/sec guideline (3-5 parallel requests typical).

    Args:
        filings: List of Filing objects to fetch
        client: Optional SECClient; if None, each thread creates/uses default
        max_workers: Max concurrent fetches (default 5, within SEC limits)
    """
    if not filings:
        return
    if client is None:
        client = SECClient()

    def fetch_one(filing: Filing) -> None:
        filing.get_xbrl_content(client)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(fetch_one, filings))


def head_filings(
    filings: List[Filing],
    n: int,
    exclude_amended: bool = True,
) -> List[Filing]:
    """
    Return first N filings, optionally excluding amended (form ending with /A).

    Args:
        filings: List of Filing objects (typically from get_filings)
        n: Maximum number of filings to return
        exclude_amended: If True, skip filings with form ending in /A

    Returns:
        List of up to n Filing objects
    """
    if exclude_amended:
        filtered = [f for f in filings if not f.form.endswith("/A")]
    else:
        filtered = list(filings)
    return filtered[:n]
