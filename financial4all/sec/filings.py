# financial4all/sec/filings.py
"""
Filing retrieval and management.

This module provides functionality for retrieving and managing SEC filings.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any

from financial4all.sec.client import SECClient
from financial4all.utils.validators import normalize_cik


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
    
    @property
    def filing_date_obj(self) -> datetime:
        """Get filing date as datetime object."""
        return datetime.strptime(self.filing_date, "%Y-%m-%d")
    
    @property
    def url(self) -> str:
        """Get URL to filing on SEC website."""
        acc_no_clean = self.accession_number.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{acc_no_clean}/{self.accession_number}.txt"
    
    def __repr__(self) -> str:
        """String representation of Filing."""
        return f"Filing(cik='{self.cik}', form='{self.form}', date='{self.filing_date}')"


def get_filings(
    cik: str,
    form: Optional[str] = None,
    client: Optional[SECClient] = None
) -> List[Filing]:
    """
    Get list of filings for a company.
    
    Args:
        cik: Company CIK
        form: Optional form type filter (e.g., "10-K", "10-Q")
        client: Optional SECClient instance
        
    Returns:
        List of Filing objects
    """
    if client is None:
        client = SECClient()
    
    cik_normalized = normalize_cik(cik)
    submissions = client.get_submissions(cik_normalized)
    
    filings_data = submissions.get("filings", {}).get("recent", {})
    
    filings = []
    for idx in range(len(filings_data.get("accessionNumber", []))):
        filing_form = filings_data["form"][idx]
        
        # Filter by form if specified
        if form and filing_form != form:
            continue
        
        filing = Filing(
            cik=cik_normalized,
            accession_number=filings_data["accessionNumber"][idx],
            filing_date=filings_data["filingDate"][idx],
            form=filing_form,
            report_date=filings_data.get("reportDate", [None] * len(filings_data["form"]))[idx],
            document_count=filings_data.get("documentCount", [None] * len(filings_data["form"]))[idx],
        )
        filings.append(filing)
    
    return filings
