# financial4all/sec/company.py
"""
Company lookup and CIK management.

This module provides functionality for looking up companies by ticker symbol
and managing CIK (Central Index Key) conversions.
"""

import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path

from financial4all.core import resource_path, get_data_dir, log
from financial4all.utils.validators import validate_ticker, validate_cik, normalize_cik
from financial4all.sec.client import SECClient
from financial4all.exceptions import CompanyNotFoundError, InvalidTickerError


class Company:
    """
    Represents a company with SEC filing data.
    
    This class provides methods to look up company information,
    retrieve filings, and access financial data.
    """
    
    _cik_dict: Optional[pd.DataFrame] = None
    
    def __init__(self, ticker: str, client: Optional[SECClient] = None):
        """
        Initialize a Company instance.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            client: Optional SECClient instance (creates new one if not provided)
            
        Raises:
            ValueError: If ticker is invalid or not found
        """
        if not validate_ticker(ticker):
            raise InvalidTickerError(f"Invalid ticker symbol: {ticker}")
        
        self.ticker = ticker.upper()
        self.client = client or SECClient()
        self._cik: Optional[str] = None
        self._company_info: Optional[Dict[str, Any]] = None
    
    @classmethod
    def _load_cik_dict(cls) -> pd.DataFrame:
        """Load the CIK dictionary from CSV file."""
        if cls._cik_dict is None:
            cik_path = resource_path("CIK_dict.csv")
            if not Path(cik_path).exists():
                # Try alternative location
                data_dir = get_data_dir()
                cik_path = data_dir / "CIK_dict.csv"
                if not cik_path.exists():
                    raise FileNotFoundError(
                        f"CIK_dict.csv not found. Expected at: {resource_path('CIK_dict.csv')} or {cik_path}"
                    )
            
            cls._cik_dict = pd.read_csv(
                str(cik_path),
                converters={"cik_str": str}
            )
            log.debug(f"Loaded CIK dictionary with {len(cls._cik_dict)} companies")
        
        return cls._cik_dict
    
    @property
    def cik(self) -> str:
        """
        Get the company's CIK (Central Index Key).
        
        Returns:
            10-digit CIK string
            
        Raises:
            ValueError: If company not found in CIK dictionary
        """
        if self._cik is None:
            cik_dict = self._load_cik_dict()
            row = cik_dict[cik_dict["ticker"].str.upper() == self.ticker]
            
            if row.empty:
                raise CompanyNotFoundError(f"Ticker {self.ticker} not found in CIK dictionary")
            
            self._cik = normalize_cik(row["cik_str"].iloc[0])
        
        return self._cik
    
    @property
    def company_info(self) -> Dict[str, Any]:
        """
        Get company information (title, CIK, etc.).
        
        Returns:
            Dictionary with company information
        """
        if self._company_info is None:
            cik_dict = self._load_cik_dict()
            row = cik_dict[cik_dict["ticker"].str.upper() == self.ticker]
            
            if row.empty:
                raise CompanyNotFoundError(f"Ticker {self.ticker} not found")
            
            self._company_info = {
                "ticker": self.ticker,
                "title": row["title"].iloc[0],
                "cik": self.cik,
            }
        
        return self._company_info
    
    def get_filings(self, form: Optional[str] = None) -> list:
        """
        Get list of filings for this company.
        
        Args:
            form: Optional form type filter (e.g., "10-K", "10-Q")
            
        Returns:
            List of Filing objects
        """
        from financial4all.sec.filings import get_filings
        return get_filings(self.cik, form=form, client=self.client)
    
    def get_financials(self):
        """
        Get financial statements for this company.
        
        Returns:
            Financials object with income statement, balance sheet, etc.
        """
        from financial4all.financials import IncomeStatement, BalanceSheet, CashFlowStatement
        
        # Get the most recent 10-K filing
        filings = self.get_filings(form="10-K")
        if not filings:
            raise ValueError(f"No 10-K filings found for {self.ticker}")
        
        # For now, use the company facts API (backward compatible)
        facts_data = self.client.get_company_facts(self.cik)
        
        # This will be enhanced in Phase 3 to use proper XBRL parsing
        # Pass CIK to enable entity info extraction for fiscal period classification
        return {
            "income_statement": IncomeStatement.from_company_facts(facts_data, cik=self.cik),
            "balance_sheet": BalanceSheet.from_company_facts(facts_data, cik=self.cik),
            "cash_flow": CashFlowStatement.from_company_facts(facts_data, cik=self.cik),
        }
    
    def __repr__(self) -> str:
        """String representation of Company."""
        return f"Company(ticker='{self.ticker}', cik='{self.cik}')"
