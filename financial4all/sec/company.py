# financial4all/sec/company.py
"""
Company lookup and financial data access by ticker.

Resolves ticker -> CIK via CIK_dict.csv, then fetches company facts (XBRL)
from the SEC and builds IncomeStatement, BalanceSheet, CashFlowStatement, and
FinancialRatios. Entry point for get_financials() used by the app and callers.
"""

import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path

from financial4all.core import resource_path, get_data_dir, log  # noqa: F401 - log used in get_financials
from financial4all.utils.validators import validate_ticker, validate_cik, normalize_cik
from financial4all.sec.client import SECClient
from financial4all.exceptions import CompanyNotFoundError, InvalidTickerError


class Company:
    """
    SEC-registered company identified by ticker, with access to standardized financials.

    Lazy-loads CIK from CIK_dict.csv and fetches company facts (XBRL) on first
    access to get_financials(). Raises InvalidTickerError or CompanyNotFoundError
    if ticker is invalid or not in the CIK dictionary.
    """

    _cik_dict: Optional[pd.DataFrame] = None

    def __init__(self, ticker: str, client: Optional[SECClient] = None) -> None:
        """
        Initialize a company by ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL", "NVDA").
            client: Optional SECClient; if None, a new client is created.

        Raises:
            InvalidTickerError: If ticker format is invalid or not found in CIK dictionary.
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

            cls._cik_dict = pd.read_csv(str(cik_path), converters={"cik_str": str})
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
                raise CompanyNotFoundError(
                    f"Ticker {self.ticker} not found in CIK dictionary"
                )

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

    def get_financials(
        self,
        periods: Optional[int] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        Get financial statements for this company.

        Uses statement-centric extraction (filing-level XBRL) when
        use_filing_xbrl=True in config, falling back to company facts API
        if filing-based extraction fails.

        Args:
            periods: If > 1, fetch this many 10-K filings and merge for multi-year
                data. If None or 1, uses only the latest 10-K (~3 years).
            use_cache: If True, use local XBRL cache (sets cache_dir to default
                when not configured).

        Returns:
            Dict with income_statement, balance_sheet, cash_flow
        """
        from financial4all.financials import (
            IncomeStatement,
            BalanceSheet,
            CashFlowStatement,
        )
        from financial4all.config import get_config
        from financial4all.sec.filings import head_filings, fetch_xbrl_parallel
        from financial4all.xbrl.xbrl import XBRLFilingWithNoXbrlData

        config = get_config()
        if use_cache and not config.cache_dir:
            import os
            default_cache = os.path.join(os.path.expanduser("~"), ".financial4all", "cache")
            config.cache_dir = default_cache
        filings = self.get_filings(form="10-K")
        if not filings:
            raise ValueError(f"No 10-K filings found for {self.ticker}")

        use_multi = periods is not None and periods > 1
        if use_multi:
            filings_to_use = head_filings(filings, periods)
            if not filings_to_use:
                raise ValueError(f"No non-amended 10-K filings found for {self.ticker}")
            # Pre-fetch XBRL in parallel (EdgarTools-style); from_filings uses cached content
            fetch_xbrl_parallel(filings_to_use, client=self.client)
        else:
            filing = next((f for f in filings if not f.form.endswith("/A")), filings[0])
            filings_to_use = [filing]

        if config.use_filing_xbrl:
            try:
                if use_multi:
                    result = {
                        "income_statement": IncomeStatement.from_filings(
                            filings_to_use, client=self.client
                        ),
                        "balance_sheet": BalanceSheet.from_filings(
                            filings_to_use, client=self.client
                        ),
                        "cash_flow": CashFlowStatement.from_filings(
                            filings_to_use, client=self.client
                        ),
                    }
                    # Hybrid gap fill: supplement with company facts for missing (concept, period)
                    if config.supplement_with_company_facts:
                        try:
                            facts_data = self.client.get_company_facts(self.cik)
                            for stmt in result.values():
                                stmt.supplement_from_company_facts(
                                    facts_data, cik=self.cik
                                )
                        except Exception as sup_e:
                            log.debug(
                                f"Company facts supplement skipped for {self.ticker}: {sup_e}"
                            )
                    return result
                result = {
                    "income_statement": IncomeStatement.from_filing(
                        filings_to_use[0], client=self.client
                    ),
                    "balance_sheet": BalanceSheet.from_filing(
                        filings_to_use[0], client=self.client
                    ),
                    "cash_flow": CashFlowStatement.from_filing(
                        filings_to_use[0], client=self.client
                    ),
                }
                if config.supplement_with_company_facts:
                    try:
                        facts_data = self.client.get_company_facts(self.cik)
                        for stmt in result.values():
                            stmt.supplement_from_company_facts(
                                facts_data, cik=self.cik
                            )
                    except Exception as sup_e:
                        log.debug(
                            f"Company facts supplement skipped for {self.ticker}: {sup_e}"
                        )
                return result
            except (XBRLFilingWithNoXbrlData, ValueError, Exception) as e:
                log.warning(
                    f"Filing-based XBRL extraction failed for {self.ticker}: {e}. "
                    "Falling back to company facts API."
                )

        # Fallback: company facts API
        facts_data = self.client.get_company_facts(self.cik)
        return {
            "income_statement": IncomeStatement.from_company_facts(
                facts_data, cik=self.cik
            ),
            "balance_sheet": BalanceSheet.from_company_facts(facts_data, cik=self.cik),
            "cash_flow": CashFlowStatement.from_company_facts(facts_data, cik=self.cik),
        }

    def __repr__(self) -> str:
        """String representation of Company."""
        return f"Company(ticker='{self.ticker}', cik='{self.cik}')"
