# data_loader.py
"""
Backward compatibility layer for data_loader.py.

This module maintains backward compatibility with the original data_loader.py API
while using the new modular structure under the hood.
"""

import warnings
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

# Import new modules
from financial4all.sec.company import Company
from financial4all.sec.client import SECClient
from financial4all.config import set_identity, get_config
from financial4all.financials import IncomeStatement, BalanceSheet, CashFlowStatement, FinancialRatios
from financial4all.xbrl.facts import FactSet

# Maintain backward compatibility
EMAIL = "your_email@example.com"

# Set identity from EMAIL variable
_config = get_config()
if EMAIL != "your_email@example.com":
    set_identity(EMAIL)

# Load CIK dictionary for backward compatibility
from financial4all.core import resource_path
from pathlib import Path

CIK_dict_path = resource_path("CIK_dict.csv")
if Path(CIK_dict_path).exists():
    CIK_dict = pd.read_csv(CIK_dict_path, converters={"cik_str": str})
else:
    # Try alternative location
    from financial4all.core import get_data_dir
    alt_path = get_data_dir() / "CIK_dict.csv"
    if alt_path.exists():
        CIK_dict = pd.read_csv(str(alt_path), converters={"cik_str": str})
    else:
        CIK_dict = pd.DataFrame()  # Empty DataFrame if not found

# Load calculation formulas for backward compatibility
import json

formulas_path = resource_path("xbrl_prep/income_statement_formulas.json")
if Path(formulas_path).exists():
    with open(formulas_path, "r") as f:
        calculation_formulas = json.load(f)
else:
    calculation_formulas = {}

# Create inverted index for backward compatibility
from collections import defaultdict
calculation_formulas_inverted = defaultdict(list)
for parent, formulas in calculation_formulas.items():
    for formula in formulas:
        for child_info in formula.get("children", []):
            child_name = child_info.get("child")
            if child_name:
                calculation_formulas_inverted[child_name].append(parent)

# Income statement mapping for backward compatibility
INCOME_STATEMENT_MAPPING = {
    "Revenue": ["SalesRevenueNet", "Revenues", "RevenueFromContractWithCustomer"],
    "SalesRevenueNet": ["SalesRevenueNet"],
    "Revenues": ["Revenues"],
    "RevenueFromContractWithCustomer": ["RevenueFromContractWithCustomer"],
    "Cost of Revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "Gross Profit": ["GrossProfit"],
    "R&D Expenses": ["ResearchAndDevelopmentExpense"],
    "SG&A Expenses": ["SellingGeneralAndAdministrativeExpense"],
    "Operating Expenses": ["OperatingExpenses"],
    "Operating Income": ["OperatingIncomeLoss"],
    "Interest Income": ["InterestIncome", "InterestAndDividendIncome", "InterestIncomeOperating"],
    "Interest Expense": ["InterestExpense", "InterestExpenseOperating"],
    "Income Before Taxes": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTax",
    ],
    "Taxes": ["IncomeTaxExpenseBenefit"],
    "Net Income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "Basic EPS": ["EarningsPerShareBasic"],
    "Diluted EPS": ["EarningsPerShareDiluted"],
}


def get_company_info(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves company info (title, CIK) for a given ticker symbol.
    
    DEPRECATED: Use financial4all.Company instead.
    """
    warnings.warn(
        "get_company_info is deprecated. Use financial4all.Company instead.",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        company = Company(ticker)
        return {
            "title": company.company_info["title"],
            "cik_str": company.cik,
        }
    except Exception:
        return None


def get_cik(ticker: str) -> str:
    """
    Retrieves the Central Index Key (CIK) for a given company ticker symbol.
    
    DEPRECATED: Use financial4all.Company instead.
    """
    warnings.warn(
        "get_cik is deprecated. Use financial4all.Company instead.",
        DeprecationWarning,
        stacklevel=2
    )
    company = Company(ticker)
    return company.cik


def get_filing_by_metrics(CIK: str) -> Dict[str, Any]:
    """
    Fetches a company's entire fact history from the SEC EDGAR API.
    
    DEPRECATED: Use financial4all.Company.get_financials() instead.
    
    Args:
        CIK (str): The company's Central Index Key.
        
    Returns:
        dict: A dictionary containing all the us-gaap facts for the company.
    """
    warnings.warn(
        "get_filing_by_metrics is deprecated. Use financial4all.Company.get_financials() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    client = SECClient()
    company_facts = client.get_company_facts(CIK)
    return company_facts.get("facts", {}).get("us-gaap", {})


def extract_metrics(filing_metrics: Dict[str, Any]) -> Dict[str, list]:
    """
    Extracts and filters all annual data points from the raw SEC filing.
    
    DEPRECATED: Use financial4all.xbrl.facts.FactSet instead.
    """
    warnings.warn(
        "extract_metrics is deprecated. Use financial4all.xbrl.facts.FactSet instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Convert to FactSet format and back for compatibility
    fact_set = FactSet()
    
    for metric, attributes in filing_metrics.items():
        for unit, entries in attributes.get("units", {}).items():
            for entry in entries:
                if entry.get("form") == "10-K" and entry.get("fp") == "FY":
                    frame = entry.get('frame', '')
                    if 'Q' in frame:
                        continue
                    
                    from financial4all.xbrl.facts import Fact
                    from financial4all.xbrl.periods import Period
                    from datetime import datetime
                    
                    period = Period.from_xbrl_dict({
                        "end": entry["end"],
                        "start": entry.get("start"),
                    })
                    
                    fact = Fact(
                        concept=metric,
                        value=entry["val"],
                        unit=unit,
                        period=period,
                        form=entry.get("form"),
                        frame=entry.get("frame"),
                        filed=datetime.fromisoformat(entry["filed"]) if entry.get("filed") else None,
                    )
                    fact_set.add(fact)
    
    # Convert back to old format
    return fact_set.to_dict()


def process_metrics(ticker: str):
    """
    A convenience wrapper function that chains together all the steps required
    to fetch and process financial data for a given ticker.
    
    This now returns the comprehensive solved dataframe, the standard income statement,
    and a dictionary of alternative calculations.
    
    DEPRECATED: Use financial4all.Company.get_financials() instead.
    """
    warnings.warn(
        "process_metrics is deprecated. Use financial4all.Company.get_financials() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        # Use the new API
        company = Company(ticker)
        financials = company.get_financials()
        
        income_statement = financials["income_statement"]
        balance_sheet = financials["balance_sheet"]
        cash_flow = financials["cash_flow"]
        
        # Get income statement DataFrame
        standard_is_df = income_statement.to_dataframe()
        
        # For backward compatibility, create df_merged (all metrics)
        # This combines income statement with other available metrics
        df_merged = standard_is_df.copy()
        
        # Add balance sheet and cash flow metrics if available
        if balance_sheet:
            bs_df = balance_sheet.to_dataframe()
            if not bs_df.empty:
                df_merged = df_merged.join(bs_df, how='outer', rsuffix='_bs')
        
        if cash_flow:
            cf_df = cash_flow.to_dataframe()
            if not cf_df.empty:
                df_merged = df_merged.join(cf_df, how='outer', rsuffix='_cf')
        
        # Calculate ratios
        ratios = FinancialRatios(income_statement, balance_sheet, cash_flow)
        key_ratios_df = ratios.calculate_all_ratios()
        
        # Create alternatives dict (empty for now - could be enhanced)
        alternatives = {}
        standard_metrics_list = list(standard_is_df.columns) if not standard_is_df.empty else []
        
        return df_merged, standard_is_df, alternatives, standard_metrics_list, key_ratios_df
    except Exception as e:
        warnings.warn(f"Error processing metrics: {e}", UserWarning)
        return pd.DataFrame(), pd.DataFrame(), {}, [], pd.DataFrame()


def get_financial_reports(ticker: str):
    """
    Fetches all available 10-K filings for a given ticker and packages them into a
    single zip archive.
    
    DEPRECATED: Use financial4all.Company.get_filings() instead.
    """
    warnings.warn(
        "get_financial_reports is deprecated. Use financial4all.Company.get_filings() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    import io
    import zipfile
    
    try:
        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        
        if not filings:
            return None
        
        client = SECClient()
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_f:
            for filing in filings:
                accession_number = filing.accession_number.replace("-", "")
                report_url = f"https://www.sec.gov/Archives/edgar/data/{filing.cik}/{accession_number}/Financial_Report.xlsx"
                
                try:
                    response = client.get(report_url)
                    if response.status_code == 200:
                        file_name = f"{ticker.upper()}_{filing.form}_{filing.filing_date}_Financial_Report.xlsx"
                        zip_f.writestr(file_name, response.content)
                except Exception:
                    continue
        
        if zip_f.namelist():
            zip_buffer.seek(0)
            return zip_buffer.getvalue()
        else:
            return None
    except Exception:
        return None


def format_metrics_efficient(extracted_metrics, target_metrics=None):
    """
    Converts extracted metrics into a clean, graphable DataFrame.
    
    DEPRECATED: Use financial4all.financials classes instead.
    """
    warnings.warn(
        "format_metrics_efficient is deprecated. Use financial4all.financials classes instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Convert extracted_metrics to FactSet format
    fact_set = FactSet.from_company_facts({"facts": {"us-gaap": extracted_metrics}})
    income_statement = IncomeStatement(fact_set)
    df = income_statement.to_dataframe()
    
    # Return in expected format: (dataframe, results_dict)
    return df, {}


def create_standard_income_statement(df, all_results):
    """
    Creates a standardized income statement from the fully solved financial data.
    
    DEPRECATED: Use financial4all.financials.IncomeStatement instead.
    """
    warnings.warn(
        "create_standard_income_statement is deprecated. Use financial4all.financials.IncomeStatement instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if df is None or df.empty:
        return pd.DataFrame(), {}, []
    
    # Filter to only income statement columns if possible
    income_cols = [
        "Revenue", "Cost of Revenue", "Gross Profit", "R&D Expenses",
        "SG&A Expenses", "Operating Expenses", "Operating Income",
        "Interest Income", "Interest Expense", "Income Before Taxes",
        "Taxes", "Net Income", "Basic EPS", "Diluted EPS"
    ]
    
    available_cols = [col for col in income_cols if col in df.columns]
    if available_cols:
        df_filtered = df[available_cols].copy()
    else:
        df_filtered = df.copy()
    
    return df_filtered, {}, list(df_filtered.columns)


def calculate_key_ratios(standard_is_df):
    """
    Calculates key profitability ratios from the standardized income statement.
    
    DEPRECATED: Use financial4all.financials.FinancialRatios instead.
    """
    warnings.warn(
        "calculate_key_ratios is deprecated. Use financial4all.financials.FinancialRatios instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if standard_is_df is None or standard_is_df.empty:
        return pd.DataFrame()
    
    ratios = pd.DataFrame(index=standard_is_df.index)
    
    # Gross Profit Margin
    if "Revenue" in standard_is_df.columns and "Gross Profit" in standard_is_df.columns:
        revenue = standard_is_df["Revenue"]
        gross_profit = standard_is_df["Gross Profit"]
        ratios["Gross Profit Margin"] = (
            gross_profit.divide(revenue).replace([np.inf, -np.inf], np.nan) * 100
        )
    
    # Operating Profit Margin
    if "Revenue" in standard_is_df.columns and "Operating Income" in standard_is_df.columns:
        revenue = standard_is_df["Revenue"]
        operating_income = standard_is_df["Operating Income"]
        ratios["Operating Profit Margin"] = (
            operating_income.divide(revenue).replace([np.inf, -np.inf], np.nan) * 100
        )
    
    # Net Profit Margin
    if "Revenue" in standard_is_df.columns and "Net Income" in standard_is_df.columns:
        revenue = standard_is_df["Revenue"]
        net_income = standard_is_df["Net Income"]
        ratios["Net Profit Margin"] = (
            net_income.divide(revenue).replace([np.inf, -np.inf], np.nan) * 100
        )
    
    return ratios.dropna(how='all')


# Helper function for backward compatibility
def find_col_with_units(base_metric: str, df_columns: List[str]) -> Optional[str]:
    """
    Finds the full column name in a DataFrame that corresponds to a base metric name.
    
    DEPRECATED: This function is kept for backward compatibility.
    """
    # Prioritize metrics with _USD unit as they are most common for income statements
    if base_metric + "_USD" in df_columns:
        return base_metric + "_USD"
    for col in df_columns:
        if col.startswith(base_metric + "_"):
            return col
    if base_metric in df_columns:
        return base_metric
    return None
