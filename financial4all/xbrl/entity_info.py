# financial4all/xbrl/entity_info.py
"""
Entity information extraction from DEI (Document and Entity Information) facts.

This module provides functionality for extracting entity metadata from XBRL DEI facts,
including fiscal year end dates, document types, and reporting periods.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime, date
import calendar

from financial4all.core import log


@dataclass
class EntityInfo:
    """
    Entity information extracted from DEI facts.
    
    Attributes:
        entity_name: Name of the entity
        ticker: Trading symbol
        cik: Central Index Key
        document_type: Document type (e.g., "10-K", "10-Q")
        reporting_end_date: Most recent reporting period end date
        document_period_end_date: Document period end date from DEI
        fiscal_year: Fiscal year focus
        fiscal_period: Fiscal period focus (FY, Q1-Q4)
        fiscal_year_end_month: Fiscal year end month (1-12)
        fiscal_year_end_day: Fiscal year end day (1-31)
        annual_report: Whether this is an annual report
        quarterly_report: Whether this is a quarterly report
        amendment: Whether this is an amendment
    """
    
    entity_name: Optional[str] = None
    ticker: Optional[str] = None
    cik: Optional[str] = None
    document_type: Optional[str] = None
    reporting_end_date: Optional[date] = None
    document_period_end_date: Optional[str] = None
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    fiscal_year_end_month: Optional[int] = None
    fiscal_year_end_day: Optional[int] = None
    annual_report: bool = False
    quarterly_report: bool = False
    amendment: bool = False
    
    def __repr__(self) -> str:
        """String representation of EntityInfo."""
        return (
            f"EntityInfo(entity_name={self.entity_name}, ticker={self.ticker}, "
            f"cik={self.cik}, fiscal_year_end={self.fiscal_year_end_month}/{self.fiscal_year_end_day})"
        )


def extract_dei_facts(company_facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract DEI (Document and Entity Information) facts from company facts API response.
    
    Args:
        company_facts: Dictionary from SEC company facts API
        
    Returns:
        Dictionary mapping DEI concept names to their values
    """
    dei_facts = {}
    
    try:
        facts = company_facts.get("facts", {})
        dei_data = facts.get("dei", {})
        
        if not dei_data:
            log.debug("No DEI facts found in company facts")
            return dei_facts
        
        # Extract all DEI facts
        for concept, concept_data in dei_data.items():
            units = concept_data.get("units", {})
            
            # Get the most recent fact value
            # DEI facts are typically instant facts, so we want the most recent one
            most_recent_value = None
            most_recent_date = None
            
            for unit, entries in units.items():
                for entry in entries:
                    filed_date = entry.get("filed")
                    if filed_date:
                        try:
                            entry_date = datetime.fromisoformat(filed_date.replace('Z', '+00:00')).date()
                            if most_recent_date is None or entry_date > most_recent_date:
                                most_recent_date = entry_date
                                most_recent_value = entry.get("val")
                        except (ValueError, AttributeError):
                            continue
            
            if most_recent_value is not None:
                # Normalize concept name (remove namespace prefix if present)
                concept_normalized = concept.replace("dei:", "").replace("dei_", "")
                dei_facts[concept_normalized] = most_recent_value
        
        log.debug(f"Extracted {len(dei_facts)} DEI facts")
        
    except Exception as e:
        log.warning(f"Error extracting DEI facts: {e}")
    
    return dei_facts


def extract_dei_facts_from_xbrl(xbrl: Any) -> Dict[str, Any]:
    """
    Extract DEI (Document and Entity Information) facts from a parsed XBRL instance.

    Scans XBRL._facts for element_ids starting with 'dei_' and builds a dict
    compatible with build_entity_info().

    Args:
        xbrl: XBRL instance with _facts and contexts

    Returns:
        Dictionary mapping DEI concept names (without dei_ prefix) to values
    """
    dei_facts: Dict[str, Any] = {}

    if not hasattr(xbrl, "_facts") or not xbrl._facts:
        return dei_facts

    for fact_key, model_fact in xbrl._facts.items():
        element_id = getattr(model_fact, "element_id", "") or ""
        if not element_id.lower().startswith("dei_"):
            continue

        concept_normalized = element_id.replace("dei_", "", 1).replace("dei:", "")
        value = getattr(model_fact, "numeric_value", None) or getattr(
            model_fact, "value", None
        )
        if value is not None:
            dei_facts[concept_normalized] = value

    return dei_facts


def build_entity_info(dei_facts: Dict[str, Any], cik: Optional[str] = None) -> EntityInfo:
    """
    Build EntityInfo object from DEI facts.
    
    Args:
        dei_facts: Dictionary of DEI facts (concept -> value)
        cik: Optional CIK to include in entity info
        
    Returns:
        EntityInfo object with extracted information
    """
    entity_info = EntityInfo(cik=cik)
    
    # Helper function to get DEI fact value with fallback options
    def get_dei(*names: str) -> Optional[str]:
        """Get DEI fact value, trying multiple possible concept names."""
        for name in names:
            value = dei_facts.get(name)
            if value:
                return str(value)
        return None
    
    # Extract basic entity information
    entity_info.entity_name = get_dei("EntityRegistrantName", "EntityName")
    entity_info.ticker = get_dei("TradingSymbol", "TickerSymbol")
    entity_info.document_type = get_dei("DocumentType")
    entity_info.document_period_end_date = get_dei("DocumentPeriodEndDate")
    
    # Extract fiscal year information
    fiscal_year_str = get_dei("DocumentFiscalYearFocus", "FiscalYearFocus", "FiscalYear")
    if fiscal_year_str:
        try:
            entity_info.fiscal_year = int(fiscal_year_str)
        except (ValueError, TypeError):
            pass
    
    # Extract fiscal period
    entity_info.fiscal_period = get_dei("DocumentFiscalPeriodFocus", "FiscalPeriodFocus")
    
    # Extract fiscal year end date (month/day)
    fye_str = get_dei("CurrentFiscalYearEndDate", "FiscalYearEnd")
    if fye_str:
        try:
            # Parse format like "--12-31" or "12-31"
            fye_clean = fye_str.lstrip("-")
            if "-" in fye_clean:
                month_str, day_str = fye_clean.split("-", 1)
                if month_str.isdigit() and day_str.isdigit():
                    month = int(month_str)
                    day = int(day_str)
                    # Validate month and day
                    if 1 <= month <= 12:
                        # Clamp day to valid range for the month
                        max_day = calendar.monthrange(2000, month)[1]  # Use 2000 as reference year
                        day = min(day, max_day)
                        entity_info.fiscal_year_end_month = month
                        entity_info.fiscal_year_end_day = day
        except (ValueError, AttributeError):
            pass
    
    # Determine report type flags
    doc_type = entity_info.document_type or ""
    entity_info.annual_report = doc_type == "10-K"
    entity_info.quarterly_report = doc_type == "10-Q"
    entity_info.amendment = "/A" in doc_type
    
    # Extract reporting end date from document period end date
    if entity_info.document_period_end_date:
        try:
            entity_info.reporting_end_date = datetime.strptime(
                entity_info.document_period_end_date, "%Y-%m-%d"
            ).date()
        except (ValueError, TypeError):
            pass
    
    return entity_info
