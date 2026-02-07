# financial4all/xbrl/periods.py
"""
Period handling and validation for XBRL data.

This module provides functionality for handling XBRL periods including
instant periods, duration periods, and comparative periods.
"""

from enum import Enum
from datetime import datetime, date
from typing import Optional, Union, Tuple
from dataclasses import dataclass
import calendar


class PeriodType(Enum):
    """XBRL period types."""
    INSTANT = "instant"  # Point in time (e.g., balance sheet date)
    DURATION = "duration"  # Time period (e.g., fiscal year, quarter)
    FOREVER = "forever"  # All time (rare)


@dataclass
class Period:
    """
    Represents an XBRL period.
    
    Attributes:
        start: Start date (None for instant periods)
        end: End date (required for all periods)
        period_type: Type of period (instant, duration, forever)
    """
    
    end: Union[datetime, date, str]
    start: Optional[Union[datetime, date, str]] = None
    period_type: PeriodType = PeriodType.DURATION
    
    def __post_init__(self):
        """Normalize dates and determine period type."""
        # Convert string dates to date objects
        if isinstance(self.end, str):
            self.end = self._parse_date(self.end)
        if isinstance(self.start, str) and self.start:
            self.start = self._parse_date(self.start)
        
        # Determine period type
        if self.start is None:
            self.period_type = PeriodType.INSTANT
        elif self.start == self.end:
            self.period_type = PeriodType.INSTANT
        else:
            self.period_type = PeriodType.DURATION
    
    @staticmethod
    def _parse_date(date_str: str) -> date:
        """Parse date string to date object."""
        # Try common formats
        for fmt in ["%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"]:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Unable to parse date: {date_str}")
    
    @classmethod
    def from_xbrl_dict(cls, period_dict: dict) -> "Period":
        """
        Create Period from XBRL period dictionary.
        
        Args:
            period_dict: Dictionary with 'start' and 'end' keys
            
        Returns:
            Period object
        """
        start = period_dict.get("start")
        end = period_dict.get("end")
        
        if not end:
            raise ValueError("Period must have an 'end' date")
        
        return cls(start=start, end=end)
    
    def is_annual(self) -> bool:
        """Check if period is approximately one year."""
        if self.start is None:
            return False
        
        start_date = self.start if isinstance(self.start, date) else datetime.strptime(str(self.start), "%Y-%m-%d").date()
        end_date = self.end if isinstance(self.end, date) else datetime.strptime(str(self.end), "%Y-%m-%d").date()
        
        days = (end_date - start_date).days
        # Annual periods are typically 360-370 days
        return 360 <= days <= 370
    
    def is_quarterly(self) -> bool:
        """Check if period is approximately one quarter."""
        if self.start is None:
            return False
        
        start_date = self.start if isinstance(self.start, date) else datetime.strptime(str(self.start), "%Y-%m-%d").date()
        end_date = self.end if isinstance(self.end, date) else datetime.strptime(str(self.end), "%Y-%m-%d").date()
        
        days = (end_date - start_date).days
        # Quarterly periods are typically 88-92 days
        return 88 <= days <= 92
    
    def __repr__(self) -> str:
        """String representation of Period."""
        if self.start:
            return f"Period(start={self.start}, end={self.end}, type={self.period_type.value})"
        else:
            return f"Period(end={self.end}, type={self.period_type.value})"


def classify_fiscal_period(
    period: Period,
    fiscal_year_end_month: Optional[int] = None,
    fiscal_year_end_day: Optional[int] = None
) -> Tuple[Optional[int], Optional[str]]:
    """
    Classify a period as a fiscal year or fiscal quarter.
    
    Args:
        period: Period to classify
        fiscal_year_end_month: Fiscal year end month (1-12), defaults to 12 (December)
        fiscal_year_end_day: Fiscal year end day (1-31), defaults to 31
        
    Returns:
        Tuple of (fiscal_year, fiscal_period) where:
        - fiscal_year: Fiscal year (e.g., 2024)
        - fiscal_period: "FY" for fiscal year, "Q1"-"Q4" for quarters, or None if cannot classify
    """
    if period.period_type != PeriodType.DURATION or period.start is None:
        return None, None
    
    # Default to calendar year end (December 31) if not specified
    if fiscal_year_end_month is None:
        fiscal_year_end_month = 12
    if fiscal_year_end_day is None:
        fiscal_year_end_day = 31
    
    # Normalize dates
    start_date = period.start if isinstance(period.start, date) else datetime.strptime(str(period.start), "%Y-%m-%d").date()
    end_date = period.end if isinstance(period.end, date) else datetime.strptime(str(period.end), "%Y-%m-%d").date()
    
    days = (end_date - start_date).days
    
    # Determine fiscal year from end date
    fiscal_year = _fiscal_year_for_date(end_date, fiscal_year_end_month, fiscal_year_end_day)
    
    # Classify period type based on duration
    if 350 <= days <= 380:
        # Annual period (fiscal year)
        return fiscal_year, "FY"
    elif 85 <= days <= 95:
        # Quarterly period
        quarter = _quarter_for_date(end_date, fiscal_year_end_month)
        return fiscal_year, quarter
    else:
        # Cannot classify (might be partial period or other)
        return fiscal_year, None


def _fiscal_year_for_date(d: date, fy_end_month: int, fy_end_day: int) -> int:
    """
    Determine the fiscal year a date belongs to.
    
    The fiscal year is named by the calendar year in which it ends.
    For example, AAPL's fiscal year ending Sep 2024 is FY2024.
    A date in Dec 2024 (after Sep end) belongs to FY2025.
    
    Args:
        d: Date to classify
        fy_end_month: Fiscal year end month (1-12)
        fy_end_day: Fiscal year end day (1-31)
        
    Returns:
        Fiscal year (integer)
    """
    # Clamp the day to the max valid day for that month
    max_day = calendar.monthrange(d.year, fy_end_month)[1]
    fy_end_day_clamped = min(fy_end_day, max_day)
    
    # Build the fiscal year end date for the same calendar year as d
    fy_end_date = d.replace(month=fy_end_month, day=fy_end_day_clamped)
    
    # If d is after the fiscal year end date, it belongs to the next fiscal year
    if d > fy_end_date:
        return d.year + 1
    return d.year


def _quarter_for_date(end_date: date, fy_end_month: int) -> str:
    """
    Determine the fiscal quarter (Q1-Q4) for a quarterly period end date.
    
    Quarters are counted from the start of the fiscal year.
    For AAPL (FY end Sep): Q1=Oct-Dec, Q2=Jan-Mar, Q3=Apr-Jun, Q4=Jul-Sep.
    
    Args:
        end_date: Period end date
        fy_end_month: Fiscal year end month (1-12)
        
    Returns:
        Quarter string ("Q1", "Q2", "Q3", or "Q4")
    """
    # Calculate months after fiscal year end determines the quarter
    # Month offset: how many months past the FY end month
    month_offset = (end_date.month - fy_end_month - 1) % 12
    quarter = (month_offset // 3) + 1
    return f"Q{quarter}"


def classify_duration(days: int) -> str:
    """
    Classify a duration in days as a period type.
    
    Args:
        days: Number of days in the period
        
    Returns:
        Period type description ("Annual", "Quarterly", "Partial", etc.)
    """
    if 350 <= days <= 380:
        return "Annual"
    elif 85 <= days <= 95:
        return "Quarterly"
    elif days < 85:
        return "Partial"
    elif days > 380:
        return "Multi-year"
    else:
        return "Other"
