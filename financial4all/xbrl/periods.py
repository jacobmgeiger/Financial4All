# financial4all/xbrl/periods.py
"""
Period handling and validation for XBRL data.

This module provides functionality for handling XBRL periods including
instant periods, duration periods, and comparative periods.
"""

from enum import Enum
from datetime import datetime, date
from typing import Optional, Union
from dataclasses import dataclass


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
