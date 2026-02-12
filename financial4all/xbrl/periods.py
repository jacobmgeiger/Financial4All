# financial4all/xbrl/periods.py
"""
XBRL period representation and helpers.

Period and PeriodType represent instant vs duration; from_xbrl_dict() builds
from SEC API start/end. Helpers: is_annual(), is_quarterly(), classify_fiscal_period,
filter/sort period lists, and determine_periods_to_display for UI. Used by
FactSet, statement resolution, and period selection.
"""

from enum import Enum
from datetime import datetime, date
from typing import Optional, Union, Tuple, List, Dict, Any
from dataclasses import dataclass
import calendar

from financial4all.xbrl.core import parse_date


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
        """
        Check if period is approximately one year.
        
        Uses a more lenient range (330-400 days) to capture annual periods
        that might have slight variations due to leap years, fiscal year adjustments,
        or reporting calendar differences.
        """
        if self.start is None:
            return False
        
        try:
            start_date = self.start if isinstance(self.start, date) else datetime.strptime(str(self.start), "%Y-%m-%d").date()
            end_date = self.end if isinstance(self.end, date) else datetime.strptime(str(self.end), "%Y-%m-%d").date()
            
            days = (end_date - start_date).days
            # Annual periods are typically 360-370 days, but allow wider range (330-400)
            # to capture variations due to leap years, fiscal year adjustments, etc.
            return 330 <= days <= 400
        except (ValueError, TypeError, AttributeError):
            return False
    
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


def calculate_fiscal_alignment_score(end_date: date, fiscal_month: int, fiscal_day: int) -> int:
    """
    Calculate how well a date aligns with fiscal year end.
    
    Args:
        end_date: Period end date
        fiscal_month: Fiscal year end month
        fiscal_day: Fiscal year end day
        
    Returns:
        Score from 0-100 indicating alignment quality
    """
    if end_date.month == fiscal_month and end_date.day == fiscal_day:
        return 100
    if end_date.month == fiscal_month and abs(end_date.day - fiscal_day) <= 15:
        return 75
    if abs(end_date.month - fiscal_month) <= 1 and abs(end_date.day - fiscal_day) <= 15:
        return 50
    return 0


def filter_periods_by_document_end_date(periods: List[Dict], document_period_end_date: str, period_type: str) -> List[Dict]:
    """
    Filter periods to only include those that end on or before the document period end date.
    
    Args:
        periods: List of period dictionaries
        document_period_end_date: Document period end date string (YYYY-MM-DD)
        period_type: Period type ('instant' or 'duration')
        
    Returns:
        Filtered list of periods
    """
    if not document_period_end_date:
        return periods

    try:
        doc_end_date = parse_date(document_period_end_date)
    except (ValueError, TypeError):
        # If we can't parse the document end date, return all periods
        return periods

    filtered_periods = []
    for period in periods:
        try:
            if period_type == 'instant':
                period_date = parse_date(period['date'])
                if period_date <= doc_end_date:
                    filtered_periods.append(period)
            else:  # duration
                period_end_date = parse_date(period['end_date'])
                if period_end_date <= doc_end_date:
                    filtered_periods.append(period)
        except (ValueError, TypeError):
            # If we can't parse the period date, include it to be safe
            filtered_periods.append(period)

    return filtered_periods


def sort_periods(periods: List[Dict], period_type: str) -> List[Dict]:
    """
    Sort periods by date, with most recent first.
    
    Args:
        periods: List of period dictionaries
        period_type: Period type ('instant' or 'duration')
        
    Returns:
        Sorted list of periods
    """
    if period_type == 'instant':
        return sorted(periods, key=lambda x: x.get('date', ''), reverse=True)
    return sorted(periods, key=lambda x: (x.get('end_date', ''), x.get('start_date', '')), reverse=True)


# Configuration for different statement types
STATEMENT_TYPE_CONFIG = {
    'BalanceSheet': {
        'period_type': 'instant',
        'max_periods': 3,
        'allow_annual_comparison': True,
        'views': [
            {
                'name': 'Three Recent Periods',
                'description': 'Shows three most recent reporting periods',
                'max_periods': 3,
                'requires_min_periods': 3
            },
            {
                'name': 'Current vs. Previous Period',
                'description': 'Shows the current period and the previous period',
                'max_periods': 2,
                'requires_min_periods': 1
            },
        ]
    },
    'IncomeStatement': {
        'period_type': 'duration',
        'max_periods': 3,
        'allow_annual_comparison': True,
        'views': [
            {
                'name': 'Three Recent Periods',
                'description': 'Shows three most recent reporting periods',
                'max_periods': 3,
                'requires_min_periods': 3
            },
            {
                'name': 'Annual Comparison',
                'description': 'Shows annual periods for comparison',
                'max_periods': 3,
                'requires_min_periods': 3,
                'annual_only': True
            }
        ]
    },
}


def get_period_views(xbrl_instance, statement_type: str) -> List[Dict[str, Any]]:
    """
    Get available period views for a statement type.

    Args:
        xbrl_instance: XBRL instance with context and entity information
        statement_type: Type of statement to get period views for

    Returns:
        List of period view options with name, description, and period keys
    """
    period_views = []

    # Get statement configuration
    config = STATEMENT_TYPE_CONFIG.get(statement_type)
    if not config:
        return period_views

    # Get useful entity info for period selection
    entity_info = getattr(xbrl_instance, 'entity_info', {})
    fiscal_period_focus = entity_info.get('fiscal_period')
    annual_report = fiscal_period_focus == 'FY'

    # Get all periods
    all_periods = getattr(xbrl_instance, 'reporting_periods', [])
    document_period_end_date = getattr(xbrl_instance, 'period_of_report', None)

    # Filter and sort periods by type
    period_type = config['period_type']
    periods = [p for p in all_periods if p.get('type') == period_type]
    periods = filter_periods_by_document_end_date(periods, document_period_end_date, period_type)
    periods = sort_periods(periods, period_type)

    # Generate views based on configuration
    for view_config in config.get('views', []):
        if view_config.get('annual_only') and not annual_report:
            continue

        if len(periods) >= view_config['requires_min_periods']:
            max_periods = min(view_config['max_periods'], len(periods))
            period_keys = [p['key'] for p in periods[:max_periods]]
            
            period_views.append({
                'name': view_config['name'],
                'description': view_config['description'],
                'period_keys': period_keys
            })

    return period_views


def determine_periods_to_display(
    xbrl_instance,
    statement_type: str,
    period_filter: Optional[str] = None,
    period_view: Optional[str] = None
) -> List[Tuple[str, str]]:
    """
    Determine which periods should be displayed for a statement.

    Uses smart period selection, which balances investor needs
    with data availability for optimal financial analysis.

    Args:
        xbrl_instance: XBRL instance with context and entity information
        statement_type: Type of statement ('BalanceSheet', 'IncomeStatement', etc.)
        period_filter: Optional period key to filter by specific reporting period
        period_view: Optional name of a predefined period view

    Returns:
        List of tuples with period keys and labels to display
    """
    periods_to_display = []

    # If a specific period is requested, use only that
    if period_filter:
        all_periods = getattr(xbrl_instance, 'reporting_periods', [])
        for period in all_periods:
            if period.get('key') == period_filter:
                periods_to_display.append((period_filter, period.get('label', period_filter)))
                break
        return periods_to_display

    # If a period view is specified, use that
    if period_view:
        available_views = get_period_views(xbrl_instance, statement_type)
        matching_view = next((view for view in available_views if view['name'] == period_view), None)

        if matching_view:
            all_periods = getattr(xbrl_instance, 'reporting_periods', [])
            for period_key in matching_view['period_keys']:
                for period in all_periods:
                    if period.get('key') == period_key:
                        periods_to_display.append((period_key, period.get('label', period_key)))
                        break
            return periods_to_display

    # Use unified period selection system
    try:
        from financial4all.xbrl.period_selector import select_periods
        return select_periods(xbrl_instance, statement_type)
    except Exception:
        # Fallback to basic selection
        all_periods = getattr(xbrl_instance, 'reporting_periods', [])
        if all_periods:
            # Take first period as default
            first_period = all_periods[0]
            periods_to_display.append((first_period.get('key', ''), first_period.get('label', '')))
        return periods_to_display
