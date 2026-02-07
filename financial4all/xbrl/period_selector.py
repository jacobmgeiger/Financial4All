# financial4all/xbrl/period_selector.py
"""
Unified period selection system for XBRL statements.

This module provides intelligent period selection logic that handles
quarterly vs annual period selection, fiscal year end date matching,
and duration-based period classification.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from financial4all.xbrl.core import parse_date
from financial4all.xbrl.periods import (
    calculate_fiscal_alignment_score,
    filter_periods_by_document_end_date,
    sort_periods
)


def select_periods(
    xbrl_instance,
    statement_type: str
) -> List[Tuple[str, str]]:
    """
    Select appropriate periods for a statement type using unified logic.

    This function provides intelligent period selection that:
    - Handles quarterly vs annual period selection
    - Matches fiscal year end dates
    - Uses duration-based period classification
    - Filters periods by document end date

    Args:
        xbrl_instance: XBRL instance with context and entity information
        statement_type: Type of statement ('BalanceSheet', 'IncomeStatement', etc.)

    Returns:
        List of tuples with period keys and labels to display
    """
    periods_to_display = []

    # Get entity info and periods
    entity_info = getattr(xbrl_instance, 'entity_info', {})
    all_periods = getattr(xbrl_instance, 'reporting_periods', [])
    document_period_end_date = getattr(xbrl_instance, 'period_of_report', None)
    fiscal_period_focus = entity_info.get('fiscal_period')
    fiscal_year_end_month = entity_info.get('fiscal_year_end_month')
    fiscal_year_end_day = entity_info.get('fiscal_year_end_day')

    # Filter and sort periods by statement type
    if statement_type == 'BalanceSheet':
        instant_periods = [p for p in all_periods if p.get('type') == 'instant']
        instant_periods = filter_periods_by_document_end_date(
            instant_periods, document_period_end_date, 'instant'
        )
        instant_periods = sort_periods(instant_periods, 'instant')

        if instant_periods:
            # Take latest instant period
            current_period = instant_periods[0]
            periods_to_display.append((current_period['key'], current_period['label']))

            # Try to find appropriate comparison period
            try:
                current_date = parse_date(current_period['date'])

                # Use fiscal information if available for better matching
                if fiscal_year_end_month is not None and fiscal_year_end_day is not None:
                    # Check if this is a fiscal year end report
                    is_fiscal_year_end = False
                    if fiscal_period_focus == 'FY' or (
                        current_date.month == fiscal_year_end_month and
                        abs(current_date.day - fiscal_year_end_day) <= 7
                    ):
                        is_fiscal_year_end = True

                    if is_fiscal_year_end and entity_info.get('fiscal_year'):
                        # For fiscal year end, find the previous fiscal year end period
                        fiscal_year_focus = entity_info.get('fiscal_year')
                        prev_fiscal_year = int(fiscal_year_focus) - 1 if isinstance(
                            fiscal_year_focus, (int, str)) and str(fiscal_year_focus).isdigit() else current_date.year - 1

                        # Look for a comparable period from previous fiscal year
                        for period in instant_periods[1:]:
                            try:
                                period_date = parse_date(period['date'])
                                if (period_date.year == prev_fiscal_year and
                                    period_date.month == fiscal_year_end_month and
                                    abs(period_date.day - fiscal_year_end_day) <= 15):
                                    periods_to_display.append((period['key'], period['label']))
                                    break
                            except (ValueError, TypeError):
                                continue

                # If no appropriate period found yet, try generic date-based comparison
                if len(periods_to_display) == 1:
                    prev_year = current_date.year - 1
                    for period in instant_periods[1:]:
                        try:
                            period_date = parse_date(period['date'])
                            if period_date.year == prev_year:
                                periods_to_display.append((period['key'], period['label']))
                                break
                        except (ValueError, TypeError):
                            continue

            except (ValueError, TypeError):
                pass

    elif statement_type in ['IncomeStatement', 'CashFlowStatement']:
        duration_periods = [p for p in all_periods if p.get('type') == 'duration']
        duration_periods = filter_periods_by_document_end_date(
            duration_periods, document_period_end_date, 'duration'
        )
        duration_periods = sort_periods(duration_periods, 'duration')

        if duration_periods:
            # For annual reports, prioritize annual periods
            if fiscal_period_focus == 'FY':
                # Get fiscal year end information if available
                fiscal_year_end_month = entity_info.get('fiscal_year_end_month')
                fiscal_year_end_day = entity_info.get('fiscal_year_end_day')

                # Find all periods that are approximately a year long
                candidate_annual_periods = []
                for period in duration_periods:
                    try:
                        start_date = parse_date(period['start_date'])
                        end_date = parse_date(period['end_date'])
                        days = (end_date - start_date).days
                        # Strict check: Annual periods must be between 300 and 370 days
                        if 300 < days <= 370:
                            period_with_score = period.copy()
                            period_with_score['fiscal_alignment_score'] = 0
                            period_with_score['duration_days'] = days
                            candidate_annual_periods.append(period_with_score)
                    except (ValueError, TypeError):
                        continue

                # Score periods based on alignment with fiscal year pattern
                if fiscal_year_end_month is not None and fiscal_year_end_day is not None:
                    for period in candidate_annual_periods:
                        try:
                            end_date = parse_date(period['end_date'])
                            score = calculate_fiscal_alignment_score(
                                end_date, fiscal_year_end_month, fiscal_year_end_day
                            )
                            period['fiscal_alignment_score'] = score
                        except (ValueError, TypeError):
                            continue

                # Sort periods by fiscal alignment (higher score first) and then by recency
                annual_periods = sorted(
                    candidate_annual_periods,
                    key=lambda x: (x['fiscal_alignment_score'], x['end_date']),
                    reverse=True
                )

                if annual_periods:
                    # Take up to 3 best matching annual periods
                    for period in annual_periods[:3]:
                        periods_to_display.append((period['key'], period['label']))
                    return periods_to_display

            # For quarterly reports, apply intelligent period selection
            else:
                # Categorize periods by duration
                quarterly_periods = []
                ytd_periods = []
                annual_periods = []

                current_year = None
                if document_period_end_date:
                    try:
                        current_year = parse_date(document_period_end_date).year
                    except (ValueError, TypeError):
                        pass

                # Categorize all duration periods by their length
                for period in duration_periods:
                    try:
                        start_date = parse_date(period['start_date'])
                        end_date = parse_date(period['end_date'])
                        days = (end_date - start_date).days

                        # Skip single-day or very short periods
                        if days < 30:
                            continue

                        # Categorize by duration with stricter checks
                        if 80 <= days <= 100:  # Quarterly period
                            period['period_type'] = 'quarterly'
                            period['days'] = days
                            quarterly_periods.append(period)
                        elif 170 <= days <= 190:  # Semi-annual/YTD for Q2
                            period['period_type'] = 'semi-annual'
                            period['days'] = days
                            ytd_periods.append(period)
                        elif 260 <= days <= 280:  # YTD for Q3
                            period['period_type'] = 'three-quarters'
                            period['days'] = days
                            ytd_periods.append(period)
                        elif 300 < days <= 370:  # Annual period
                            period['period_type'] = 'annual'
                            period['days'] = days
                            annual_periods.append(period)
                    except (ValueError, TypeError):
                        continue

                # Build the optimal set of periods for quarterly reporting
                selected_periods = []

                # 1. Add the most recent quarterly period (current quarter)
                if quarterly_periods:
                    recent_quarterly = quarterly_periods[0]
                    selected_periods.append(recent_quarterly)

                    # Try to find the same quarter from previous year for comparison
                    if current_year:
                        for qp in quarterly_periods[1:]:
                            try:
                                qp_end = parse_date(qp['end_date'])
                                recent_end = parse_date(recent_quarterly['end_date'])
                                if (qp_end.year == current_year - 1 and
                                    qp_end.month == recent_end.month and
                                    abs(qp_end.day - recent_end.day) <= 15):
                                    selected_periods.append(qp)
                                    break
                            except (ValueError, TypeError):
                                continue

                # 2. Add the most recent YTD period if available
                if ytd_periods:
                    selected_periods.append(ytd_periods[0])

                # 3. If we don't have enough periods yet, add more quarterly periods
                if len(selected_periods) < 3:
                    for period in quarterly_periods:
                        if period not in selected_periods and len(selected_periods) < 3:
                            selected_periods.append(period)

                # Convert selected periods to display format
                for period in selected_periods[:3]:
                    periods_to_display.append((period['key'], period['label']))

    return periods_to_display
