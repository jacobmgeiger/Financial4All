# financial4all/xbrl/period_data_check.py
"""
Period data validation and checking.

This module provides functions for validating period data consistency,
checking date ranges, detecting anomalies, and validating fiscal year alignment.
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any

from financial4all.xbrl.core import parse_date
from financial4all.xbrl.periods import calculate_fiscal_alignment_score


def check_period_data(periods: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate period data consistency.
    
    Args:
        periods: List of period dictionaries
        
    Returns:
        Dictionary with validation results and issues
    """
    issues = []
    
    for i, period in enumerate(periods):
        period_type = period.get('type')
        
        if period_type == 'instant':
            date_str = period.get('date')
            if date_str:
                try:
                    parse_date(date_str)
                except ValueError as e:
                    issues.append({
                        'period_index': i,
                        'period_key': period.get('key'),
                        'issue': f"Invalid instant date: {e}",
                        'severity': 'error'
                    })
        
        elif period_type == 'duration':
            start_str = period.get('start_date')
            end_str = period.get('end_date')
            
            if start_str and end_str:
                try:
                    start_date = parse_date(start_str)
                    end_date = parse_date(end_str)
                    
                    if end_date < start_date:
                        issues.append({
                            'period_index': i,
                            'period_key': period.get('key'),
                            'issue': f"End date ({end_str}) is before start date ({start_str})",
                            'severity': 'error'
                        })
                    
                    # Check for reasonable duration
                    days = (end_date - start_date).days
                    if days < 0:
                        issues.append({
                            'period_index': i,
                            'period_key': period.get('key'),
                            'issue': f"Negative duration: {days} days",
                            'severity': 'error'
                        })
                    elif days > 400:
                        issues.append({
                            'period_index': i,
                            'period_key': period.get('key'),
                            'issue': f"Unusually long duration: {days} days",
                            'severity': 'warning'
                        })
                        
                except ValueError as e:
                    issues.append({
                        'period_index': i,
                        'period_key': period.get('key'),
                        'issue': f"Invalid date format: {e}",
                        'severity': 'error'
                    })
    
    return {
        'is_valid': len([i for i in issues if i['severity'] == 'error']) == 0,
        'issues': issues,
        'total_periods': len(periods),
        'error_count': len([i for i in issues if i['severity'] == 'error']),
        'warning_count': len([i for i in issues if i['severity'] == 'warning'])
    }


def validate_period_ranges(periods: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check period date ranges for consistency.
    
    Args:
        periods: List of period dictionaries
        
    Returns:
        Dictionary with validation results
    """
    instant_periods = [p for p in periods if p.get('type') == 'instant']
    duration_periods = [p for p in periods if p.get('type') == 'duration']
    
    results = {
        'instant_periods': len(instant_periods),
        'duration_periods': len(duration_periods),
        'overlaps': [],
        'gaps': []
    }
    
    # Check for overlapping duration periods
    for i, period1 in enumerate(duration_periods):
        for j, period2 in enumerate(duration_periods[i+1:], start=i+1):
            try:
                start1 = parse_date(period1.get('start_date', ''))
                end1 = parse_date(period1.get('end_date', ''))
                start2 = parse_date(period2.get('start_date', ''))
                end2 = parse_date(period2.get('end_date', ''))
                
                # Check for overlap
                if not (end1 < start2 or end2 < start1):
                    results['overlaps'].append({
                        'period1': period1.get('key'),
                        'period2': period2.get('key')
                    })
            except (ValueError, TypeError):
                continue
    
    return results


def detect_period_anomalies(periods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find unusual period patterns.
    
    Args:
        periods: List of period dictionaries
        
    Returns:
        List of detected anomalies
    """
    anomalies = []
    
    duration_periods = [p for p in periods if p.get('type') == 'duration']
    
    # Group periods by approximate duration
    quarterly = []
    annual = []
    other = []
    
    for period in duration_periods:
        try:
            start_date = parse_date(period.get('start_date', ''))
            end_date = parse_date(period.get('end_date', ''))
            days = (end_date - start_date).days
            
            if 80 <= days <= 100:
                quarterly.append(period)
            elif 350 <= days <= 380:
                annual.append(period)
            else:
                other.append(period)
        except (ValueError, TypeError):
            continue
    
    # Detect anomalies
    if len(quarterly) > 0 and len(annual) > 0:
        # Check if quarterly periods are consistent
        if len(quarterly) > 1:
            # Check spacing between quarters
            quarterly_sorted = sorted(quarterly, key=lambda p: parse_date(p.get('end_date', '')))
            for i in range(len(quarterly_sorted) - 1):
                end1 = parse_date(quarterly_sorted[i].get('end_date', ''))
                start2 = parse_date(quarterly_sorted[i+1].get('start_date', ''))
                gap = (start2 - end1).days
                
                if gap > 5:  # More than 5 days gap
                    anomalies.append({
                        'type': 'quarterly_gap',
                        'period1': quarterly_sorted[i].get('key'),
                        'period2': quarterly_sorted[i+1].get('key'),
                        'gap_days': gap
                    })
    
    # Detect unusual durations
    for period in other:
        try:
            start_date = parse_date(period.get('start_date', ''))
            end_date = parse_date(period.get('end_date', ''))
            days = (end_date - start_date).days
            
            if days > 400:
                anomalies.append({
                    'type': 'unusually_long',
                    'period': period.get('key'),
                    'days': days
                })
            elif 0 < days < 30:
                anomalies.append({
                    'type': 'unusually_short',
                    'period': period.get('key'),
                    'days': days
                })
        except (ValueError, TypeError):
            continue
    
    return anomalies


def validate_fiscal_alignment(
    periods: List[Dict[str, Any]],
    fiscal_year_end_month: int,
    fiscal_year_end_day: int
) -> Dict[str, Any]:
    """
    Check fiscal year alignment for periods.
    
    Args:
        periods: List of period dictionaries
        fiscal_year_end_month: Fiscal year end month (1-12)
        fiscal_year_end_day: Fiscal year end day (1-31)
        
    Returns:
        Dictionary with alignment scores and results
    """
    alignment_results = []
    
    for period in periods:
        period_type = period.get('type')
        
        if period_type == 'instant':
            date_str = period.get('date')
            if date_str:
                try:
                    period_date = parse_date(date_str)
                    score = calculate_fiscal_alignment_score(
                        period_date, fiscal_year_end_month, fiscal_year_end_day
                    )
                    alignment_results.append({
                        'period_key': period.get('key'),
                        'date': date_str,
                        'alignment_score': score,
                        'well_aligned': score >= 75
                    })
                except ValueError:
                    continue
        
        elif period_type == 'duration':
            end_str = period.get('end_date')
            if end_str:
                try:
                    end_date = parse_date(end_str)
                    score = calculate_fiscal_alignment_score(
                        end_date, fiscal_year_end_month, fiscal_year_end_day
                    )
                    alignment_results.append({
                        'period_key': period.get('key'),
                        'end_date': end_str,
                        'alignment_score': score,
                        'well_aligned': score >= 75
                    })
                except ValueError:
                    continue
    
    well_aligned_count = sum(1 for r in alignment_results if r['well_aligned'])
    
    return {
        'total_periods': len(alignment_results),
        'well_aligned_count': well_aligned_count,
        'alignment_percentage': (well_aligned_count / len(alignment_results) * 100) if alignment_results else 0,
        'results': alignment_results
    }
