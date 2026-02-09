# financial4all/analysis/validators.py
"""
Financial statement validation and cross-validation framework.

This module provides validation rules to ensure financial statement data
is consistent, reasonable, and follows accounting relationships.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    metric: str
    period: str
    severity: ValidationSeverity
    message: str
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    suggested_correction: Optional[float] = None


class FinancialStatementValidator:
    """
    Validates financial statement data for consistency and reasonableness.
    
    Provides cross-validation between income statement, balance sheet, and cash flow.
    Includes statistical and rule-based outlier detection.
    """
    
    # Validation thresholds (configurable)
    CAPEX_PCT_OF_SALES_MIN = 0.0  # Minimum expected CapEx % of Sales
    CAPEX_PCT_OF_SALES_MAX = 0.30  # Maximum expected CapEx % of Sales (30%)
    CAPEX_PCT_OF_SALES_OUTLIER = 0.50  # Flag as outlier if > 50%
    CAPEX_TO_DA_MIN = 0.5  # Minimum CapEx/D&A ratio
    CAPEX_TO_DA_MAX = 5.0  # Maximum CapEx/D&A ratio (5x)
    CAPEX_TO_PPE_MIN = 0.05  # Minimum CapEx/PP&E ratio (5%)
    CAPEX_TO_PPE_MAX = 0.20  # Maximum CapEx/PP&E ratio (20%)
    
    # Outlier detection thresholds
    STATISTICAL_OUTLIER_STD_DEVS = 3.0  # Standard deviations for statistical outlier detection
    STATISTICAL_OUTLIER_MIN_SAMPLES = 3  # Minimum samples needed for statistical detection
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator with optional configuration.
        
        Args:
            config: Optional dictionary to override default thresholds
        """
        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def validate_capex(
        self,
        capex_series: pd.Series,
        revenue_series: Optional[pd.Series] = None,
        da_series: Optional[pd.Series] = None,
        ppe_series: Optional[pd.Series] = None,
    ) -> List[ValidationIssue]:
        """
        Validate CapEx values for reasonableness.
        
        Includes checks for:
        - CapEx % of Sales (detects acquisitions if > 50%)
        - CapEx/D&A ratio (detects acquisitions if > 10x)
        - CapEx/PP&E ratio
        - CapEx vs PP&E change (detects acquisitions if > 3x PP&E change)
        
        Args:
            capex_series: Series of CapEx values indexed by period
            revenue_series: Optional series of Revenue values for % of Sales validation
            da_series: Optional series of D&A values for CapEx/D&A ratio validation
            ppe_series: Optional series of PP&E values for CapEx/PP&E ratio and PP&E change validation
            
        Returns:
            List of validation issues found
        """
        issues = []
        MAX_CAPEX_TO_PPE_CHANGE_RATIO = 3.0  # CapEx shouldn't exceed 3x PP&E change (indicates acquisitions)
        MAX_CAPEX_TO_DA_RATIO = 10.0  # CapEx shouldn't exceed 10x D&A (indicates acquisitions)
        
        for period, capex in capex_series.items():
            if pd.isna(capex) or capex == 0:
                continue
            
            period_str = str(period)
            
            # CapEx in cash flow statements is typically negative (cash outflow convention)
            # Use absolute value for validation checks
            capex_abs = abs(capex)
            
            # Check for sign inconsistency: if most values are negative (cash flow convention)
            # but this one is positive, it might indicate an error
            # Note: We'll check this in outlier detection, not here
            
            # Validate CapEx % of Sales (detects acquisitions)
            if revenue_series is not None and period in revenue_series.index:
                revenue = revenue_series[period]
                if not pd.isna(revenue) and revenue > 0:
                    capex_pct = abs(capex) / revenue
                    
                    if capex_pct > self.CAPEX_PCT_OF_SALES_OUTLIER:
                        issues.append(ValidationIssue(
                            metric="CapEx % of Sales",
                            period=period_str,
                            severity=ValidationSeverity.ERROR,
                            message=f"CapEx % of Sales is {capex_pct*100:.2f}%, exceeds outlier threshold ({self.CAPEX_PCT_OF_SALES_OUTLIER*100}%). May include business acquisitions.",
                            actual_value=capex_pct,
                            expected_value=self.CAPEX_PCT_OF_SALES_MAX
                        ))
                    elif capex_pct > self.CAPEX_PCT_OF_SALES_MAX:
                        issues.append(ValidationIssue(
                            metric="CapEx % of Sales",
                            period=period_str,
                            severity=ValidationSeverity.WARNING,
                            message=f"CapEx % of Sales is {capex_pct*100:.2f}%, unusually high (typical range: {self.CAPEX_PCT_OF_SALES_MIN*100:.1f}%-{self.CAPEX_PCT_OF_SALES_MAX*100:.1f}%)",
                            actual_value=capex_pct,
                            expected_value=self.CAPEX_PCT_OF_SALES_MAX
                        ))
            
            # Validate CapEx/D&A ratio (detects acquisitions)
            if da_series is not None and period in da_series.index:
                da = da_series[period]
                if not pd.isna(da) and da > 0:
                    capex_to_da = abs(capex) / da
                    
                    if capex_to_da > MAX_CAPEX_TO_DA_RATIO:
                        issues.append(ValidationIssue(
                            metric="CapEx/D&A",
                            period=period_str,
                            severity=ValidationSeverity.WARNING,
                            message=f"CapEx/D&A ratio is {capex_to_da:.2f}x, exceeds threshold ({MAX_CAPEX_TO_DA_RATIO}x). May include business acquisitions.",
                            actual_value=capex_to_da,
                            expected_value=MAX_CAPEX_TO_DA_RATIO
                        ))
                    elif capex_to_da > self.CAPEX_TO_DA_MAX:
                        issues.append(ValidationIssue(
                            metric="CapEx/D&A",
                            period=period_str,
                            severity=ValidationSeverity.WARNING,
                            message=f"CapEx/D&A ratio is {capex_to_da:.2f}x, unusually high (typical range: {self.CAPEX_TO_DA_MIN:.1f}x-{self.CAPEX_TO_DA_MAX:.1f}x)",
                            actual_value=capex_to_da,
                            expected_value=self.CAPEX_TO_DA_MAX
                        ))
                    elif capex_to_da < self.CAPEX_TO_DA_MIN:
                        issues.append(ValidationIssue(
                            metric="CapEx/D&A",
                            period=period_str,
                            severity=ValidationSeverity.INFO,
                            message=f"CapEx/D&A ratio is {capex_to_da:.2f}x, low (typical range: {self.CAPEX_TO_DA_MIN:.1f}x-{self.CAPEX_TO_DA_MAX:.1f}x)",
                            actual_value=capex_to_da,
                            expected_value=self.CAPEX_TO_DA_MIN
                        ))
            
            # Validate CapEx/PP&E ratio and CapEx vs PP&E change (detects acquisitions)
            if ppe_series is not None:
                # Find closest PP&E period (balance sheet uses INSTANT periods)
                ppe_value = self._get_aligned_value(ppe_series, period, ppe_series.index.tolist())
                if ppe_value is not None and ppe_value > 0:
                    capex_to_ppe = abs(capex) / ppe_value
                    
                    if capex_to_ppe > self.CAPEX_TO_PPE_MAX:
                        issues.append(ValidationIssue(
                            metric="CapEx/PP&E",
                            period=period_str,
                            severity=ValidationSeverity.WARNING,
                            message=f"CapEx/PP&E ratio is {capex_to_ppe*100:.2f}%, unusually high (typical range: {self.CAPEX_TO_PPE_MIN*100:.1f}%-{self.CAPEX_TO_PPE_MAX*100:.1f}%)",
                            actual_value=capex_to_ppe,
                            expected_value=self.CAPEX_TO_PPE_MAX
                        ))
                    
                    # Check CapEx vs PP&E change (detects acquisitions)
                    # Find previous PP&E period to calculate change
                    try:
                        period_dt = pd.to_datetime(period)
                        prev_ppe_value = None
                        for prev_period in sorted(ppe_series.index, reverse=True):
                            try:
                                prev_dt = pd.to_datetime(prev_period)
                                if prev_dt < period_dt:
                                    diff_days = abs((period_dt - prev_dt).days)
                                    if 300 <= diff_days <= 400:  # ~1 year
                                        prev_ppe_value = ppe_series.get(prev_period)
                                        break
                            except (ValueError, TypeError):
                                continue
                        
                        if prev_ppe_value is not None and not pd.isna(prev_ppe_value):
                            ppe_change = abs(float(ppe_value) - float(prev_ppe_value))
                            if ppe_change > 0:
                                capex_to_ppe_change = capex_abs / ppe_change
                                if capex_to_ppe_change > MAX_CAPEX_TO_PPE_CHANGE_RATIO:
                                    issues.append(ValidationIssue(
                                        metric="CapEx vs PP&E Change",
                                        period=period_str,
                                        severity=ValidationSeverity.WARNING,
                                        message=f"CapEx ({capex_abs:.0f}) is {capex_to_ppe_change:.1f}x PP&E change ({ppe_change:.0f}), exceeds threshold ({MAX_CAPEX_TO_PPE_CHANGE_RATIO}x). May include business acquisitions.",
                                        actual_value=capex_to_ppe_change,
                                        expected_value=MAX_CAPEX_TO_PPE_CHANGE_RATIO
                                    ))
                    except (ValueError, TypeError):
                        pass  # Skip PP&E change check if period parsing fails
        
        return issues
    
    def validate_da(
        self,
        da_series: pd.Series,
        revenue_series: Optional[pd.Series] = None,
        ppe_series: Optional[pd.Series] = None,
    ) -> List[ValidationIssue]:
        """
        Validate D&A values for reasonableness.
        
        Includes checks for:
        - D&A % of Revenue (detects unit scale issues if > 15%)
        - D&A % of PP&E (detects unit scale issues if > 30%)
        - Historical consistency (detects jumps > 2x)
        
        Args:
            da_series: Series of D&A values indexed by period
            revenue_series: Optional series of Revenue values for ratio validation
            ppe_series: Optional series of PP&E values for ratio validation
            
        Returns:
            List of validation issues found
        """
        issues = []
        MAX_DA_TO_PPE_RATIO = 0.30  # D&A shouldn't exceed 30% of PP&E
        MAX_DA_TO_REVENUE_RATIO = 0.15  # D&A shouldn't exceed 15% of revenue
        MAX_DA_HISTORICAL_JUMP_RATIO = 2.0  # D&A shouldn't jump more than 2x
        
        for period, da in da_series.items():
            if pd.isna(da) or da == 0:
                continue
            
            period_str = str(period)
            da_abs = abs(da)
            
            # Validate D&A % of PP&E (detects unit scale issues)
            if ppe_series is not None:
                ppe_value = self._get_aligned_value(ppe_series, period, ppe_series.index.tolist())
                if ppe_value is not None and ppe_value > 0:
                    da_to_ppe = da_abs / ppe_value
                    if da_to_ppe > MAX_DA_TO_PPE_RATIO:
                        issues.append(ValidationIssue(
                            metric="D&A % of PP&E",
                            period=period_str,
                            severity=ValidationSeverity.ERROR,
                            message=f"D&A ({da_abs:.0f}) is {da_to_ppe*100:.1f}% of PP&E ({ppe_value:.0f}), exceeds threshold ({MAX_DA_TO_PPE_RATIO*100:.0f}%). May indicate unit scale issue or wrong tag.",
                            actual_value=da_to_ppe,
                            expected_value=MAX_DA_TO_PPE_RATIO
                        ))
            
            # Validate D&A % of Revenue (detects unit scale issues)
            if revenue_series is not None and period in revenue_series.index:
                revenue = revenue_series[period]
                if not pd.isna(revenue) and revenue > 0:
                    da_to_revenue = da_abs / revenue
                    if da_to_revenue > MAX_DA_TO_REVENUE_RATIO:
                        issues.append(ValidationIssue(
                            metric="D&A % of Revenue",
                            period=period_str,
                            severity=ValidationSeverity.WARNING,
                            message=f"D&A ({da_abs:.0f}) is {da_to_revenue*100:.1f}% of Revenue ({revenue:.0f}), exceeds threshold ({MAX_DA_TO_REVENUE_RATIO*100:.0f}%). May indicate unit scale issue.",
                            actual_value=da_to_revenue,
                            expected_value=MAX_DA_TO_REVENUE_RATIO
                        ))
            
            # Validate historical consistency
            try:
                period_dt = pd.to_datetime(period)
                previous_periods = [
                    (k, pd.to_datetime(k))
                    for k in da_series.index
                    if pd.to_datetime(k) < period_dt
                ]
                
                if previous_periods:
                    previous_periods.sort(key=lambda x: x[1], reverse=True)
                    prev_key = previous_periods[0][0]
                    prev_da = da_series[prev_key]
                    if not pd.isna(prev_da) and prev_da != 0:
                        prev_da_abs = abs(prev_da)
                        jump_ratio = da_abs / prev_da_abs if prev_da_abs > 0 else float('inf')
                        if jump_ratio > MAX_DA_HISTORICAL_JUMP_RATIO:
                            issues.append(ValidationIssue(
                                metric="D&A Historical Consistency",
                                period=period_str,
                                severity=ValidationSeverity.WARNING,
                                message=f"D&A jumped {jump_ratio:.1f}x from previous period ({prev_da_abs:.0f} → {da_abs:.0f}). May indicate unit scale issue or wrong tag.",
                                actual_value=jump_ratio,
                                expected_value=MAX_DA_HISTORICAL_JUMP_RATIO
                            ))
            except (ValueError, TypeError):
                pass  # Skip historical check if period parsing fails
        
        return issues
    
    def detect_outliers_statistical(
        self,
        series: pd.Series,
        metric_name: str,
    ) -> List[ValidationIssue]:
        """
        Detect outliers using statistical methods (z-score).
        
        Args:
            series: Series of values to check for outliers
            metric_name: Name of the metric being checked
            
        Returns:
            List of validation issues for detected outliers
        """
        issues = []
        
        # Need at least minimum samples for statistical detection
        valid_values = series.dropna()
        if len(valid_values) < self.STATISTICAL_OUTLIER_MIN_SAMPLES:
            return issues
        
        # Calculate z-scores
        mean = valid_values.mean()
        std = valid_values.std()
        
        if std == 0:
            return issues  # No variation, no outliers
        
        z_scores = (valid_values - mean) / std
        outlier_threshold = self.STATISTICAL_OUTLIER_STD_DEVS
        
        for period, value in valid_values.items():
            z_score = abs(z_scores[period])
            if z_score > outlier_threshold:
                issues.append(ValidationIssue(
                    metric=metric_name,
                    period=str(period),
                    severity=ValidationSeverity.WARNING,
                    message=f"{metric_name} value {value:.2f} is a statistical outlier (z-score: {z_score:.2f}, mean: {mean:.2f}, std: {std:.2f})",
                    actual_value=value,
                    expected_value=mean
                ))
        
        return issues
    
    def detect_outliers_rule_based(
        self,
        capex_series: pd.Series,
        revenue_series: Optional[pd.Series] = None,
        da_series: Optional[pd.Series] = None,
    ) -> List[ValidationIssue]:
        """
        Detect outliers using rule-based methods.
        
        Args:
            capex_series: Series of CapEx values
            revenue_series: Optional series of Revenue values
            da_series: Optional series of D&A values
            
        Returns:
            List of validation issues for detected outliers
        """
        issues = []
        
        for period, capex in capex_series.items():
            if pd.isna(capex) or capex == 0:
                continue
            
            period_str = str(period)
            
            # Rule 1: CapEx > Revenue (should never happen)
            if revenue_series is not None and period in revenue_series.index:
                revenue = revenue_series[period]
                if not pd.isna(revenue) and revenue > 0 and abs(capex) > revenue:
                    issues.append(ValidationIssue(
                        metric="CapEx",
                        period=period_str,
                        severity=ValidationSeverity.ERROR,
                        message=f"CapEx ({abs(capex):.2f}) exceeds Revenue ({revenue:.2f}), this is impossible",
                        actual_value=capex,
                        expected_value=revenue * 0.15,  # Suggest 15% as reasonable
                        suggested_correction=revenue * 0.15
                    ))
            
            # Rule 2: CapEx > 5x D&A (unless company is growing rapidly)
            if da_series is not None and period in da_series.index:
                da = da_series[period]
                if not pd.isna(da) and da > 0:
                    capex_to_da = abs(capex) / da
                    if capex_to_da > 5.0:
                        # Check if this is part of a growth trend
                        # If previous periods also have high CapEx/D&A, might be legitimate growth
                        is_growth_trend = False
                        if len(capex_series) > 1:
                            prev_periods = [p for p in capex_series.index if p < period]
                            if prev_periods:
                                prev_period = max(prev_periods)
                                if prev_period in da_series.index:
                                    prev_da = da_series[prev_period]
                                    prev_capex = capex_series[prev_period]
                                    if not (pd.isna(prev_da) or pd.isna(prev_capex)) and prev_da > 0:
                                        prev_ratio = abs(prev_capex) / prev_da
                                        if prev_ratio > 3.0:  # Previous period also high
                                            is_growth_trend = True
                        
                        if not is_growth_trend:
                            issues.append(ValidationIssue(
                                metric="CapEx/D&A",
                                period=period_str,
                                severity=ValidationSeverity.WARNING,
                                message=f"CapEx/D&A ratio is {capex_to_da:.2f}x, exceeds outlier threshold (5x). May indicate data error.",
                                actual_value=capex_to_da,
                                expected_value=self.CAPEX_TO_DA_MAX,
                                suggested_correction=da * self.CAPEX_TO_DA_MAX
                            ))
            
            # Rule 3: Sign inconsistency check
            # Cash flow statements typically use negative convention for CapEx (cash outflow)
            # Check if this value has inconsistent sign compared to others
            other_values = [v for p, v in capex_series.items() if p != period and not pd.isna(v)]
            if other_values:
                negative_count = sum(1 for v in other_values if v < 0)
                positive_count = sum(1 for v in other_values if v > 0)
                
                # If majority are negative (cash flow convention) but this one is positive
                if negative_count > len(other_values) * 0.5 and capex > 0:
                    issues.append(ValidationIssue(
                        metric="CapEx",
                        period=period_str,
                        severity=ValidationSeverity.WARNING,
                        message=f"CapEx is positive ({capex:.2f}) while other periods are negative (cash flow convention), may indicate sign error",
                        actual_value=capex,
                        suggested_correction=-abs(capex)
                    ))
                # If majority are positive but this one is negative (unusual but possible)
                elif positive_count > len(other_values) * 0.5 and capex < 0:
                    issues.append(ValidationIssue(
                        metric="CapEx",
                        period=period_str,
                        severity=ValidationSeverity.INFO,
                        message=f"CapEx is negative ({capex:.2f}) while other periods are positive, may indicate reporting convention difference",
                        actual_value=capex,
                        suggested_correction=abs(capex)
                    ))
        
        return issues
    
    def validate_balance_sheet(
        self,
        bs_df: pd.DataFrame,
    ) -> List[ValidationIssue]:
        """
        Validate balance sheet for accounting equation and reasonableness.
        
        Args:
            bs_df: Balance sheet DataFrame with metrics as columns, periods as index
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        # Check accounting equation: Assets = Liabilities + Equity
        if "Total Assets" in bs_df.columns:
            if "Total Liabilities" in bs_df.columns and "Stockholders Equity" in bs_df.columns:
                for period in bs_df.index:
                    assets = bs_df.loc[period, "Total Assets"]
                    liabilities = bs_df.loc[period, "Total Liabilities"]
                    equity = bs_df.loc[period, "Stockholders Equity"]
                    
                    if not (pd.isna(assets) or pd.isna(liabilities) or pd.isna(equity)):
                        expected_equity = assets - liabilities
                        diff = abs(equity - expected_equity)
                        diff_pct = diff / abs(assets) if assets != 0 else 0
                        
                        # Allow small rounding differences (0.1%)
                        if diff_pct > 0.001:
                            issues.append(ValidationIssue(
                                metric="Accounting Equation",
                                period=str(period),
                                severity=ValidationSeverity.ERROR,
                                message=f"Assets ({assets:.2f}) ≠ Liabilities ({liabilities:.2f}) + Equity ({equity:.2f}), difference: {diff:.2f} ({diff_pct*100:.3f}%)",
                                actual_value=equity,
                                expected_value=expected_equity
                            ))
        
        # Validate component relationships
        if "Current Assets" in bs_df.columns:
            if "Receivables" in bs_df.columns and "Inventory" in bs_df.columns:
                for period in bs_df.index:
                    current_assets = bs_df.loc[period, "Current Assets"]
                    receivables = bs_df.loc[period, "Receivables"]
                    inventory = bs_df.loc[period, "Inventory"]
                    
                    if not (pd.isna(current_assets) or pd.isna(receivables) or pd.isna(inventory)):
                        components = receivables + inventory
                        if current_assets < components:
                            issues.append(ValidationIssue(
                                metric="Current Assets",
                                period=str(period),
                                severity=ValidationSeverity.WARNING,
                                message=f"Current Assets ({current_assets:.2f}) < Receivables ({receivables:.2f}) + Inventory ({inventory:.2f})",
                                actual_value=current_assets,
                                expected_value=components
                            ))
        
        # Validate PP&E relative to Total Assets
        if "Property, Plant & Equipment" in bs_df.columns and "Total Assets" in bs_df.columns:
            for period in bs_df.index:
                ppe = bs_df.loc[period, "Property, Plant & Equipment"]
                total_assets = bs_df.loc[period, "Total Assets"]
                
                if not (pd.isna(ppe) or pd.isna(total_assets)) and total_assets > 0:
                    ppe_pct = ppe / total_assets
                    # PP&E should typically be 10-60% of total assets (varies by industry)
                    if ppe_pct > 0.80:
                        issues.append(ValidationIssue(
                            metric="PP&E/Total Assets",
                            period=str(period),
                            severity=ValidationSeverity.INFO,
                            message=f"PP&E is {ppe_pct*100:.2f}% of Total Assets, unusually high",
                            actual_value=ppe_pct,
                            expected_value=0.60
                        ))
        
        return issues
    
    def validate_cash_flow(
        self,
        cf_df: pd.DataFrame,
    ) -> List[ValidationIssue]:
        """
        Validate cash flow statement for consistency.
        
        Args:
            cf_df: Cash flow DataFrame with metrics as columns, periods as index
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        # Validate: Operating CF + Investing CF + Financing CF ≈ Net Change in Cash
        required_metrics = ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Net Change in Cash"]
        if all(metric in cf_df.columns for metric in required_metrics):
            for period in cf_df.index:
                operating = cf_df.loc[period, "Operating Cash Flow"]
                investing = cf_df.loc[period, "Investing Cash Flow"]
                financing = cf_df.loc[period, "Financing Cash Flow"]
                net_change = cf_df.loc[period, "Net Change in Cash"]
                
                if not any(pd.isna(val) for val in [operating, investing, financing, net_change]):
                    calculated_change = operating + investing + financing
                    diff = abs(net_change - calculated_change)
                    
                    # Allow small rounding differences (0.1% or $1000)
                    if diff > max(abs(net_change) * 0.001, 1000):
                        issues.append(ValidationIssue(
                            metric="Cash Flow Equation",
                            period=str(period),
                            severity=ValidationSeverity.WARNING,
                            message=f"Net Change in Cash ({net_change:.2f}) ≠ Operating ({operating:.2f}) + Investing ({investing:.2f}) + Financing ({financing:.2f}), difference: {diff:.2f}",
                            actual_value=net_change,
                            expected_value=calculated_change
                        ))
        
        return issues
    
    def validate_cross_statement(
        self,
        is_df: Optional[pd.DataFrame] = None,
        bs_df: Optional[pd.DataFrame] = None,
        cf_df: Optional[pd.DataFrame] = None,
    ) -> List[ValidationIssue]:
        """
        Validate relationships across financial statements.
        
        Args:
            is_df: Optional income statement DataFrame
            bs_df: Optional balance sheet DataFrame
            cf_df: Optional cash flow DataFrame
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        # Validate D&A consistency between income statement and cash flow
        if is_df is not None and cf_df is not None:
            # Note: D&A might be in income statement or cash flow, or both
            # This is a cross-check if both exist
            pass  # Can be implemented if needed
        
        # Validate PP&E changes align with CapEx and D&A
        if bs_df is not None and cf_df is not None:
            if "Property, Plant & Equipment" in bs_df.columns:
                if "CapEx" in cf_df.columns and "Depreciation & Amortization" in cf_df.columns:
                    ppe_series = bs_df["Property, Plant & Equipment"]
                    capex_series = cf_df["CapEx"]
                    da_series = cf_df["Depreciation & Amortization"]
                    
                    # Calculate expected PP&E change: CapEx - D&A
                    for period in capex_series.index:
                        if period in da_series.index:
                            capex = capex_series[period]
                            da = da_series[period]
                            
                            if not (pd.isna(capex) or pd.isna(da)):
                                # Find PP&E values for this period and previous period
                                ppe_end = self._get_aligned_value(ppe_series, period, ppe_series.index.tolist())
                                
                                if ppe_end is not None:
                                    # Find previous period
                                    period_dt = pd.to_datetime(period)
                                    ppe_start = None
                                    for ppe_period in sorted(ppe_series.index, reverse=True):
                                        try:
                                            ppe_period_dt = pd.to_datetime(ppe_period)
                                            if ppe_period_dt < period_dt:
                                                ppe_start = ppe_series[ppe_period]
                                                break
                                        except (ValueError, TypeError):
                                            continue
                                    
                                    if ppe_start is not None:
                                        actual_change = ppe_end - ppe_start
                                        expected_change = abs(capex) - da  # CapEx increases PP&E, D&A decreases it
                                        diff = abs(actual_change - expected_change)
                                        
                                        # Allow reasonable differences (disposals, impairments, etc.)
                                        if diff > max(abs(expected_change) * 0.10, abs(ppe_end) * 0.05):
                                            issues.append(ValidationIssue(
                                                metric="PP&E Change",
                                                period=str(period),
                                                severity=ValidationSeverity.INFO,
                                                message=f"PP&E change ({actual_change:.2f}) doesn't align with CapEx ({capex:.2f}) - D&A ({da:.2f}) = {expected_change:.2f}, difference: {diff:.2f}",
                                                actual_value=actual_change,
                                                expected_value=expected_change
                                            ))
        
        return issues
    
    @staticmethod
    def _get_aligned_value(
        series: pd.Series,
        target_date: str,
        available_periods: list
    ) -> Optional[float]:
        """
        Get value from series aligned to target date period.
        
        Handles period alignment between balance sheet (instant periods) and 
        income statement/cash flow (period-end dates).
        
        Args:
            series: Series with values indexed by period
            target_date: Target date string to align to
            available_periods: List of available periods in the series
            
        Returns:
            Aligned value or None if not found
        """
        if series.empty or not available_periods:
            return None
        
        # Try exact match first
        if target_date in available_periods:
            value = series.get(target_date)
            if pd.isna(value):
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        
        # Try to parse target_date and find closest period
        try:
            target_dt = pd.to_datetime(target_date)
            
            # Find closest period by comparing dates
            closest_period = None
            min_diff = None
            
            for period in available_periods:
                try:
                    period_dt = pd.to_datetime(period)
                    diff = abs((target_dt - period_dt).days)
                    
                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                        closest_period = period
                except (ValueError, TypeError):
                    continue
            
            if closest_period is not None and min_diff is not None and min_diff <= 365:
                # Only use if within 1 year
                value = series.get(closest_period)
                if pd.isna(value):
                    return None
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None
        except (ValueError, TypeError):
            pass
        
        return None
    
    def validate_all(
        self,
        is_df: Optional[pd.DataFrame] = None,
        bs_df: Optional[pd.DataFrame] = None,
        cf_df: Optional[pd.DataFrame] = None,
    ) -> List[ValidationIssue]:
        """
        Run all validation checks.
        
        Args:
            is_df: Optional income statement DataFrame
            bs_df: Optional balance sheet DataFrame
            cf_df: Optional cash flow DataFrame
            
        Returns:
            List of all validation issues found
        """
        all_issues = []
        
        # Validate CapEx if cash flow available
        if cf_df is not None and "CapEx" in cf_df.columns:
            capex_series = cf_df["CapEx"]
            revenue_series = is_df["Revenue"] if is_df is not None and "Revenue" in is_df.columns else None
            da_series = cf_df["Depreciation & Amortization"] if "Depreciation & Amortization" in cf_df.columns else None
            ppe_series = bs_df["Property, Plant & Equipment"] if bs_df is not None and "Property, Plant & Equipment" in bs_df.columns else None
            
            all_issues.extend(self.validate_capex(capex_series, revenue_series, da_series, ppe_series))
            
            # Detect outliers
            all_issues.extend(self.detect_outliers_statistical(capex_series, "CapEx"))
            all_issues.extend(self.detect_outliers_rule_based(capex_series, revenue_series, da_series))
        
        # Validate balance sheet
        if bs_df is not None:
            all_issues.extend(self.validate_balance_sheet(bs_df))
        
        # Validate cash flow
        if cf_df is not None:
            all_issues.extend(self.validate_cash_flow(cf_df))
        
        # Cross-statement validation
        all_issues.extend(self.validate_cross_statement(is_df, bs_df, cf_df))
        
        return all_issues
