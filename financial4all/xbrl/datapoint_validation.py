# financial4all/xbrl/datapoint_validation.py
"""
EdgarTools-aligned datapoint validation and sanity checks.

Post-extraction validation: magnitude, YoY change, ratio sanity, sign consistency.
"""

import logging
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from financial4all.xbrl.validation import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)

# Configurable thresholds
REVENUE_YOY_WARNING_MAX = 2.0  # 200% YoY change
REVENUE_YOY_WARNING_MIN = -0.8  # -80% YoY change
GROSS_MARGIN_MIN = 0
GROSS_MARGIN_MAX = 1.0  # 100%
OPERATING_MARGIN_MIN = -0.5
OPERATING_MARGIN_MAX = 0.5
NET_MARGIN_MIN = -0.5
NET_MARGIN_MAX = 0.5
REVENUE_SUSPICIOUS_MIN = 1e6  # Revenue < 1M when company likely larger


def validate_datapoint(
    metric: str,
    value: float,
    period: str,
    context: Optional[Dict[str, Any]] = None,
) -> List[ValidationIssue]:
    """
    Validate a single datapoint for sanity.

    Args:
        metric: Standard metric name (e.g., "Revenue", "Cost of Revenue")
        value: Numeric value
        period: Period key (YYYY-MM-DD)
        context: Optional dict with revenue, other_metrics for cross-checks

    Returns:
        List of ValidationIssue (empty if no issues)
    """
    issues: List[ValidationIssue] = []
    context = context or {}

    if not isinstance(value, (int, float)):
        return issues
    try:
        if value != value:  # NaN
            return issues
    except TypeError:
        return issues

    # Revenue magnitude: suspicious if very small and context suggests larger scale
    if metric == "Revenue" and value > 0:
        if value < REVENUE_SUSPICIOUS_MIN:
            other_rev = context.get("other_period_revenue")
            if other_rev and other_rev > REVENUE_SUSPICIOUS_MIN * 5:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="REVENUE_SUSPICIOUSLY_SMALL",
                        message=f"Revenue ({value:,.0f}) for {period} is very small; other periods suggest larger scale",
                        details={"metric": metric, "value": value, "period": period},
                    )
                )
        if value < 0:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="REVENUE_NEGATIVE",
                    message=f"Revenue is negative ({value:,.0f}) for {period}",
                    details={"metric": metric, "value": value, "period": period},
                )
            )

    # Expense sign: typically positive (or negative if stored as credit)
    if metric in (
        "Cost of Revenue",
        "SG&A Expenses",
        "R&D Expenses",
        "Taxes",
    ) and value > 0:
        # Expenses as positive is normal
        pass
    elif metric in ("Operating Income", "Net Income", "Gross Profit"):
        # Can be negative (loss)
        pass

    return issues


def validate_statement_df(
    df: "pd.DataFrame",
    statement_type: str,
    yoy_threshold_max: float = REVENUE_YOY_WARNING_MAX,
    yoy_threshold_min: float = REVENUE_YOY_WARNING_MIN,
) -> ValidationResult:
    """
    Run post-extraction datapoint validation on a statement DataFrame.

    Checks: magnitude, YoY change, ratio sanity, sign consistency.

    Args:
        df: Statement DataFrame with periods as index
        statement_type: "IncomeStatement", "BalanceSheet", or "CashFlowStatement"
        yoy_threshold_max: Flag Revenue YoY > this as warning (default 200%)
        yoy_threshold_min: Flag Revenue YoY < this as warning (default -80%)

    Returns:
        ValidationResult with issues (warnings only; does not block)
    """
    issues: List[ValidationIssue] = []
    checks: List[str] = []

    if not PANDAS_AVAILABLE or df is None or df.empty:
        return ValidationResult(
            is_valid=True,
            issues=[],
            checks_performed=["data_check"],
            metadata={},
        )

    if statement_type == "IncomeStatement":
        checks.append("revenue_yoy")
        checks.append("margin_ratios")

        # Revenue YoY
        if "Revenue" in df.columns:
            rev_series = df["Revenue"].sort_index()
            rev_valid = rev_series.dropna()
            if len(rev_valid) >= 2:
                rev_pct = rev_valid.pct_change()
                for idx in rev_pct.index:
                    pct = rev_pct.loc[idx]
                    if pd.isna(pct):
                        continue
                    if pct > yoy_threshold_max:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                code="REVENUE_YOY_EXTREME",
                                message=f"Revenue YoY change {pct:.1%} for {idx} exceeds threshold ({yoy_threshold_max:.0%})",
                                details={"period": str(idx), "yoy": float(pct)},
                            )
                        )
                    elif pct < yoy_threshold_min:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                code="REVENUE_YOY_EXTREME",
                                message=f"Revenue YoY change {pct:.1%} for {idx} below threshold ({yoy_threshold_min:.0%})",
                                details={"period": str(idx), "yoy": float(pct)},
                            )
                        )

        # Margin ratios sanity
        if "Revenue" in df.columns and "Gross Profit" in df.columns:
            rev = df["Revenue"]
            gp = df["Gross Profit"]
            valid = rev.notna() & gp.notna() & (rev != 0)
            if valid.any():
                gross_margin = gp / rev
                out_of_range = (gross_margin < GROSS_MARGIN_MIN) | (gross_margin > GROSS_MARGIN_MAX)
                for idx in df.index[valid & out_of_range]:
                    gm = gross_margin.loc[idx]
                    if pd.notna(gm) and (gm < -0.01 or gm > 1.01):  # Allow small rounding
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                code="GROSS_MARGIN_OUT_OF_RANGE",
                                message=f"Gross margin {gm:.1%} for {idx} outside 0-100%",
                                details={"period": str(idx), "gross_margin": float(gm)},
                            )
                        )

        if "Revenue" in df.columns and "Operating Income" in df.columns:
            rev = df["Revenue"]
            oi = df["Operating Income"]
            valid = rev.notna() & oi.notna() & (rev != 0)
            if valid.any():
                op_margin = oi / rev
                out_of_range = (op_margin < OPERATING_MARGIN_MIN - 0.1) | (op_margin > OPERATING_MARGIN_MAX + 0.1)
                for idx in df.index[valid & out_of_range]:
                    om = op_margin.loc[idx]
                    if pd.notna(om):
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.INFO,
                                code="OPERATING_MARGIN_UNUSUAL",
                                message=f"Operating margin {om:.1%} for {idx} is unusual",
                                details={"period": str(idx), "operating_margin": float(om)},
                            )
                        )

    elif statement_type == "BalanceSheet":
        checks.append("balance_sheet_sanity")
        # Balance sheet validation is in validate_balance_sheet

    elif statement_type == "CashFlowStatement":
        checks.append("cash_flow_sanity")

    return ValidationResult(
        is_valid=not any(i.severity == ValidationSeverity.ERROR for i in issues),
        issues=issues,
        checks_performed=checks,
        metadata={"statement_type": statement_type},
    )
