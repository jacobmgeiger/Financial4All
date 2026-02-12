# financial4all/analysis/profitability_analyzer.py
"""
Profitability ratios and expense/revenue breakdowns from income (and optional BS/CF) DataFrames.

ProfitabilityAnalyzer.calculate_ratios() produces a transposed DataFrame (Metric + date
columns) with Y/Y revenue growth, expenses as % of revenue, operating margin, tax rate,
CapEx % of sales, and related metrics. Input format matches IncomeStatement.to_dataframe().
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union


class ProfitabilityAnalyzer:
    """
    Analyzer for calculating profitability ratios from income statements.

    Calculates year-over-year revenue growth, expenses as percentage of revenue,
    operating margin, and tax rate.
    """

    @staticmethod
    def calculate_ratios(
        is_df: pd.DataFrame,
        bs_df: Optional[pd.DataFrame] = None,
        cf_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate profitability ratios from an income statement DataFrame.

        Args:
            is_df: Income statement DataFrame with periods as index and metrics as columns.
                  This is the format returned by IncomeStatement.to_dataframe().
            bs_df: Optional balance sheet DataFrame with periods as index and metrics as columns.
            cf_df: Optional cash flow DataFrame with periods as index and metrics as columns.

        Returns:
            DataFrame in transposed format (Metric column + date columns) ready for display.
            Values are returned as decimals (0.50 for 50%).
        """
        if is_df.empty:
            return pd.DataFrame()

        # Get date columns (periods) from index
        date_columns = is_df.index.tolist()

        # Initialize ordered list to preserve metric order
        # Format: [(metric_name, {date: value, ...}), ...]
        ordered_metrics = []

        # Store percentage values separately for Y/Y change calculation
        pct_values = {}

        # 1. Y/Y Revenue Growth (percentage change for initial display)
        revenue_yoy_data = None
        revenue_growth_rates = None
        if "Revenue" in is_df.columns:
            revenue_growth = ProfitabilityAnalyzer._calculate_yoy_growth(
                is_df["Revenue"], date_columns
            )
            ordered_metrics.append(("Revenue Y/Y % Change", revenue_growth))
            # Store growth rates for Trends section calculation
            revenue_growth_rates = revenue_growth

        # Add Y/Y % change for Outstanding Shares Basic and Diluted (alongside Revenue)
        shares_growth_rates = {}  # Store growth rates for Trends section calculation
        shares_metrics = ["Outstanding Shares Basic", "Outstanding Shares Diluted"]
        for shares_metric in shares_metrics:
            if shares_metric in is_df.columns:
                shares_growth = ProfitabilityAnalyzer._calculate_yoy_growth(
                    is_df[shares_metric], date_columns
                )
                ordered_metrics.append((f"{shares_metric} Y/Y % Change", shares_growth))
                # Store growth rates for Trends section calculation
                shares_growth_rates[shares_metric] = shares_growth

        # 2. Expenses as % of Revenue section
        # Add section header row (empty values)
        ordered_metrics.append(
            ("Expenses as % of Revenue", {col: np.nan for col in date_columns})
        )

        # Calculate each expense as % of Revenue
        expense_metrics = [
            ("Gross Margin", "Gross Profit"),
            ("Research and development", "R&D Expenses"),
            ("Sales, general and administrative", "SG&A Expenses"),
            ("Restructuring and other charges", "Restructuring and other charges"),
            ("Acquisition termination cost", "Acquisition termination cost"),
            ("Interest income", "Interest Income"),
            ("Interest expense", "Interest Expense"),
            ("Other, net", "Other, net"),
        ]

        for display_name, metric_name in expense_metrics:
            if metric_name in is_df.columns and "Revenue" in is_df.columns:
                expense_pct = ProfitabilityAnalyzer._calculate_percentage_of_revenue(
                    is_df[metric_name], is_df["Revenue"], date_columns
                )
                ordered_metrics.append((display_name, expense_pct))
                # Store for Y/Y change calculation
                pct_values[display_name] = expense_pct

        # 3. Operating Margin = Operating Income / Revenue
        if "Operating Income" in is_df.columns and "Revenue" in is_df.columns:
            operating_margin = ProfitabilityAnalyzer._calculate_percentage_of_revenue(
                is_df["Operating Income"], is_df["Revenue"], date_columns
            )
            ordered_metrics.append(("Operating Margin", operating_margin))
            # Store for Y/Y change calculation
            pct_values["Operating Margin"] = operating_margin

        # 4. Tax Rate = Income Tax Expense (benefit) / Profit Before Taxes
        # Note: "Taxes" is the standard metric name, "Income Before Taxes" is the standard name
        if "Taxes" in is_df.columns and "Income Before Taxes" in is_df.columns:
            tax_rate = ProfitabilityAnalyzer._calculate_percentage_of_revenue(
                is_df["Taxes"], is_df["Income Before Taxes"], date_columns
            )
            ordered_metrics.append(("Tax rate", tax_rate))
            # Store for Y/Y change calculation
            pct_values["Tax rate"] = tax_rate

        # 5. Year-over-Year % Change (Trends) section
        # Add section header row (empty values)
        ordered_metrics.append(
            ("**%Change y/y Change (Trends)**", {col: np.nan for col in date_columns})
        )

        # Revenue Y/Y change: difference between this period's growth rate and last period's growth rate
        if revenue_growth_rates is not None:
            # revenue_growth_rates is a dictionary mapping date columns to growth rates (as decimals)
            # Calculate the difference between consecutive growth rates
            revenue_trend_diff = {}
            for i, date_col in enumerate(date_columns):
                current_growth = revenue_growth_rates.get(date_col)
                # Check if there's a next (older) period to compare to
                if i + 1 < len(date_columns):
                    next_date_col = date_columns[i + 1]
                    next_growth = revenue_growth_rates.get(next_date_col)

                    if (
                        current_growth is not None
                        and next_growth is not None
                        and not pd.isna(current_growth)
                        and not pd.isna(next_growth)
                    ):
                        try:
                            # Difference = current growth rate - previous growth rate
                            # Both are already decimals (e.g., 0.10 for 10%)
                            diff = float(current_growth) - float(next_growth)
                            revenue_trend_diff[date_col] = diff
                        except (ValueError, TypeError):
                            revenue_trend_diff[date_col] = np.nan
                    else:
                        revenue_trend_diff[date_col] = np.nan
                else:
                    # No older period to compare to (oldest period)
                    revenue_trend_diff[date_col] = np.nan

            ordered_metrics.append(("Revenue Y/Y % Change", revenue_trend_diff))

        # Add "Change of Y/Y % Change" for Outstanding Shares Basic and Diluted
        # This calculates the difference between consecutive Y/Y % change values
        for shares_metric, shares_growth_rates_dict in shares_growth_rates.items():
            # Calculate the difference between consecutive growth rates (trend difference)
            shares_trend_diff = {}
            for i, date_col in enumerate(date_columns):
                current_growth = shares_growth_rates_dict.get(date_col)
                # Check if there's a next (older) period to compare to
                if i + 1 < len(date_columns):
                    next_date_col = date_columns[i + 1]
                    next_growth = shares_growth_rates_dict.get(next_date_col)

                    if (
                        current_growth is not None
                        and next_growth is not None
                        and not pd.isna(current_growth)
                        and not pd.isna(next_growth)
                    ):
                        try:
                            # Difference = current growth rate - previous growth rate
                            # Both are already decimals (e.g., 0.10 for 10%)
                            diff = float(current_growth) - float(next_growth)
                            shares_trend_diff[date_col] = diff
                        except (ValueError, TypeError):
                            shares_trend_diff[date_col] = np.nan
                    else:
                        shares_trend_diff[date_col] = np.nan
                else:
                    # No older period to compare to (oldest period)
                    shares_trend_diff[date_col] = np.nan

            ordered_metrics.append(
                (f"{shares_metric} Change of Y/Y % Change", shares_trend_diff)
            )

        # Calculate Y/Y change for percentage metrics (absolute difference: this year's % - last year's %)
        # These are calculated from the percentage values stored above
        for display_name, pct_data in pct_values.items():
            pct_series = pd.Series(pct_data, index=date_columns)
            yoy_change = ProfitabilityAnalyzer._calculate_yoy_absolute_difference(
                pct_series, date_columns
            )
            # Add Y/Y change values - these appear after the Y/Y header
            ordered_metrics.append((display_name, yoy_change))

        # 6. Capital & Working Capital section (after Trends section)
        # Extract metrics from Cash Flow and Balance Sheet if available
        if cf_df is not None and not cf_df.empty:
            # Align periods between cf_df and is_df
            cf_periods = cf_df.index.tolist()

            # Depreciation & Amortization
            if "Depreciation & Amortization" in cf_df.columns:
                da_data = {}
                for date_col in date_columns:
                    # Find closest matching period in cf_df
                    cf_value = ProfitabilityAnalyzer._get_aligned_value(
                        cf_df["Depreciation & Amortization"], date_col, cf_periods
                    )
                    da_data[date_col] = cf_value
                ordered_metrics.append(("Depreciation & Amortization", da_data))

                # D&A % of Sales
                if "Revenue" in is_df.columns:
                    da_pct = ProfitabilityAnalyzer._calculate_percentage_from_dict(
                        da_data, is_df["Revenue"], date_columns
                    )
                    ordered_metrics.append(("D&A % of Sales", da_pct))
                    pct_values["D&A % of Sales"] = da_pct

            # CapEx
            if "CapEx" in cf_df.columns:
                capex_data = {}
                for date_col in date_columns:
                    # Find closest matching period in cf_df
                    cf_value = ProfitabilityAnalyzer._get_aligned_value(
                        cf_df["CapEx"], date_col, cf_periods
                    )
                    capex_data[date_col] = cf_value
                ordered_metrics.append(("CapEx", capex_data))

                # CapEx % of Sales
                if "Revenue" in is_df.columns:
                    capex_pct = ProfitabilityAnalyzer._calculate_percentage_from_dict(
                        capex_data, is_df["Revenue"], date_columns
                    )
                    ordered_metrics.append(("CapEx % of Sales", capex_pct))
                    pct_values["CapEx % of Sales"] = capex_pct

        if bs_df is not None and not bs_df.empty:
            # Align periods between bs_df and is_df
            bs_periods = bs_df.index.tolist()

            # Receivables
            receivables_data = {}
            if "Receivables" in bs_df.columns:
                for date_col in date_columns:
                    bs_value = ProfitabilityAnalyzer._get_aligned_value(
                        bs_df["Receivables"], date_col, bs_periods
                    )
                    receivables_data[date_col] = bs_value
                ordered_metrics.append(("Receivables", receivables_data))

                # Receivables % of Sales
                if "Revenue" in is_df.columns:
                    receivables_pct = (
                        ProfitabilityAnalyzer._calculate_percentage_from_dict(
                            receivables_data, is_df["Revenue"], date_columns
                        )
                    )
                    ordered_metrics.append(("Receivables % of Sales", receivables_pct))
                    pct_values["Receivables % of Sales"] = receivables_pct

            # Inventory
            inventory_data = {}
            if "Inventory" in bs_df.columns:
                for date_col in date_columns:
                    bs_value = ProfitabilityAnalyzer._get_aligned_value(
                        bs_df["Inventory"], date_col, bs_periods
                    )
                    inventory_data[date_col] = bs_value
                ordered_metrics.append(("Inventory", inventory_data))

                # Inventory % of Sales
                if "Revenue" in is_df.columns:
                    inventory_pct = (
                        ProfitabilityAnalyzer._calculate_percentage_from_dict(
                            inventory_data, is_df["Revenue"], date_columns
                        )
                    )
                    ordered_metrics.append(("Inventory % of Sales", inventory_pct))
                    pct_values["Inventory % of Sales"] = inventory_pct

            # Payables
            payables_data = {}
            if "Payables" in bs_df.columns:
                for date_col in date_columns:
                    bs_value = ProfitabilityAnalyzer._get_aligned_value(
                        bs_df["Payables"], date_col, bs_periods
                    )
                    payables_data[date_col] = bs_value
                ordered_metrics.append(("Payables", payables_data))

                # Payables % of Sales
                if "Revenue" in is_df.columns:
                    payables_pct = (
                        ProfitabilityAnalyzer._calculate_percentage_from_dict(
                            payables_data, is_df["Revenue"], date_columns
                        )
                    )
                    ordered_metrics.append(("Payables % of Sales", payables_pct))
                    pct_values["Payables % of Sales"] = payables_pct

            # Change in WC = (Receivables + Inventory - Payables) current - (Receivables + Inventory - Payables) previous
            if receivables_data and inventory_data and payables_data:
                wc_change_data = {}
                for i, date_col in enumerate(date_columns):
                    # Calculate current period WC
                    rec_current = receivables_data.get(date_col)
                    inv_current = inventory_data.get(date_col)
                    pay_current = payables_data.get(date_col)

                    rec_float = ProfitabilityAnalyzer._parse_numeric_value(rec_current)
                    inv_float = ProfitabilityAnalyzer._parse_numeric_value(inv_current)
                    pay_float = ProfitabilityAnalyzer._parse_numeric_value(pay_current)

                    if (
                        rec_float is not None
                        and inv_float is not None
                        and pay_float is not None
                    ):
                        current_wc = rec_float + inv_float - pay_float

                        # Calculate previous period WC if available
                        if i + 1 < len(date_columns):
                            prev_date_col = date_columns[i + 1]
                            rec_prev = receivables_data.get(prev_date_col)
                            inv_prev = inventory_data.get(prev_date_col)
                            pay_prev = payables_data.get(prev_date_col)

                            rec_prev_float = ProfitabilityAnalyzer._parse_numeric_value(
                                rec_prev
                            )
                            inv_prev_float = ProfitabilityAnalyzer._parse_numeric_value(
                                inv_prev
                            )
                            pay_prev_float = ProfitabilityAnalyzer._parse_numeric_value(
                                pay_prev
                            )

                            if (
                                rec_prev_float is not None
                                and inv_prev_float is not None
                                and pay_prev_float is not None
                            ):
                                prev_wc = (
                                    rec_prev_float + inv_prev_float - pay_prev_float
                                )
                                wc_change_data[date_col] = current_wc - prev_wc
                            else:
                                wc_change_data[date_col] = np.nan
                        else:
                            wc_change_data[date_col] = np.nan
                    else:
                        wc_change_data[date_col] = np.nan

                ordered_metrics.append(("Change in WC", wc_change_data))

        # Convert to DataFrame directly from ordered list to preserve order and handle duplicates
        if not ordered_metrics:
            return pd.DataFrame()

        # Build DataFrame rows directly
        rows = []
        for metric_name, values_dict in ordered_metrics:
            row = {"Metric": metric_name}
            # Add values for each date column
            for date_col in date_columns:
                row[date_col] = values_dict.get(date_col, np.nan)
            rows.append(row)

        # Create DataFrame from rows
        result_df = pd.DataFrame(rows)

        return result_df

    @staticmethod
    def _parse_numeric_value(value: Union[str, float, int, None]) -> Optional[float]:
        """
        Parse a numeric value, handling parentheses notation for negatives.

        Args:
            value: Value that may be a number, string with parentheses, or None

        Returns:
            Float value (negative if parentheses notation was used), or None if invalid
        """
        if value is None or pd.isna(value):
            return None

        # If already a numeric type, return as float
        if isinstance(value, (int, float)):
            return float(value)

        # If string, check for parentheses notation
        if isinstance(value, str):
            value = value.strip()
            # Check if value is wrapped in parentheses (accounting notation for negative)
            if value.startswith("(") and value.endswith(")"):
                # Remove parentheses and parse as negative
                inner_value = value[1:-1].strip()
                try:
                    return -float(inner_value.replace(",", ""))
                except (ValueError, TypeError):
                    return None
            else:
                # Regular parsing
                try:
                    return float(value.replace(",", ""))
                except (ValueError, TypeError):
                    return None

        # Try direct conversion
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _calculate_yoy_growth(
        series: pd.Series, date_columns: list
    ) -> Dict[str, float]:
        """
        Calculate year-over-year growth for a metric.

        With periods ordered newest-first, compares each period to the next (older) period.
        For example, with [2024, 2023, 2022]:
        - 2024 growth compares 2024 to 2023
        - 2023 growth compares 2023 to 2022
        - 2022 growth is NaN (no older period)

        Args:
            series: Series with values indexed by period
            date_columns: List of date column names (periods), ordered newest-first

        Returns:
            Dictionary mapping date columns to growth rates (as decimals)
        """
        growth_data = {}

        # Iterate through periods (newest to oldest)
        # Compare each period to the next period (which is older)
        for i, date_col in enumerate(date_columns):
            current_value = series.get(date_col)
            current_float = ProfitabilityAnalyzer._parse_numeric_value(current_value)

            # Check if there's a next (older) period to compare to
            if i + 1 < len(date_columns):
                next_date_col = date_columns[i + 1]
                next_value = series.get(next_date_col)
                next_float = ProfitabilityAnalyzer._parse_numeric_value(next_value)

                if current_float is not None and next_float is not None:
                    try:
                        if next_float != 0:
                            # Growth = (current - previous) / previous
                            # where "previous" is the next (older) period
                            growth = (current_float - next_float) / next_float
                            growth_data[date_col] = growth
                        else:
                            growth_data[date_col] = np.nan
                    except (ValueError, TypeError, ZeroDivisionError):
                        growth_data[date_col] = np.nan
                else:
                    growth_data[date_col] = np.nan
            else:
                # No older period to compare to (oldest period)
                growth_data[date_col] = np.nan

        return growth_data

    @staticmethod
    def _calculate_yoy_absolute_difference(
        series: pd.Series, date_columns: list
    ) -> Dict[str, float]:
        """
        Calculate year-over-year absolute difference for percentage metrics.

        With periods ordered newest-first, calculates: current % - previous %
        For example, with [2024, 2023, 2022]:
        - 2024 difference = 2024 % - 2023 %
        - 2023 difference = 2023 % - 2022 %
        - 2022 difference is NaN (no older period)

        Args:
            series: Series with percentage values indexed by period
            date_columns: List of date column names (periods), ordered newest-first

        Returns:
            Dictionary mapping date columns to absolute differences (as decimals, e.g., 0.05 for 5%)
        """
        diff_data = {}

        # Iterate through periods (newest to oldest)
        # Compare each period to the next period (which is older)
        for i, date_col in enumerate(date_columns):
            current_value = series.get(date_col)
            current_float = ProfitabilityAnalyzer._parse_numeric_value(current_value)

            # Check if there's a next (older) period to compare to
            if i + 1 < len(date_columns):
                next_date_col = date_columns[i + 1]
                next_value = series.get(next_date_col)
                next_float = ProfitabilityAnalyzer._parse_numeric_value(next_value)

                if current_float is not None and next_float is not None:
                    try:
                        # Absolute difference = current % - previous %
                        diff = current_float - next_float
                        diff_data[date_col] = diff
                    except (ValueError, TypeError):
                        diff_data[date_col] = np.nan
                else:
                    diff_data[date_col] = np.nan
            else:
                # No older period to compare to (oldest period)
                diff_data[date_col] = np.nan

        return diff_data

    @staticmethod
    def _calculate_percentage_of_revenue(
        numerator_series: pd.Series, denominator_series: pd.Series, date_columns: list
    ) -> Dict[str, float]:
        """
        Calculate numerator as percentage of denominator for each period.

        Args:
            numerator_series: Series with numerator values indexed by period
            denominator_series: Series with denominator values indexed by period
            date_columns: List of date column names (periods)

        Returns:
            Dictionary mapping date columns to percentages (as decimals, e.g., 0.50 for 50%)
            Negative values are preserved as negative (e.g., -0.50 for -50%)
        """
        pct_data = {}

        for date_col in date_columns:
            numerator_value = numerator_series.get(date_col)
            denominator_value = denominator_series.get(date_col)

            # Parse values, handling parentheses notation for negatives
            num_float = ProfitabilityAnalyzer._parse_numeric_value(numerator_value)
            denom_float = ProfitabilityAnalyzer._parse_numeric_value(denominator_value)

            if num_float is not None and denom_float is not None:
                try:
                    if denom_float != 0:
                        pct = num_float / denom_float
                        pct_data[date_col] = pct
                    else:
                        pct_data[date_col] = np.nan
                except (ValueError, TypeError, ZeroDivisionError):
                    pct_data[date_col] = np.nan
            else:
                pct_data[date_col] = np.nan

        return pct_data

    @staticmethod
    def _normalize_period_key(period_value) -> str:
        """
        Normalize period to YYYY-MM-DD string for consistent alignment.

        Args:
            period_value: Period (date, datetime, or str)

        Returns:
            String in YYYY-MM-DD format
        """
        try:
            if hasattr(period_value, "strftime"):
                return period_value.strftime("%Y-%m-%d")
            return pd.to_datetime(period_value).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return str(period_value)

    @staticmethod
    def _get_aligned_value(
        series: pd.Series, target_date: str, available_periods: list
    ) -> Optional[float]:
        """
        Get value from series aligned to target date period.

        Handles period alignment between balance sheet (instant periods) and
        income statement (period-end dates). Finds the closest matching period.
        Normalizes target_date and available_periods to YYYY-MM-DD to avoid
        type/format mismatches.

        Args:
            series: Series with values indexed by period
            target_date: Target date string to align to
            available_periods: List of available periods in the series

        Returns:
            Aligned value or None if not found
        """
        if series.empty or not available_periods:
            return None

        target_norm = ProfitabilityAnalyzer._normalize_period_key(target_date)
        periods_norm = [
            ProfitabilityAnalyzer._normalize_period_key(p) for p in available_periods
        ]
        # Build period -> original key for series lookup (series index may be date or str)
        period_to_original = dict(zip(periods_norm, available_periods))

        # Try exact match first
        if target_norm in period_to_original:
            original_key = period_to_original[target_norm]
            value = series.get(original_key)
            return ProfitabilityAnalyzer._parse_numeric_value(value)

        # Try to parse target_date and find closest period
        try:
            target_dt = pd.to_datetime(target_norm)

            closest_period_original = None
            min_diff = None

            for i, period_norm in enumerate(periods_norm):
                try:
                    period_dt = pd.to_datetime(period_norm)
                    diff = abs((target_dt - period_dt).days)

                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                        closest_period_original = available_periods[i]
                except (ValueError, TypeError):
                    continue

            if (
                closest_period_original is not None
                and min_diff is not None
                and min_diff <= 365
            ):  # Within a year
                value = series.get(closest_period_original)
                return ProfitabilityAnalyzer._parse_numeric_value(value)
        except (ValueError, TypeError):
            pass

        return None

    @staticmethod
    def _calculate_percentage_from_dict(
        numerator_dict: Dict[str, Optional[float]],
        denominator_series: pd.Series,
        date_columns: list,
    ) -> Dict[str, float]:
        """
        Calculate percentage from dictionary of numerator values and denominator series.

        Args:
            numerator_dict: Dictionary mapping date columns to numerator values
            denominator_series: Series with denominator values indexed by period
            date_columns: List of date column names (periods)

        Returns:
            Dictionary mapping date columns to percentages (as decimals)
        """
        pct_data = {}

        for date_col in date_columns:
            numerator_value = numerator_dict.get(date_col)
            denominator_value = denominator_series.get(date_col)

            num_float = (
                numerator_value
                if isinstance(numerator_value, (int, float))
                else ProfitabilityAnalyzer._parse_numeric_value(numerator_value)
            )
            denom_float = ProfitabilityAnalyzer._parse_numeric_value(denominator_value)

            if num_float is not None and denom_float is not None:
                try:
                    if denom_float != 0:
                        pct = num_float / denom_float
                        pct_data[date_col] = pct
                    else:
                        pct_data[date_col] = np.nan
                except (ValueError, TypeError, ZeroDivisionError):
                    pct_data[date_col] = np.nan
            else:
                pct_data[date_col] = np.nan

        return pct_data
