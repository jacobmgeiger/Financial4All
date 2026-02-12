# financial4all/analysis/trend_analyzer.py
"""
Trend analysis on period-indexed financial DataFrames.

TrendAnalyzer takes a DataFrame (e.g. from IncomeStatement.to_dataframe()) and
computes year-over-year growth, CAGR, and similar trend metrics. Used by
reporting and dashboards for growth and comparison views.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime

from financial4all.core import log


class TrendAnalyzer:
    """
    Analyzes trends in financial data.
    
    This class calculates growth rates, identifies trends, and performs
    comparative analysis.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize trend analyzer with a DataFrame.
        
        Args:
            df: DataFrame with financial metrics (index should be dates/periods)
        """
        self.df = df.sort_index()
    
    def calculate_yoy_growth(self, metric: str) -> pd.Series:
        """
        Calculate year-over-year growth rate for a metric.
        
        Args:
            metric: Column name to calculate growth for
            
        Returns:
            Series with YoY growth percentages
        """
        if metric not in self.df.columns:
            return pd.Series(dtype=float)
        
        values = self.df[metric]
        yoy_growth = values.pct_change(periods=1) * 100
        
        return yoy_growth.replace([np.inf, -np.inf], np.nan)
    
    def calculate_cagr(self, metric: str, periods: Optional[int] = None) -> float:
        """
        Calculate Compound Annual Growth Rate (CAGR).
        
        Args:
            metric: Column name to calculate CAGR for
            periods: Number of periods (default: all available)
            
        Returns:
            CAGR as percentage, or NaN if insufficient data
        """
        if metric not in self.df.columns:
            return np.nan
        
        values = self.df[metric].dropna()
        
        if len(values) < 2:
            return np.nan
        
        if periods is None:
            periods = len(values) - 1
        
        if periods < 1:
            return np.nan
        
        start_value = values.iloc[0]
        end_value = values.iloc[-1]
        
        if start_value == 0 or start_value is None or end_value is None:
            return np.nan
        
        if start_value < 0 and end_value > 0:
            # Handle sign change
            return np.nan
        
        cagr = ((end_value / abs(start_value)) ** (1 / periods) - 1) * 100
        
        return cagr if not np.isinf(cagr) else np.nan
    
    def calculate_all_growth_rates(self) -> pd.DataFrame:
        """
        Calculate year-over-year growth rates for all metrics.
        
        Returns:
            DataFrame with growth rates for each metric
        """
        growth_df = pd.DataFrame(index=self.df.index)
        
        for col in self.df.columns:
            growth_col = f"{col} YoY Growth"
            growth_df[growth_col] = self.calculate_yoy_growth(col)
        
        return growth_df.dropna(how='all')
    
    def identify_trends(self, metric: str, threshold: float = 0.05) -> Dict[str, any]:
        """
        Identify trend direction and strength for a metric.
        
        Args:
            metric: Column name to analyze
            threshold: Minimum change to consider significant (default: 5%)
            
        Returns:
            Dictionary with trend information
        """
        if metric not in self.df.columns:
            return {
                "trend": "insufficient_data",
                "strength": 0.0,
                "avg_growth": 0.0,
                "periods": 0
            }
        
        values = self.df[metric].dropna()
        
        if len(values) < 2:
            return {
                "trend": "insufficient_data",
                "strength": 0.0,
                "avg_growth": 0.0,
                "periods": 0
            }
        
        # Calculate average growth rate
        growth_rates = self.calculate_yoy_growth(metric).dropna()
        
        if len(growth_rates) == 0:
            return {
                "trend": "no_change",
                "strength": 0.0,
                "avg_growth": 0.0,
                "periods": 0
            }
        
        avg_growth = growth_rates.mean()
        
        # Determine trend
        if avg_growth > threshold:
            trend = "increasing"
        elif avg_growth < -threshold:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Calculate strength (coefficient of variation)
        if len(growth_rates) > 1:
            strength = abs(avg_growth) / (growth_rates.std() + 1e-10)
        else:
            strength = abs(avg_growth) / 100
        
        return {
            "trend": trend,
            "strength": min(strength, 1.0),  # Cap at 1.0
            "avg_growth": avg_growth,
            "periods": len(growth_rates)
        }
    
    def get_trend_summary(self) -> pd.DataFrame:
        """
        Get trend summary for all metrics.
        
        Returns:
            DataFrame with trend information for each metric
        """
        summary_data = []
        
        for col in self.df.columns:
            trend_info = self.identify_trends(col)
            cagr = self.calculate_cagr(col)
            
            # Use .get() with defaults for safety
            summary_data.append({
                "Metric": col,
                "Trend": trend_info.get("trend", "unknown"),
                "Avg Growth %": trend_info.get("avg_growth", 0.0),
                "CAGR %": cagr if not pd.isna(cagr) else 0.0,
                "Strength": trend_info.get("strength", 0.0)
            })
        
        return pd.DataFrame(summary_data)
