# financial4all Ratio Formulas

This document describes the financial ratio and metric formulas used in financial4all and their alignment with EdgarTools and standard practice.

## FinancialRatios (`financial4all/financials/ratios.py`)

### Profitability Ratios (as % of Revenue)

| Ratio | Formula | Notes |
|-------|---------|-------|
| Gross Profit Margin | Gross Profit / Revenue × 100 | Standard |
| Operating Profit Margin | Operating Income / Revenue × 100 | Standard |
| Net Profit Margin | Net Income / Revenue × 100 | Standard |

### Liquidity Ratios

| Ratio | Formula | EdgarTools |
|-------|---------|------------|
| Current Ratio | Current Assets / Current Liabilities | ✓ Matches |

### Efficiency Ratios

| Ratio | Formula | Notes |
|-------|---------|-------|
| Asset Turnover | Revenue / Total Assets | Period-aligned via common_index |

### Leverage Ratios

| Ratio | Formula | EdgarTools |
|-------|---------|------------|
| Debt-to-Equity | Total Liabilities / Stockholders Equity | N/A (EdgarTools has debt-to-assets only) |
| Debt-to-Assets | Total Liabilities / Total Assets | ✓ Matches |

### Return Ratios (as % of Base)

| Ratio | Formula | Notes |
|-------|---------|-------|
| Return on Assets (ROA) | Net Income / Total Assets × 100 | End-of-period balance sheet |
| Return on Equity (ROE) | Net Income / Stockholders Equity × 100 | End-of-period balance sheet |

### Cash Ratios

| Ratio | Formula | EdgarTools |
|-------|---------|------------|
| Free Cash Flow | Operating Cash Flow − \|CapEx\| | ✓ Matches |

CapEx is typically negative (outflow) in XBRL; `abs()` ensures correct subtraction regardless of sign convention.

---

## ProfitabilityAnalyzer (`financial4all/analysis/profitability_analyzer.py`)

### Growth and Margins

| Metric | Formula | Notes |
|-------|---------|-------|
| Revenue Y/Y % Change | (Current − Previous) / Previous | Decimals (0.10 = 10%) |
| Outstanding Shares Y/Y % Change | (Current − Previous) / Previous | Same as above |
| Gross Margin | Gross Profit / Revenue | Decimal (0.50 = 50%) |
| Operating Margin | Operating Income / Revenue | Decimal |
| Tax rate | Income Tax Expense / Income Before Taxes | Decimal |
| D&A % of Sales | Depreciation & Amortization / Revenue | Decimal |
| CapEx % of Sales | CapEx / Revenue | CapEx is typically negative (outflow) |
| Receivables % of Sales | Receivables / Revenue | Decimal |
| Inventory % of Sales | Inventory / Revenue | Decimal |
| Payables % of Sales | Payables / Revenue | Decimal |

### Working Capital

| Metric | Formula |
|-------|---------|
| Change in WC | (Rec + Inv − Pay) current − (Rec + Inv − Pay) previous |

---

## CommonSizeGenerator (`financial4all/analysis/common_size.py`)

| Statement | Base | Formula |
|-----------|------|---------|
| Income Statement | Revenue | Each line / Revenue × 100 |
| Balance Sheet | Total Assets | Each line / Total Assets × 100 |
| Cash Flow | Revenue (or first column) | Each line / Base × 100 |

---

## TrendAnalyzer (`financial4all/analysis/trend_analyzer.py`)

| Metric | Formula |
|--------|---------|
| YoY Growth | `pct_change(periods=1) × 100` |
| CAGR | `(End/Start)^(1/periods) − 1` × 100 |

---

## Edge Cases

- **Division by zero**: Returns `NaN` via `.replace([np.inf, -np.inf], np.nan)` or explicit zero checks.
- **Negative denominators**: ROE/ROA with negative equity/assets can produce meaningful values; we compute them. Some analysts prefer NaN for negative equity—financial4all returns the computed value.
- **Missing data**: Ratios are computed only where required columns exist; missing metrics yield NaN.

---

## EdgarTools Alignment Summary

EdgarTools provides raw financial metrics and a small set of ratios via `get_financial_metrics()`:

- **current_ratio** = Current Assets / Current Liabilities ✓
- **debt_to_assets** = Total Liabilities / Total Assets ✓
- **free_cash_flow** = Operating Cash Flow − |CapEx| ✓

Our implementations match these formulas. EdgarTools does not compute margins, ROA, ROE, asset turnover, or debt-to-equity; financial4all uses standard definitions for those.
