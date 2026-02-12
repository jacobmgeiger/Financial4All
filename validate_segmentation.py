# validate_segmentation.py
# Validation script for EdgarTools Segmentation Alignment plan.
# Checks NVDA, ANF, MSFT segment extraction and main Revenue consistency.

from financial4all import Company, set_identity

# SEC requires User-Agent with contact; use env or default
import os
if os.getenv("F4A_SEC_EMAIL"):
    set_identity(os.getenv("F4A_SEC_EMAIL"))
else:
    set_identity("financial4all@example.com")


def validate_ticker(ticker: str, use_cache: bool = False) -> dict:
    """Load company and return standard + detailed income statement metrics."""
    try:
        company = Company(ticker)
        financials = company.get_financials(use_cache=use_cache)
        income = financials.get("income_statement")
        if not income:
            return {"error": f"No income statement for {ticker}"}
        std_df = income.to_dataframe(view="standard")
        det_df = income.to_dataframe(view="detailed")
        return {
            "ticker": ticker,
            "standard_columns": list(std_df.columns) if std_df is not None else [],
            "detailed_columns": list(det_df.columns) if det_df is not None else [],
            "standard_revenue": std_df["Revenue"].tolist() if "Revenue" in std_df.columns and std_df is not None else None,
            "periods": std_df.index.tolist() if std_df is not None and hasattr(std_df.index, "tolist") else [],
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def main():
    for ticker in ["NVDA", "ANF", "MSFT"]:
        print(f"\n--- {ticker} ---")
        r = validate_ticker(ticker, use_cache=False)
        if "error" in r:
            print(f"  Error: {r['error']}")
            continue
        print(f"  Standard columns: {len(r['standard_columns'])}")
        print(f"  Detailed columns: {len(r['detailed_columns'])}")
        # Segment rows = detailed - standard (extra columns from segment breakdowns)
        seg_cols = [c for c in r["detailed_columns"] if c not in r["standard_columns"]]
        print(f"  Segment rows (detailed-only): {seg_cols[:15]}{'...' if len(seg_cols) > 15 else ''}")
        if r.get("standard_revenue"):
            rev = r["standard_revenue"]
            print(f"  Revenue (latest): {rev[0] if rev else 'N/A'}")


if __name__ == "__main__":
    import logging
    logging.getLogger("financial4all").setLevel(logging.ERROR)
    main()
