# financial4all/analysis/excel_exporter.py
"""
Excel export functionality for financial analysis reports.

This module provides functionality for exporting comprehensive financial
analysis reports to Excel with professional formatting.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
from io import BytesIO

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

from financial4all.analysis.report_generator import FinancialAnalysisReport
from financial4all.core import log

if not OPENPYXL_AVAILABLE:
    log.warning("openpyxl not available. Excel export will not work.")


class ExcelExporter:
    """
    Exports financial analysis reports to Excel.
    
    This class handles formatting, multi-sheet workbooks, and professional
    presentation of financial data.
    """
    
    def export_analysis(
        self,
        report: FinancialAnalysisReport,
        file_path_or_buffer: Union[str, BytesIO],
        include_charts: bool = False
    ) -> None:
        """
        Export analysis to Excel with formatting.
        
        Args:
            report: FinancialAnalysisReport object
            file_path_or_buffer: File path (string) or BytesIO buffer
            include_charts: Whether to include charts (default: False)
        """
        if not OPENPYXL_AVAILABLE:
            raise ImportError("openpyxl is required for Excel export. Install it with: pip install openpyxl")
        
        # Generate all report data
        report_data = report.generate_report()
        
        # Create workbook
        wb = openpyxl.Workbook()
        
        # Remove default sheet
        if "Sheet" in wb.sheetnames:
            wb.remove(wb["Sheet"])
        
        # 1. Summary Sheet
        self._create_summary_sheet(wb, report, report_data)
        
        # 2. Income Statement Sheet
        multi_year = report_data.get("multi_year_comparison", {})
        if "income_statement" in multi_year:
            self._create_statement_sheet(
                wb, "Income Statement", multi_year["income_statement"]
            )
        
        # 3. Balance Sheet Sheet
        if "balance_sheet" in multi_year:
            self._create_statement_sheet(
                wb, "Balance Sheet", multi_year["balance_sheet"]
            )
        
        # 4. Cash Flow Sheet
        if "cash_flow" in multi_year:
            self._create_statement_sheet(
                wb, "Cash Flow", multi_year["cash_flow"]
            )
        
        # 5. Ratios Sheet
        ratios_df = report_data.get("ratios", pd.DataFrame())
        if not ratios_df.empty:
            self._create_ratios_sheet(wb, ratios_df)
        
        # 6. Trends Sheet
        trends_df = report_data.get("trends", pd.DataFrame())
        if not trends_df.empty:
            self._create_trends_sheet(wb, trends_df)
        
        # 7. Common Size Sheet
        common_size = report_data.get("common_size", {})
        if common_size:
            self._create_common_size_sheet(wb, common_size)
        
        # Save workbook
        wb.save(file_path_or_buffer)
    
    def _create_summary_sheet(self, wb, report: FinancialAnalysisReport, report_data: dict) -> None:
        """Create summary sheet with key metrics."""
        ws = wb.create_sheet("Summary", 0)
        
        summary = report.get_summary_metrics()
        
        # Header
        ws["A1"] = "Financial Analysis Summary"
        ws["A1"].font = Font(bold=True, size=16)
        ws.merge_cells("A1:B1")
        
        row = 3
        
        # Company Information
        ws[f"A{row}"] = "Company Information"
        ws[f"A{row}"].font = Font(bold=True, size=12)
        row += 1
        
        ws[f"A{row}"] = "Company Name:"
        ws[f"B{row}"] = summary.get("company_name", "N/A")
        row += 1
        
        ws[f"A{row}"] = "Ticker:"
        ws[f"B{row}"] = summary.get("ticker", "N/A")
        row += 1
        
        ws[f"A{row}"] = "CIK:"
        ws[f"B{row}"] = summary.get("cik", "N/A")
        row += 1
        
        ws[f"A{row}"] = "Analysis Date:"
        ws[f"B{row}"] = summary.get("analysis_date", "N/A")
        row += 2
        
        # Key Metrics
        ws[f"A{row}"] = "Key Financial Metrics (Most Recent Period)"
        ws[f"A{row}"].font = Font(bold=True, size=12)
        row += 1
        
        metrics = [
            ("Revenue", summary.get("revenue")),
            ("Net Income", summary.get("net_income")),
            ("Gross Profit", summary.get("gross_profit")),
            ("Operating Income", summary.get("operating_income")),
            ("Total Assets", summary.get("total_assets")),
            ("Total Liabilities", summary.get("total_liabilities")),
            ("Stockholders Equity", summary.get("equity")),
            ("Operating Cash Flow", summary.get("operating_cash_flow")),
        ]
        
        for label, value in metrics:
            if value is not None and not pd.isna(value):
                ws[f"A{row}"] = f"{label}:"
                if isinstance(value, (int, float)) and abs(value) > 1000:
                    ws[f"B{row}"] = f"${value:,.0f}"
                else:
                    ws[f"B{row}"] = value
                row += 1
        
        row += 1
        
        # Key Ratios
        ws[f"A{row}"] = "Key Ratios (Most Recent Period)"
        ws[f"A{row}"].font = Font(bold=True, size=12)
        row += 1
        
        ratio_metrics = [
            ("Gross Profit Margin", summary.get("gross_margin"), "%"),
            ("Net Profit Margin", summary.get("net_margin"), "%"),
            ("Return on Assets (ROA)", summary.get("roa"), "%"),
            ("Return on Equity (ROE)", summary.get("roe"), "%"),
        ]
        
        for label, value, unit in ratio_metrics:
            if value is not None and not pd.isna(value):
                ws[f"A{row}"] = f"{label}:"
                ws[f"B{row}"] = f"{value:.2f}{unit}"
                row += 1
        
        # Format columns
        ws.column_dimensions["A"].width = 25
        ws.column_dimensions["B"].width = 20
    
    def _create_statement_sheet(self, wb, sheet_name: str, df: pd.DataFrame) -> None:
        """Create a financial statement sheet."""
        ws = wb.create_sheet(sheet_name)
        
        # Write header
        ws["A1"] = sheet_name
        ws["A1"].font = Font(bold=True, size=14)
        
        # Write DataFrame
        # Reset index to include dates as first column
        df_write = df.reset_index()
        
        # Write column headers
        for col_idx, col_name in enumerate(df_write.columns, start=1):
            cell = ws.cell(row=2, column=col_idx)
            cell.value = col_name
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.font = Font(bold=True, color="FFFFFF")
            cell.alignment = Alignment(horizontal="center")
        
        # Write data
        for row_idx, row_data in enumerate(df_write.values, start=3):
            for col_idx, value in enumerate(row_data, start=1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if isinstance(value, (int, float)):
                    if abs(value) > 1000:
                        cell.value = value
                        cell.number_format = '$#,##0'
                    else:
                        cell.value = value
                        cell.number_format = '#,##0.00'
                else:
                    cell.value = value
        
        # Format date column
        if "end" in df_write.columns:
            date_col_idx = df_write.columns.get_loc("end") + 1
            for row in range(3, len(df_write) + 3):
                cell = ws.cell(row=row, column=date_col_idx)
                if cell.value:
                    try:
                        cell.number_format = "YYYY-MM-DD"
                    except:
                        pass
        
        # Auto-adjust column widths
        for col_idx, col_name in enumerate(df_write.columns, start=1):
            max_length = max(
                len(str(col_name)),
                max((len(str(cell.value)) for cell in ws[get_column_letter(col_idx)] if cell.value), default=0)
            )
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_length + 2, 50)
        
        # Freeze first row and column
        ws.freeze_panes = "B3"
    
    def _create_ratios_sheet(self, wb, ratios_df: pd.DataFrame) -> None:
        """Create ratios sheet."""
        ws = wb.create_sheet("Ratios")
        
        # Write header
        ws["A1"] = "Financial Ratios"
        ws["A1"].font = Font(bold=True, size=14)
        
        # Write DataFrame
        df_write = ratios_df.reset_index()
        
        # Write column headers
        for col_idx, col_name in enumerate(df_write.columns, start=1):
            cell = ws.cell(row=2, column=col_idx)
            cell.value = col_name
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.font = Font(bold=True, color="FFFFFF")
            cell.alignment = Alignment(horizontal="center")
        
        # Write data
        for row_idx, row_data in enumerate(df_write.values, start=3):
            for col_idx, value in enumerate(row_data, start=1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if isinstance(value, (int, float)) and not pd.isna(value):
                    cell.value = value
                    # Format as percentage for ratio columns
                    if "%" in str(df_write.columns[col_idx - 1]):
                        cell.number_format = '0.00%'
                    else:
                        cell.number_format = '#,##0.00'
                else:
                    cell.value = value
        
        # Auto-adjust column widths
        for col_idx in range(1, len(df_write.columns) + 1):
            max_length = max(
                len(str(df_write.columns[col_idx - 1])),
                max((len(str(cell.value)) for cell in ws[get_column_letter(col_idx)] if cell.value), default=0)
            )
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_length + 2, 50)
        
        ws.freeze_panes = "B3"
    
    def _create_trends_sheet(self, wb, trends_df: pd.DataFrame) -> None:
        """Create trends sheet with growth rates."""
        ws = wb.create_sheet("Trends")
        
        # Write header
        ws["A1"] = "Year-over-Year Growth Rates"
        ws["A1"].font = Font(bold=True, size=14)
        
        # Write DataFrame
        df_write = trends_df.reset_index()
        
        # Write column headers
        for col_idx, col_name in enumerate(df_write.columns, start=1):
            cell = ws.cell(row=2, column=col_idx)
            cell.value = col_name
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.font = Font(bold=True, color="FFFFFF")
            cell.alignment = Alignment(horizontal="center")
        
        # Write data
        for row_idx, row_data in enumerate(df_write.values, start=3):
            for col_idx, value in enumerate(row_data, start=1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if isinstance(value, (int, float)) and not pd.isna(value):
                    cell.value = value / 100  # Convert percentage to decimal for Excel
                    cell.number_format = '0.00%'
                else:
                    cell.value = value
        
        # Auto-adjust column widths
        for col_idx in range(1, len(df_write.columns) + 1):
            max_length = max(
                len(str(df_write.columns[col_idx - 1])),
                max((len(str(cell.value)) for cell in ws[get_column_letter(col_idx)] if cell.value), default=0)
            )
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_length + 2, 50)
        
        ws.freeze_panes = "B3"
    
    def _create_common_size_sheet(self, wb, common_size: dict) -> None:
        """Create common-size statements sheet."""
        ws = wb.create_sheet("Common Size")
        
        # Write header
        ws["A1"] = "Common-Size Financial Statements"
        ws["A1"].font = Font(bold=True, size=14)
        
        row = 3
        
        # Income Statement Common Size
        if "income_statement" in common_size:
            ws[f"A{row}"] = "Income Statement (% of Revenue)"
            ws[f"A{row}"].font = Font(bold=True, size=12)
            row += 1
            
            is_df = common_size["income_statement"].reset_index()
            
            # Write headers
            for col_idx, col_name in enumerate(is_df.columns, start=1):
                cell = ws.cell(row=row, column=col_idx)
                cell.value = col_name
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                cell.font = Font(bold=True, color="FFFFFF")
            
            row += 1
            
            # Write data
            for _, row_data in is_df.iterrows():
                for col_idx, value in enumerate(row_data, start=1):
                    cell = ws.cell(row=row, column=col_idx)
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        cell.value = value / 100  # Convert to decimal
                        cell.number_format = '0.00%'
                    else:
                        cell.value = value
                row += 1
            
            row += 2
        
        # Balance Sheet Common Size
        if "balance_sheet" in common_size:
            ws[f"A{row}"] = "Balance Sheet (% of Total Assets)"
            ws[f"A{row}"].font = Font(bold=True, size=12)
            row += 1
            
            bs_df = common_size["balance_sheet"].reset_index()
            
            # Write headers
            for col_idx, col_name in enumerate(bs_df.columns, start=1):
                cell = ws.cell(row=row, column=col_idx)
                cell.value = col_name
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                cell.font = Font(bold=True, color="FFFFFF")
            
            row += 1
            
            # Write data
            for _, row_data in bs_df.iterrows():
                for col_idx, value in enumerate(row_data, start=1):
                    cell = ws.cell(row=row, column=col_idx)
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        cell.value = value / 100
                        cell.number_format = '0.00%'
                    else:
                        cell.value = value
                row += 1
        
        # Auto-adjust column widths
        for col_idx in range(1, 10):
            ws.column_dimensions[get_column_letter(col_idx)].width = 20
