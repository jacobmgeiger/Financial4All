import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import json
import dash_bootstrap_components as dbc
import io
import zipfile
from typing import Optional
from financial4all import Company
from financial4all.analysis import ProfitabilityAnalyzer
from financial4all.analysis.excel_exporter import ExcelExporter
from financial4all.financials import FinancialRatios
from financial4all.sec.client import SECClient

# app.py
# This script creates a comprehensive Dash web application for financial analysis.
# It allows users to input a ticker symbol, fetch financial data from the SEC,
# visualize trends, view standardized statements with interactive formula switching, and export data.

# --- NEW: Metric Definitions for Tooltips ---
METRIC_DEFINITIONS = {
    "Revenue": "The total amount of money a company generates from its primary business activities, such as selling goods or services.",
    "Cost of Revenue": "The direct costs associated with producing the goods or services a company sells. This includes materials and direct labor.",
    "Gross Profit": "The profit a company makes after deducting the costs associated with making and selling its products. It's calculated as Revenue minus Cost of Revenue.",
    "R&D Expenses": "Research and Development Expenses. The costs a company incurs for activities that create or improve its products and services.",
    "SG&A Expenses": "Selling, General, and Administrative Expenses. The operational costs of running a business that are not directly related to producing a product or service. Includes salaries, marketing, and rent.",
    "Other Operating Expenses": "Miscellaneous expenses related to a company's main business operations that don't fit into other categories like R&D or SG&A.",
    "Operating Income": "A company's profit after subtracting operating expenses from gross profit. It shows how much profit a company generates from its core business operations.",
    "Interest Income": "The income a company earns from its cash holdings and investments, such as interest from savings accounts or bonds.",
    "Interest Expense": "The cost a company pays for its borrowed funds, such as loans and bonds.",
    "Other, net": "Other income and expense items, net of each other. Includes miscellaneous non-operating items.",
    "Other income (expense), net": "Calculated as: Interest Income - Interest Expense + Other, net. Represents net non-operating income/expense.",
    "Income from Equity Method Investments": "The share of profit or loss that a company reports from its investments in other companies where it has significant influence but not full control.",
    "Other Non-operating Income (Expense)": "Income or expenses that do not come from a company's core business operations, such as gains or losses from selling assets.",
    "Income Before Taxes": "A company's total profit before any income taxes are deducted. It is a measure of a company's profitability.",
    "Taxes": "The amount of money a company pays to the government as income tax.",
    "Net Income": "The company's total profit after all expenses, including taxes, have been deducted from revenue. Also known as the 'bottom line'.",
    "Outstanding Shares Basic": "The weighted average number of common shares outstanding during the period, used to calculate basic earnings per share.",
    "Outstanding Shares Diluted": "The weighted average number of common shares outstanding during the period, adjusted for all dilutive potential common shares, used to calculate diluted earnings per share.",
    "Basic EPS": "Basic Earnings Per Share. A company's net income divided by the number of its outstanding common shares. It shows how much profit is available to each shareholder.",
    "Diluted EPS": "Diluted Earnings Per Share. A more conservative measure of earnings per share that includes the impact of all potential shares that could be created, such as from stock options and convertible bonds.",
}

# --- App Initialization ---
# Initialize the Dash app with a dark theme from Dash Bootstrap Components for a professional look.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# --- App Layout ---
# The layout is structured using HTML Divs and Dash Core Components.
# It includes input fields, selectors, graphs, and buttons, styled for a dark theme.
app.layout = html.Div(
    [
        html.H1("Financial 4 All", style={"textAlign": "center", "color": "#E0E0E0"}),
        # --- Ticker Input Section ---
        html.Div(
            [
                html.Label("Ticker Symbol:", style={"color": "#B0B0B0"}),
                dcc.Input(
                    id="ticker-input",
                    type="text",
                    value="",
                    placeholder="Enter Ticker Symbol (e.g., AAPL)",
                    style={
                        "width": "100%",
                        "backgroundColor": "#333333",
                        "color": "#E0E0E0",
                        "border": "1px solid #555555",
                    },
                ),
                html.Div(
                    id="status-output", style={"marginTop": "10px", "color": "#B0B0B0"}
                ),
            ],
            style={"width": "50%", "margin": "auto", "padding": "10px"},
        ),
        # --- Hidden div to trigger the main data loading callback ---
        html.Div(id="load-trigger", style={"display": "none"}),
        html.Hr(style={"borderColor": "#555555"}),
        # --- Main Content Area with Loading Spinner ---
        # The dcc.Loading component wraps the main part of the app.
        # It will display a spinner automatically whenever a callback updating
        # one of its children is running. This provides clear feedback to the
        # user that data is being loaded.
        dcc.Loading(
            id="loading-indicator",
            type="default",
            children=html.Div(
                id="main-content",
                children=[
                    # --- NEW: Key Ratios Display ---
                    html.Div(id="key-ratios-display", style={"marginBottom": "20px"}),
                    html.Hr(style={"borderColor": "#555555"}),
                    # --- Action Buttons ---
                    html.Div(
                        [
                            dbc.Button(
                                "Export XBRL to CSV",
                                id="export-excel-button",
                                className="me-2",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#007BFF",
                                    "color": "white",
                                },
                            ),
                            dbc.Button(
                                "Download All 10-Ks",
                                id="download-10k-button",
                                className="me-2",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#17A2B8",
                                    "color": "white",
                                },
                            ),
                            dbc.Button(
                                "Generate Comprehensive Analysis",
                                id="generate-analysis-btn",
                                className="me-2",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#28A745",
                                    "color": "white",
                                },
                            ),
                        ],
                        style={"textAlign": "center", "padding": "10px"},
                    ),
                    # --- Unit Scale Selection ---
                    html.Div(
                        [
                            html.Label("Display Units:", style={"color": "#B0B0B0", "marginRight": "10px"}),
                            dcc.Dropdown(
                                id="unit-scale-dropdown",
                                options=[
                                    {"label": "Auto-detect", "value": "auto"},
                                    {"label": "Millions", "value": "millions"},
                                    {"label": "Billions", "value": "billions"},
                                    {"label": "Thousands", "value": "thousands"},
                                    {"label": "Raw Values", "value": "raw"},
                                ],
                                value="millions",  # Default to millions
                                clearable=False,
                                style={
                                    "width": "200px",
                                    "backgroundColor": "#333333",
                                    "color": "#E0E0E0",
                                    "display": "inline-block",
                                },
                            ),
                        ],
                        style={
                            "width": "80%",
                            "margin": "auto",
                            "padding": "10px",
                            "textAlign": "left",
                        },
                    ),
                    # --- Output Sections ---
                    html.Div(id="standard-is-output", style={"marginTop": "20px"}),
                ],
            ),
        ),
        # --- Download and Store components are placed outside the main visible layout ---
        dcc.Download(id="download-dataframe-excel"),
        dcc.Download(id="download-10k-zip"),
        dcc.Download(id="download-analysis-excel"),
        # --- Data Stores ---
        # dcc.Store components are used to store data in the user's browser,
        # avoiding the need for global variables and ensuring data persists between callbacks.
        dcc.Store(id="current-df-store"),
        dcc.Store(id="all-plottable-metrics-store"),
        dcc.Store(id="standard-is-store"),
        dcc.Store(id="alternatives-store"),  # NEW: Store for alternative calculations
        dcc.Store(id="key-ratios-store"),  # NEW: Store for calculated key ratios
        dcc.Store(
            id="is-selections-store", data={}
        ),  # NEW: Store for user's dropdown selections
        dcc.Store(
            id="standard-metrics-store"
        ),  # NEW: Store for the list of standard metrics
        dcc.Store(
            id="unit-scale-store", data="millions"
        ),  # NEW: Store for user's unit scale preference
    ],
    style={
        "backgroundColor": "#1E1E1E",
        "color": "#E0E0E0",
        "padding": "20px",
        "fontFamily": "Arial, sans-serif",
    },
)

# --- Callbacks ---


# --- NEW: Step 1 - Immediate Feedback Callback ---
@app.callback(
    [Output("status-output", "children"), Output("load-trigger", "children")],
    [Input("ticker-input", "n_submit")],
    [State("ticker-input", "value")],
    prevent_initial_call=True,
)
def on_ticker_submit(n_submit, ticker):
    """
    This callback provides immediate feedback when the user submits a ticker.
    It validates the ticker and triggers the main data loading callback.
    """
    if not ticker or not ticker.strip():
        return html.P("Please enter a ticker symbol.", style={"color": "orange"}), None

    ticker_upper = ticker.strip().upper()
    return (
        html.P(f"Loading data for {ticker_upper}...", style={"color": "#007BFF"}),
        ticker_upper,
    )


# --- UPDATED: Step 2 - Main Data Loading Callback ---
@app.callback(
    [
        Output("status-output", "children", allow_duplicate=True),
        Output("current-df-store", "data"),
        Output("all-plottable-metrics-store", "data"),
        Output("standard-is-store", "data"),
        Output("alternatives-store", "data"),
        Output("is-selections-store", "data"),
        Output("standard-metrics-store", "data"),
        Output("key-ratios-store", "data"),  # NEW: Output for key ratios
    ],
    [Input("load-trigger", "children")],  # Triggered by the first callback
    prevent_initial_call=True,
)
def on_ticker_change(ticker_upper):
    """
    This callback performs the heavy data lifting after being triggered by the
    initial feedback callback.
    """
    if not ticker_upper:
        # This case should ideally not be hit if the first callback validates the ticker
        raise dash.exceptions.PreventUpdate

    try:
        # Use the new Company API directly
        company = Company(ticker_upper)
        financials = company.get_financials()
        
        income_statement = financials["income_statement"]
        balance_sheet = financials["balance_sheet"]
        cash_flow = financials["cash_flow"]
        
        # Get income statement DataFrame
        standard_is_df = income_statement.to_dataframe()
        
        if standard_is_df is None or standard_is_df.empty:
            status_message = html.P(
                f"No data retrieved for {ticker_upper}.", style={"color": "orange"}
            )
            return status_message, None, [], None, None, {}, [], None
        
        # For backward compatibility, create df_merged (all metrics)
        # This combines income statement with other available metrics
        df_merged = standard_is_df.copy()
        
        # Add balance sheet and cash flow metrics if available
        if balance_sheet:
            bs_df = balance_sheet.to_dataframe()
            if not bs_df.empty:
                df_merged = df_merged.join(bs_df, how='outer', rsuffix='_bs')
        
        if cash_flow:
            # Pass balance sheet and income statement for CapEx fallback calculation and validation
            bs_df_for_capex = balance_sheet.to_dataframe() if balance_sheet else None
            is_df_for_capex = standard_is_df if income_statement else None
            cf_df = cash_flow.to_dataframe(bs_df=bs_df_for_capex, is_df=is_df_for_capex)
            if not cf_df.empty:
                df_merged = df_merged.join(cf_df, how='outer', rsuffix='_cf')
        
        # Calculate ratios
        ratios = FinancialRatios(income_statement, balance_sheet, cash_flow)
        key_ratios_df = ratios.calculate_all_ratios()
        
        # Create alternatives dict (empty for now - could be enhanced)
        alternatives = {}
        standard_metrics = list(standard_is_df.columns) if not standard_is_df.empty else []

        # --- Logic to determine best default formula based on non-zero count ---
        default_selections = {}
        # `standard_is_df` has years on the index and metrics on the columns.
        for metric_name in standard_is_df.columns:
            all_options = []

            # 1. Add the primary series (the column from the dataframe)
            primary_series = standard_is_df[metric_name]
            all_options.append({"source": "default", "series": primary_series})

            # 2. Add alternative series from the alternatives dictionary
            metric_alternatives = alternatives.get(metric_name, [])
            for alt in metric_alternatives:
                alt_series = pd.Series(alt["values"])
                all_options.append({"source": alt["source"], "series": alt_series})

            # 3. Score and find the best one by max non-zero values
            if all_options:
                best_option = max(
                    all_options, key=lambda opt: (opt["series"].fillna(0) != 0).sum()
                )
                default_selections[metric_name] = best_option["source"]

        # Store all metrics for the comprehensive graphing view
        all_plottable_metrics = [
            {
                "label": col,
                "value": col,
                "fill_rate": df_merged[col].count() / len(df_merged),
            }
            for col in df_merged.columns
            if col != "end"
        ]

        # The primary display df for the table is the transposed version of the standard IS
        transposed_df = standard_is_df.T.reset_index().rename(
            columns={"index": "Metric"}
        )

        # Get company info directly from Company instance
        info = company.company_info
        status_message = html.Div(
            [
                html.P(f"Company: {info['title']} (CIK: {info['cik']})"),
                html.P(f"Data loaded for {ticker_upper}."),
            ]
        )
        # Safely convert DataFrames to JSON
        # Empty DataFrames can still be converted to JSON (they'll just have empty data arrays)
        try:
            df_merged_json = df_merged.to_json(date_format="iso", orient="split") if not df_merged.empty else None
        except Exception:
            df_merged_json = None
        
        try:
            transposed_df_json = transposed_df.to_json(date_format="iso", orient="split") if not transposed_df.empty else None
        except Exception:
            transposed_df_json = None
        
        try:
            key_ratios_json = key_ratios_df.to_json(date_format="iso", orient="split") if not key_ratios_df.empty else None
        except Exception:
            key_ratios_json = None
        
        return (
            status_message,
            df_merged_json,
            all_plottable_metrics,
            transposed_df_json,
            alternatives,
            default_selections,
            standard_metrics,
            key_ratios_json,  # NEW: Store the ratios
        )
    except Exception as e:
        status_message = html.P(
            f"Error loading data for {ticker_upper}: {e}", style={"color": "red"}
        )
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return status_message, None, [], None, None, {}, [], None


# --- NEW: Callback to display key ratio cards ---
@app.callback(
    Output("key-ratios-display", "children"),
    [Input("key-ratios-store", "data")],
    [State("ticker-input", "value")],
)
def update_key_ratios_display(ratios_json, ticker):
    """
    Creates and displays a row of cards for key financial ratios.
    """
    try:
        if not ratios_json:
            return []

        try:
            # Parse JSON - Dash Store components return the data as-is (dict or string)
            if isinstance(ratios_json, str):
                # If it's a string, parse it first, then use StringIO to avoid file path interpretation
                ratios_data = json.loads(ratios_json)
                # Use StringIO to make pandas treat it as a string, not a file path
                ratios_df = pd.read_json(io.StringIO(json.dumps(ratios_data)), orient="split")
            elif isinstance(ratios_json, dict):
                # Already parsed - construct DataFrame directly
                ratios_df = pd.DataFrame(
                    data=ratios_json.get("data", []),
                    columns=ratios_json.get("columns", []),
                    index=ratios_json.get("index", [])
                )
            else:
                return []
        except (ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
            # Invalid JSON or empty data
            return []

        if ratios_df is None or ratios_df.empty or len(ratios_df.columns) == 0:
            return []

        cards = []
        for ratio_name in ratios_df.columns:
            # Get the series for this ratio
            series = ratios_df[ratio_name].dropna()
            if series.empty:
                continue

            try:
                # Ensure chronological order: sort by index to get oldest to newest (left to right)
                # Try to parse index as dates for proper sorting
                try:
                    # Convert index to datetime if possible
                    date_index = pd.to_datetime(series.index, errors='coerce')
                    if not date_index.isna().all():
                        # Sort by date ascending (oldest first, newest last)
                        sorted_indices = date_index.sort_values().index
                        sorted_series = series.reindex(sorted_indices)
                    else:
                        # Index is not dates, try sorting as strings/numbers
                        # Sort index in ascending order (assumes oldest is smaller)
                        sorted_series = series.sort_index(ascending=True)
                except (ValueError, TypeError, AttributeError):
                    # Fallback: sort index ascending
                    sorted_series = series.sort_index(ascending=True)
                
                # After sorting, the series is now oldest to newest
                # Get the latest value (last item = newest)
                latest_value = float(sorted_series.iloc[-1])
                
                # Get values in chronological order (oldest to newest) for sparkline
                sparkline_values = sorted_series.values.tolist()
                
                if len(sparkline_values) == 0:
                    continue

                # Create x-axis positions (0, 1, 2, ...) for left-to-right display
                # Position 0 = oldest (left), last position = newest (right)
                x_positions = list(range(len(sparkline_values)))

                # Create a sparkline figure with data ordered left-to-right (oldest to newest)
                # The first value in sparkline_values is oldest, last value is newest
                sparkline = go.Figure(
                    go.Scatter(
                        x=x_positions,
                        y=sparkline_values,
                        mode="lines",
                        line=dict(color="#007BFF", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(0, 123, 255, 0.2)",
                    )
                )
                sparkline.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(
                        visible=False,
                        range=[0, len(sparkline_values) - 1],  # Explicitly set range: 0 (left) = oldest, max (right) = newest
                        autorange=False,
                    ),
                    yaxis=dict(visible=False),
                    height=50,
                )

                # Format the value display - check if it's a percentage ratio
                if "%" in ratio_name or "Margin" in ratio_name or "RO" in ratio_name:
                    value_display = f"{latest_value:.2f}%"
                else:
                    value_display = f"{latest_value:.2f}"

                card_content = [
                    dbc.CardHeader(f"{ratio_name}"),
                    dbc.CardBody(
                        [
                            html.H4(value_display, className="card-title"),
                            dcc.Graph(figure=sparkline, config={'displayModeBar': False})
                        ]
                    ),
                ]
                # Add vertical margin to cards for spacing between rows
                cards.append(
                    dbc.Col(
                        dbc.Card(card_content, color="dark", outline=True), 
                        width=4,
                        style={"marginBottom": "10px"}
                    )
                )
            except Exception:
                # Skip this ratio if there's an error creating the card
                continue

        if not cards:
            return html.Div("No ratio data available.", style={"color": "#B0B0B0", "textAlign": "center"})
        
        # Return Row with proper spacing between cards
        return dbc.Row(cards, style={"marginBottom": "10px"})
    except Exception as e:
        # Catch any unexpected errors and return empty list
        return html.Div(
            f"Error displaying ratios: {str(e)}",
            style={"color": "#FF6B6B", "textAlign": "center"}
        )


# --- NEW: Callback to update user selections in the store ---
@app.callback(
    Output("is-selections-store", "data", allow_duplicate=True),
    [Input({"type": "metric-dropdown", "index": dash.dependencies.ALL}, "value")],
    [
        State({"type": "metric-dropdown", "index": dash.dependencies.ALL}, "id"),
        State("is-selections-store", "data"),
    ],
    prevent_initial_call=True,
)
def update_selections(values, ids, current_selections):
    """
    When a user changes a dropdown in the income statement, this callback
    updates the central store of selections.
    """
    if not ids:
        return dash.no_update

    for i, dropdown_id in enumerate(ids):
        metric_name = dropdown_id["index"]
        current_selections[metric_name] = values[i]
    return current_selections


# --- NEW: Helper function to apply user selections to the income statement ---
def _apply_user_selections_to_is(standard_is_json, alternatives_json, selections):
    """
    Applies user's alternative formula selections to the standardized income statement DataFrame.
    """
    if not standard_is_json:
        return None

    try:
        # Handle both string and dict formats from Dash Store
        if isinstance(standard_is_json, str):
            df = pd.read_json(io.StringIO(standard_is_json), orient="split")
        elif isinstance(standard_is_json, dict):
            df = pd.DataFrame(
                data=standard_is_json.get("data", []),
                columns=standard_is_json.get("columns", []),
                index=standard_is_json.get("index", [])
            )
        else:
            df = pd.DataFrame()
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        # Invalid JSON
        return None
    
    if df.empty:
        return None
    
    alternatives = alternatives_json or {}
    selections = selections or {}

    # Update the dataframe with any user-selected alternative calculations
    for metric, selection in selections.items():
        if selection and selection != "default":
            # Find the chosen alternative and update the row
            chosen_alt = next(
                (
                    alt
                    for alt in alternatives.get(metric, [])
                    if alt["source"] == selection
                ),
                None,
            )
            if chosen_alt:
                # The values are stored as a dict of {date_str: value}.
                alt_series = pd.Series(chosen_alt["values"])
                # Get the integer index of the metric row to update
                row_idx = df.index[df["Metric"] == metric].tolist()
                if not row_idx:
                    continue
                row_idx = row_idx[0]
                # Update the numeric columns based on the alternative series.
                # The columns in df are the date strings from the JSON.
                for date_str, value in alt_series.items():
                    if date_str in df.columns:
                        df.loc[row_idx, date_str] = value
    return df


# --- Helper function to detect unit scale ---
def _detect_unit_scale(df: pd.DataFrame, user_preference: Optional[str] = None):
    """
    Detect appropriate unit scale for financial data.
    
    Analyzes numeric values to determine if they should be displayed
    in billions, millions, thousands, hundreds, or raw values.
    If user_preference is provided and not "auto", uses that instead.
    
    Args:
        df: DataFrame with financial data (first column is "Metric", rest are numeric)
        user_preference: Optional user preference ("auto", "millions", "billions", "thousands", "raw")
        
    Returns:
        Tuple of (scale_factor, unit_label)
        e.g., (1e6, "millions") means divide by 1e6 and show "(millions)"
    """
    # If user has a preference and it's not "auto", use it
    if user_preference and user_preference != "auto":
        scale_map = {
            "millions": (1e6, "millions"),
            "billions": (1e9, "billions"),
            "thousands": (1e3, "thousands"),
            "raw": (1.0, ""),
        }
        return scale_map.get(user_preference, (1e6, "millions"))  # Default to millions
    
    # Auto-detect based on median value
    # Get all numeric values (skip "Metric" column)
    numeric_values = []
    for col in df.columns[1:]:
        numeric_values.extend(df[col].dropna().abs().tolist())
    
    if not numeric_values:
        return (1e6, "millions")  # Default to millions if no data
    
    median_value = pd.Series(numeric_values).median()
    
    if median_value >= 1e9:
        return (1e9, "billions")
    elif median_value >= 1e6:
        return (1e6, "millions")
    elif median_value >= 1e3:
        return (1e3, "thousands")
    elif median_value >= 100:
        return (100, "hundreds")
    else:
        return (1e6, "millions")  # Default to millions for small values


# --- UPDATED: Callback to generate the interactive income statement ---
@app.callback(
    Output("unit-scale-store", "data"),
    Input("unit-scale-dropdown", "value"),
    prevent_initial_call=False,
)
def update_unit_scale(selected_unit):
    """Update unit scale preference."""
    return selected_unit or "millions"


@app.callback(
    Output("standard-is-output", "children"),
    [
        Input("standard-is-store", "data"),
        Input("is-selections-store", "data"),
        Input("unit-scale-store", "data"),
    ],
    [
        State("alternatives-store", "data"),
        State("ticker-input", "value"),
    ],
    prevent_initial_call=True,
)
def display_standard_is(
    standard_is_json, selections, unit_scale_preference, alternatives_json, ticker
):
    """
    Generates and displays an interactive standardized income statement.
    It now includes dropdowns for metrics with alternative calculation paths.
    """
    if not standard_is_json:
        return html.Div("No income statement data available.", style={"color": "#B0B0B0"})
    
    # Use the new helper function to get the correct DataFrame
    try:
        df_standard = _apply_user_selections_to_is(
            standard_is_json, alternatives_json, selections
        )
    except Exception:
        return html.Div("Error processing income statement.", style={"color": "#FF6B6B"})

    if df_standard is None or df_standard.empty:
        return html.Div("No income statement data available.", style={"color": "#B0B0B0"})

    # --- Calculate profitability ratios using library ---
    # Convert transposed format back to periods-as-index for analyzer
    try:
        # Use original df_standard before date formatting for analysis
        df_for_analysis = df_standard.set_index("Metric").T
        
        # Fetch balance sheet and cash flow DataFrames from Company API
        bs_df = pd.DataFrame()
        cf_df = pd.DataFrame()
        if ticker:
            try:
                company = Company(ticker.strip().upper())
                financials = company.get_financials()
                bs_df = financials["balance_sheet"].to_dataframe() if financials["balance_sheet"] else pd.DataFrame()
                # Pass balance sheet and income statement for CapEx fallback calculation and validation
                is_df_for_capex = df_for_analysis if "Revenue" in df_for_analysis.columns else None
                cf_df = financials["cash_flow"].to_dataframe(bs_df=bs_df, is_df=is_df_for_capex) if financials["cash_flow"] else pd.DataFrame()
            except Exception:
                # If fetching fails, continue with empty DataFrames
                pass
        
        analyzer = ProfitabilityAnalyzer()
        profitability_df = analyzer.calculate_ratios(df_for_analysis, bs_df, cf_df)
        
        # Run validation and log issues
        try:
            from financial4all.analysis.validators import FinancialStatementValidator
            from financial4all.core import log as financial_log
            validator = FinancialStatementValidator()
            validation_issues = validator.validate_all(
                is_df=df_for_analysis,
                bs_df=bs_df,
                cf_df=cf_df
            )
            
            # Log validation issues
            for issue in validation_issues:
                if issue.severity.value == "error":
                    financial_log.error(f"Validation ERROR - {issue.metric} ({issue.period}): {issue.message}")
                elif issue.severity.value == "warning":
                    financial_log.warning(f"Validation WARNING - {issue.metric} ({issue.period}): {issue.message}")
                else:
                    financial_log.info(f"Validation INFO - {issue.metric} ({issue.period}): {issue.message}")
        except Exception as e:
            try:
                from financial4all.core import log as financial_log
                financial_log.debug(f"Validation failed: {e}")
            except Exception:
                pass
        
        # Store original revenue values for converting Revenue absolute difference to percentage
        # This will be used when displaying Revenue Y/Y change
        if "Revenue" in df_for_analysis.columns:
            profitability_df.attrs = {"original_revenue": df_for_analysis["Revenue"].to_dict()}
        
        # Format date columns in profitability_df to match df_display format
        if not profitability_df.empty:
            formatted_profitability_cols = ["Metric"]
            for col in profitability_df.columns[1:]:
                try:
                    formatted_col = pd.to_datetime(col).strftime('%Y-%m-%d')
                    formatted_profitability_cols.append(formatted_col)
                except (ValueError, TypeError):
                    formatted_profitability_cols.append(str(col))
            profitability_df.columns = formatted_profitability_cols
    except Exception:
        # If profitability calculation fails, continue without it
        profitability_df = pd.DataFrame()
    
    # --- Build the interactive table ---
    # Convert numeric column names to string for display, and format dates correctly.
    df_display = df_standard.copy()
    alternatives = alternatives_json or {}  # FIX: Define alternatives from the JSON data

    # Detect unit scale for the data (use user preference if provided)
    scale_factor, unit_label = _detect_unit_scale(df_display, user_preference=unit_scale_preference)
    
    # Format date columns (unit indicator will go in top-left cell)
    formatted_columns = ["Metric"]
    for col in df_standard.columns[1:]:
        try:
            # Attempt to convert column to datetime and format it
            formatted_columns.append(pd.to_datetime(col).strftime('%Y-%m-%d'))
        except (ValueError, TypeError):
            # If it's not a date-like string, keep it as is
            formatted_columns.append(str(col))
    df_display.columns = formatted_columns

    # Build header with unit indicator in top-left cell
    header_cells = []
    for i, col in enumerate(df_display.columns):
        if i == 0:
            # Top-left cell: "Metric" with unit below if available
            header_content = [
                html.Div("Metric", style={"fontWeight": "bold"}),
            ]
            if unit_label:
                header_content.append(
                    html.Div(
                        f"({unit_label})",
                        style={"fontSize": "0.85em", "color": "#B0B0B0", "marginTop": "2px"},
                    )
                )
            header_cells.append(
                html.Th(
                    header_content,
                    style={
                        "minWidth": "200px",
                        "padding": "6px 8px",
                        "verticalAlign": "middle",
                        "textAlign": "left",
                        "border": "1px solid #444",
                        "backgroundColor": "#1a1a1a",
                        "fontSize": "0.95em",
                    },
                )
            )
        else:
            # Date columns
            header_cells.append(
                html.Th(
                    col,
                    style={
                        "padding": "6px 8px",
                        "textAlign": "right",
                        "verticalAlign": "middle",
                        "border": "1px solid #444",
                        "backgroundColor": "#1a1a1a",
                        "fontSize": "0.95em",
                    },
                )
            )
    
    table_header = [html.Tr(header_cells)]

    table_rows = []
    for idx, row in df_display.iterrows():
        metric_name = row["Metric"]
        metric_def = METRIC_DEFINITIONS.get(metric_name, "")

        # Check if this metric has alternatives
        has_alternatives = metric_name in alternatives and len(alternatives[metric_name]) > 0

        # Determine if this is a final calculated value (bold) or component (non-bold)
        final_calculations = [
            "Gross Profit",
            "Operating Income",
            "Other income (expense), net",
            "Income Before Taxes",
            "Net Income",
        ]
        is_final_calculation = metric_name in final_calculations
        
        # Metric name cell with tooltip
        metric_cell_content = [
            html.Div(
                metric_name,
                style={
                    "fontWeight": "bold" if is_final_calculation else "normal",
                    "fontSize": "0.9em",
                },
                title=metric_def if metric_def else None,  # Tooltip on hover
            ),
        ]
        
        # Add dropdown if alternatives exist
        if has_alternatives:
            metric_cell_content.append(
                html.Div(
                    [
                        dcc.Dropdown(
                            id={"type": "metric-dropdown", "index": metric_name},
                            options=[
                                {"label": "Default", "value": "default"},
                            ]
                            + [
                                {"label": alt["label"], "value": alt["source"]}
                                for alt in alternatives.get(metric_name, [])
                            ],
                            value=selections.get(metric_name, "default"),
                            style={
                                "width": "100%",
                                "fontSize": "0.8em",
                                "backgroundColor": "#333333",
                                "color": "#E0E0E0",
                            },
                        )
                    ],
                    style={"marginTop": "4px"},
                )
            )
        
        cells = [
            html.Td(
                metric_cell_content,
                style={
                    "padding": "6px 8px",
                    "verticalAlign": "middle",
                    "minWidth": "200px",
                    "border": "1px solid #444",
                    "fontSize": "0.9em",
                },
            )
        ]

        # Add data cells with scaled values
        # EPS values should not be scaled (they're already per-share)
        is_eps_metric = "EPS" in metric_name
        
        for col in df_display.columns[1:]:
            value = row[col]
            if pd.isna(value):
                display_value = "—"
            else:
                try:
                    if is_eps_metric:
                        # EPS values are not scaled - display with 2 decimal places
                        eps_value = float(value)
                        if eps_value < 0:
                            display_value = f"({abs(eps_value):.2f})"
                        else:
                            display_value = f"{eps_value:.2f}"
                    else:
                        # Scale the value by the detected scale factor (includes shares)
                        scaled_value = float(value) / scale_factor
                        
                        # Round to whole numbers (no decimals) for all non-EPS values
                        rounded_value = round(scaled_value)
                        
                        # Format: negative values in parentheses, no $ sign (standard financial format)
                        if rounded_value < 0:
                            display_value = f"({abs(rounded_value):,})"
                        else:
                            display_value = f"{rounded_value:,}"
                except (ValueError, TypeError):
                    display_value = str(value)

            cells.append(
                html.Td(
                    display_value,
                    style={
                        "padding": "6px 8px",
                        "textAlign": "right",
                        "verticalAlign": "middle",
                        "border": "1px solid #444",
                        "fontSize": "0.9em",
                        "fontFamily": "monospace",  # Spreadsheet-like monospace font
                        "fontWeight": "bold" if is_final_calculation else "normal",
                    },
                )
            )

        table_rows.append(html.Tr(cells))
        
        # Add blank spacer rows after key section dividers
        spacer_metrics = [
            "Gross Profit",
            "Operating Income",
            "Other income (expense), net",
            "Income Before Taxes",
            "Net Income",
        ]
        
        if metric_name in spacer_metrics:
            # Create a blank row with increased height for visual separation
            spacer_cells = [
                html.Td("", style={"padding": "8px 8px", "border": "1px solid #444", "height": "16px", "backgroundColor": "#2C2C2C"})
                for _ in range(len(df_display.columns))
            ]
            table_rows.append(html.Tr(spacer_cells))
    
    # --- Add calculated metrics section using library ---
    if not profitability_df.empty:
        # Add a blank spacer row before calculated metrics
        spacer_cells = [
            html.Td("", style={"padding": "8px 8px", "border": "1px solid #444", "height": "16px", "backgroundColor": "#2C2C2C"})
            for _ in range(len(df_display.columns))
        ]
        table_rows.append(html.Tr(spacer_cells))
        
        # Helper function to format percentage for display
        def format_percentage_display(value):
            """Format a decimal value (0.50) as a percentage string (50.00%)."""
            if pd.isna(value) or value is None:
                return "—"
            try:
                pct_value = float(value) * 100
                if pct_value < 0:
                    return f"-{abs(pct_value):.2f}%"
                else:
                    return f"{pct_value:.2f}%"
            except (ValueError, TypeError):
                return "—"
        
        # Helper function to format currency for display
        def format_currency_display(value, scale_factor):
            """Format a raw dollar value as scaled currency string with commas."""
            if pd.isna(value) or value is None:
                return "—"
            try:
                scaled_value = float(value) / scale_factor
                rounded_value = round(scaled_value)
                # Format with commas, negative values in parentheses
                if rounded_value < 0:
                    return f"({abs(rounded_value):,})"
                else:
                    return f"{rounded_value:,}"
            except (ValueError, TypeError):
                return "—"
        
        # Identify dollar amount metrics (not percentages)
        dollar_amount_metrics = {
            "Depreciation & Amortization",
            "CapEx",
            "Receivables",
            "Inventory",
            "Payables",
            "Change in WC"
        }
        # Identify "% of Sales" metrics from Capital & Working Capital section
        # These should NOT be conditionally formatted (they're not in trends section)
        capital_wc_percentage_metrics = {
            "D&A % of Sales",
            "CapEx % of Sales",
            "Receivables % of Sales",
            "Inventory % of Sales",
            "Payables % of Sales"
        }
        
        # Metrics that should be bold
        bold_metrics = ["Operating Margin"]
        
        # Metrics that should have a spacer row after them
        # Note: "Revenue" removed - no spacer between Revenue growth and Expenses section
        spacer_after_metrics = ["Operating Margin"]
        
        # Check if we're in the Y/Y change section (for conditional formatting)
        in_yoy_section = False
        
        # Iterate through profitability DataFrame and add rows
        for idx, row in profitability_df.iterrows():
            metric_name = row["Metric"]
            is_bold = metric_name in bold_metrics
            is_header = metric_name == "Expenses as % of Revenue"
            is_yoy_header = metric_name == "**%Change y/y Change (Trends)**"
            is_dollar_amount = metric_name in dollar_amount_metrics
            is_capital_wc_percentage = metric_name in capital_wc_percentage_metrics
            
            # Track when we enter the Y/Y change section
            if is_yoy_header:
                in_yoy_section = True
            
            # Build metric name cell
            metric_cell = html.Td(
                metric_name.replace("**", ""),  # Remove markdown bold markers
                style={
                    "padding": "10px 12px",
                    "verticalAlign": "middle",
                    "minWidth": "220px",
                    "border": "1px solid #444",
                    "fontSize": "1em",
                    "fontWeight": "bold" if (is_bold or is_header or is_yoy_header) else "normal",
                },
            )
            
            cells = [metric_cell]
            
            # Add data cells for each date column
            for col in df_display.columns[1:]:
                value = row.get(col)
                
                # Header rows (like "Expenses as % of Revenue" or Y/Y header) should have blank cells, not dashes
                if is_header or is_yoy_header:
                    display_value = ""
                    cell_style = {
                        "padding": "10px 12px",
                        "textAlign": "right",
                        "verticalAlign": "middle",
                        "border": "1px solid #444",
                        "fontSize": "1em",
                        "fontFamily": "monospace",
                        "fontWeight": "bold" if is_bold else "normal",
                    }
                else:
                    # Format display value based on metric type
                    if is_dollar_amount:
                        # Dollar amounts: format as scaled currency
                        display_value = format_currency_display(value, scale_factor)
                    else:
                        # Percentages: format as percentage
                        display_value = format_percentage_display(value)
                    
                    # Conditional formatting for Y/Y change section
                    cell_style = {
                        "padding": "10px 12px",
                        "textAlign": "right",
                        "verticalAlign": "middle",
                        "border": "1px solid #444",
                        "fontSize": "1em",
                        "fontFamily": "monospace",
                        "fontWeight": "bold" if is_bold else "normal",
                    }
                    
                    # Apply conditional formatting for Y/Y change values ONLY
                    # Exclude dollar amount metrics and Capital & Working Capital % metrics
                    if in_yoy_section and not is_yoy_header and not is_dollar_amount and not is_capital_wc_percentage and value is not None:
                        try:
                            # Parse the percentage value
                            if isinstance(value, (int, float)) and not pd.isna(value):
                                pct_value = float(value)
                                if pct_value > 0:
                                    # Positive change - green background with better contrast
                                    cell_style["backgroundColor"] = "#4CAF50"  # Darker green for better contrast
                                    cell_style["color"] = "#FFFFFF"  # White text for readability
                                elif pct_value < 0:
                                    # Negative change - red background with better contrast
                                    cell_style["backgroundColor"] = "#F44336"  # Darker red for better contrast
                                    cell_style["color"] = "#FFFFFF"  # White text for readability
                                else:
                                    # Zero - no special background, ensure text is visible
                                    cell_style["color"] = "#E0E0E0"
                                # Zero or NaN - no special background (default)
                        except (ValueError, TypeError):
                            pass  # Keep default styling if parsing fails
                    else:
                        # For non-Y/Y section cells, ensure text color is visible
                        cell_style["color"] = "#E0E0E0"
                
                cells.append(
                    html.Td(
                        display_value,
                        style=cell_style,
                    )
                )
            
            table_rows.append(html.Tr(cells))
            
            # Add spacer row after specific metrics
            if metric_name in spacer_after_metrics:
                spacer_cells = [
                    html.Td("", style={"padding": "8px 8px", "border": "1px solid #444", "height": "16px", "backgroundColor": "#2C2C2C"})
                    for _ in range(len(df_display.columns))
                ]
                table_rows.append(html.Tr(spacer_cells))

    table = html.Table(
        [
            html.Thead(table_header),
            html.Tbody(table_rows),
        ],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "backgroundColor": "#2C2C2C",
            "color": "#E0E0E0",
            "border": "1px solid #444",
            "fontSize": "1em",
        },
    )

    return html.Div(
        [
            html.H2(
                "Standardized Income Statement",
                style={
                    "color": "#E0E0E0",
                    "textAlign": "center",
                    "marginBottom": "10px",
                    "fontSize": "1.2em",
                }
            ),
            table,
        ],
        style={"marginTop": "20px"},
    )


# --- Callback to handle Excel export ---
@app.callback(
    Output("download-dataframe-excel", "data"),
    [Input("export-excel-button", "n_clicks")],
    [State("current-df-store", "data"), State("ticker-input", "value")],
    prevent_initial_call=True,
)
def export_to_excel(n_clicks, current_df_json, ticker):
    """
    Exports the current dataframe to Excel format.
    """
    if n_clicks == 0 or not current_df_json:
        return None

    try:
        # Handle both string and dict formats from Dash Store
        if isinstance(current_df_json, str):
            df = pd.read_json(io.StringIO(current_df_json), orient="split")
        elif isinstance(current_df_json, dict):
            df = pd.DataFrame(
                data=current_df_json.get("data", []),
                columns=current_df_json.get("columns", []),
                index=current_df_json.get("index", [])
            )
        else:
            df = pd.DataFrame()
        
        if df.empty:
            return None

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        buffer.seek(0)

        ticker_upper = ticker.upper() if ticker else "data"
        return dcc.send_bytes(
            buffer.getvalue(),
            f"{ticker_upper}_financial_data.xlsx",
        )
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return None


# --- Callback to handle 10-K download ---
@app.callback(
    Output("download-10k-zip", "data"),
    [Input("download-10k-button", "n_clicks")],
    [State("ticker-input", "value")],
    prevent_initial_call=True,
)
def download_all_10ks(n_clicks, ticker):
    """
    Handles the logic for downloading all available 10-K filings for a given
    ticker as a single zip archive. When the user clicks the 'Download All 10-Ks'
    button, this function is triggered.

    It performs the following steps:
    1. Checks if the button was clicked and if a ticker is provided.
    2. Fetches 10-K filings using the Company API and creates a zip archive in memory.
    3. If the zip file is successfully created, it sends the data to the user's
       browser for download using `dcc.send_bytes`.
    4. If no files are found or an error occurs, it does nothing.

    Args:
        n_clicks (int): The number of times the download button has been clicked.
        ticker (str): The ticker symbol from the input field.

    Returns:
        A Dash `send_bytes` object for the browser to download, or None.
    """
    if n_clicks == 0 or not ticker:
        return None

    ticker_upper = ticker.strip().upper()
    try:
        # Fetch filings using Company API
        company = Company(ticker_upper)
        filings = company.get_filings(form="10-K")
        
        if not filings:
            return None
        
        # Create zip file in memory
        client = SECClient()
        zip_buffer = io.BytesIO()
        files_added = []
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_f:
            for filing in filings:
                accession_number = filing.accession_number.replace("-", "")
                report_url = f"https://www.sec.gov/Archives/edgar/data/{filing.cik}/{accession_number}/Financial_Report.xlsx"
                
                try:
                    response = client.get(report_url)
                    if response.status_code == 200:
                        file_name = f"{ticker_upper}_{filing.form}_{filing.filing_date}_Financial_Report.xlsx"
                        zip_f.writestr(file_name, response.content)
                        files_added.append(file_name)
                except Exception:
                    continue
        
        if files_added:
            zip_buffer.seek(0)
            # Send the zip file to the browser for download.
            return dcc.send_bytes(
                zip_buffer.getvalue(),
                f"{ticker_upper}_10K_filings.zip",
            )
        else:
            # If no data is returned, do not trigger a download.
            return None
    except Exception as e:
        # Log the error and prevent the app from crashing.
        print(f"Error generating 10-K zip for {ticker_upper}: {e}")
        return None




# --- NEW: Comprehensive Analysis Generation Callback ---
@app.callback(
    Output("download-analysis-excel", "data"),
    [Input("generate-analysis-btn", "n_clicks")],
    [
        State("ticker-input", "value"),
        State("standard-is-store", "data"),
        State("is-selections-store", "data"),
        State("unit-scale-store", "data"),
        State("alternatives-store", "data"),
    ],
    prevent_initial_call=True,
)
def generate_analysis_report(n_clicks, ticker, standard_is_json, selections, unit_scale_preference, alternatives_json):
    """
    Generate and download comprehensive financial analysis report.
    
    This callback creates an Excel analysis report with the Income Statement
    formatted exactly as shown on the web page, including profitability calculations.
    
    Args:
        n_clicks: Number of times the button was clicked
        ticker: Ticker symbol from input field
        standard_is_json: Income Statement data from standard-is-store
        selections: User selections from is-selections-store
        unit_scale_preference: User's unit scale preference from unit-scale-store
        alternatives_json: Alternative calculations from alternatives-store
        
    Returns:
        Dash send_bytes object for Excel file download, or None
    """
    if n_clicks == 0 or not ticker:
        return None
    
    if not standard_is_json:
        return None
    
    ticker_upper = ticker.strip().upper()
    
    try:
        # Apply user selections to get the correct DataFrame
        df_standard = _apply_user_selections_to_is(
            standard_is_json, alternatives_json or {}, selections or {}
        )
        
        if df_standard is None or df_standard.empty:
            return None
        
        # Detect unit scale for the data
        scale_factor, unit_label = _detect_unit_scale(
            df_standard, user_preference=unit_scale_preference or "millions"
        )
        
        # Format date columns - helper function to format dates consistently
        def format_date_column(col):
            """Format a date column to YYYY-MM-DD format."""
            try:
                return pd.to_datetime(col).strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                return str(col)
        
        formatted_columns = ["Metric"]
        for col in df_standard.columns[1:]:
            formatted_columns.append(format_date_column(col))
        df_display = df_standard.copy()
        df_display.columns = formatted_columns
        
        # Calculate profitability ratios using library
        try:
            # Convert transposed format back to periods-as-index for analyzer
            df_for_analysis = df_standard.set_index("Metric").T
            
            # Fetch balance sheet and cash flow DataFrames from Company API
            bs_df = pd.DataFrame()
            cf_df = pd.DataFrame()
            try:
                company = Company(ticker_upper)
                financials = company.get_financials()
                bs_df = financials["balance_sheet"].to_dataframe() if financials["balance_sheet"] else pd.DataFrame()
                # Pass balance sheet and income statement for CapEx fallback calculation and validation
                is_df_for_capex = df_for_analysis if "Revenue" in df_for_analysis.columns else None
                cf_df = financials["cash_flow"].to_dataframe(bs_df=bs_df, is_df=is_df_for_capex) if financials["cash_flow"] else pd.DataFrame()
            except Exception:
                # If fetching fails, continue with empty DataFrames
                pass
            
            analyzer = ProfitabilityAnalyzer()
            profitability_df = analyzer.calculate_ratios(df_for_analysis, bs_df, cf_df)
            
            # Run validation and log issues
            try:
                from financial4all.analysis.validators import FinancialStatementValidator
                from financial4all.core import log as financial_log
                validator = FinancialStatementValidator()
                validation_issues = validator.validate_all(
                    is_df=df_for_analysis,
                    bs_df=bs_df,
                    cf_df=cf_df
                )
                
                # Log validation issues
                for issue in validation_issues:
                    if issue.severity.value == "error":
                        financial_log.error(f"Validation ERROR - {issue.metric} ({issue.period}): {issue.message}")
                    elif issue.severity.value == "warning":
                        financial_log.warning(f"Validation WARNING - {issue.metric} ({issue.period}): {issue.message}")
                    else:
                        financial_log.info(f"Validation INFO - {issue.metric} ({issue.period}): {issue.message}")
            except Exception as e:
                try:
                    from financial4all.core import log as financial_log
                    financial_log.debug(f"Validation failed: {e}")
                except Exception:
                    pass
            
            # Format date columns in profitability_df to match df_display format exactly
            if not profitability_df.empty:
                formatted_profitability_cols = ["Metric"]
                for col in profitability_df.columns[1:]:
                    formatted_profitability_cols.append(format_date_column(col))
                profitability_df.columns = formatted_profitability_cols
        except Exception:
            # If profitability calculation fails, continue without it
            profitability_df = pd.DataFrame()
        
        # Create Excel buffer and export using ExcelExporter
        buffer = io.BytesIO()
        exporter = ExcelExporter()
        exporter.export_income_statement_analysis(
            df_display,
            profitability_df,
            scale_factor,
            unit_label,
            buffer
        )
        
        buffer.seek(0)
        
        # Send to browser for download
        return dcc.send_bytes(
            buffer.getvalue(),
            f"{ticker_upper}_Income_Statement_Analysis.xlsx"
        )
    except Exception as e:
        # Log error but don't crash the app
        print(f"Error generating analysis report for {ticker_upper}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    app.run(debug=True)
