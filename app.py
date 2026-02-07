import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import json
import dash_bootstrap_components as dbc
import io
import zipfile
from financial4all import Company
from financial4all.analysis import ProfitabilityAnalyzer
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
                    # --- Metric Filtering and Selection ---
                    html.Div(
                        [
                            html.Label("Filter Metrics:", style={"color": "#B0B0B0"}),
                            dcc.Input(
                                id="metric-filter-input",
                                type="text",
                                value="",
                                placeholder="Type to filter metrics...",
                                style={
                                    "width": "100%",
                                    "backgroundColor": "#333333",
                                    "color": "#E0E0E0",
                                    "border": "1px solid #555555",
                                },
                            ),
                            dcc.Checklist(
                                id="fill-rate-checkbox",
                                options=[
                                    {
                                        "label": "Only show metrics with >= 80% data fill",
                                        "value": "80_percent",
                                    }
                                ],
                                value=[],
                                style={"marginTop": "10px", "color": "#B0B0B0"},
                            ),
                            dcc.Checklist(
                                id="only-financial-checkbox",
                                options=[
                                    {
                                        "label": "Show only standardized metrics",
                                        "value": "standardized_only",
                                    }
                                ],
                                value=[
                                    "standardized_only"
                                ],  # Default to showing only standard metrics
                                style={"marginTop": "5px", "color": "#B0B0B0"},
                            ),
                        ],
                        style={"width": "80%", "margin": "auto", "padding": "10px"},
                    ),
                    # --- Metric Selection Dropdown ---
                    html.Div(
                        [
                            html.Label("Select Metrics to Plot:", style={"color": "#B0B0B0"}),
                            dcc.Dropdown(
                                id="available-metrics-selector",
                                options=[],
                                value=[],
                                multi=True,
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#E0E0E0",
                                },
                            ),
                        ],
                        style={"width": "80%", "margin": "auto", "padding": "10px"},
                    ),
                    # --- Regression Option ---
                    html.Div(
                        [
                            dcc.Checklist(
                                id="linear-regression-checkbox",
                                options=[
                                    {"label": "Show Linear Regression", "value": "show_regression"}
                                ],
                                value=[],
                                style={"color": "#B0B0B0"},
                            ),
                        ],
                        style={"width": "80%", "margin": "auto", "padding": "10px"},
                    ),
                    # --- Main Graph ---
                    dcc.Graph(id="live-update-graph"),
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
                    # --- Output Sections ---
                    html.Div(id="standard-is-output", style={"marginTop": "20px"}),
                    html.H2(
                        "Selected Metrics Statistics:",
                        style={
                            "color": "#E0E0E0",
                            "marginTop": "20px",
                            "textAlign": "center",
                        },
                    ),
                    html.Div(id="statistics-output", style={"color": "#E0E0E0"}),
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
        Output("available-metrics-selector", "options"),
        Output("available-metrics-selector", "value"),
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
            return status_message, [], [], None, [], None, None, {}, [], None
        
        # For backward compatibility, create df_merged (all metrics)
        # This combines income statement with other available metrics
        df_merged = standard_is_df.copy()
        
        # Add balance sheet and cash flow metrics if available
        if balance_sheet:
            bs_df = balance_sheet.to_dataframe()
            if not bs_df.empty:
                df_merged = df_merged.join(bs_df, how='outer', rsuffix='_bs')
        
        if cash_flow:
            cf_df = cash_flow.to_dataframe()
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
                html.P(f"Data loaded for {ticker_upper}. Select metrics to plot."),
            ]
        )

        # Correctly filter the initial view to show only standardized metrics by default
        filtered_options, current_selected_metrics = _apply_all_filters(
            df_merged,
            all_plottable_metrics,
            "", # No text filter initially
            ["standardized_only"], # Checkbox is checked by default
            [], # No fill rate filter initially
            [], # No metrics are selected yet
            standard_metrics,
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
            filtered_options,
            current_selected_metrics,
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
        return status_message, [], [], None, [], None, None, {}, [], None


def _apply_all_filters(
    df,
    all_metrics,
    text_filter,
    financial_checked,
    fill_rate_checked,
    current_selection,
    standard_metrics,
):
    """
    A helper function to apply the user's selected filters to the list of available metrics.
    """
    filtered_metrics = []

    for metric_info in all_metrics:
        metric_name = metric_info["value"]
        fill_rate = metric_info["fill_rate"]

        # Text filter
        if text_filter and text_filter.lower() not in metric_name.lower():
            continue

        # Standardized metrics filter
        if "standardized_only" in financial_checked:
            if metric_name not in standard_metrics:
                continue

        # Fill rate filter
        if "80_percent" in fill_rate_checked:
            if fill_rate < 0.8:
                continue

        filtered_metrics.append(metric_info)

    # Determine current selection based on filters
    current_selected = []
    if current_selection:
        # Only include selected metrics that pass the filters
        for metric in current_selection:
            if any(m["value"] == metric for m in filtered_metrics):
                current_selected.append(metric)

    return filtered_metrics, current_selected


# --- Callback to update metric options based on filters ---
@app.callback(
    [
        Output("available-metrics-selector", "options", allow_duplicate=True),
        Output("available-metrics-selector", "value", allow_duplicate=True),
    ],
    [
        Input("metric-filter-input", "value"),
        Input("fill-rate-checkbox", "value"),
        Input("only-financial-checkbox", "value"),
    ],
    [
        State("all-plottable-metrics-store", "data"),
        State("available-metrics-selector", "value"),
        State("standard-metrics-store", "data"),
    ],
    prevent_initial_call=True,
)
def update_metric_filters(text_filter, fill_rate_checked, financial_checked, all_metrics, current_selection, standard_metrics):
    """
    Updates the available metrics dropdown based on user filters.
    """
    if not all_metrics:
        return [], []

    filtered_options, current_selected = _apply_all_filters(
        None,  # df not needed for filtering
        all_metrics,
        text_filter or "",
        financial_checked or [],
        fill_rate_checked or [],
        current_selection or [],
        standard_metrics or [],
    )

    return filtered_options, current_selected


# --- Callback to update graph based on selected metrics ---
@app.callback(
    Output("live-update-graph", "figure"),
    [
        Input("available-metrics-selector", "value"),
        Input("linear-regression-checkbox", "value"),
    ],
    [State("ticker-input", "value"), State("current-df-store", "data")],
)
def update_graph(selected_metrics, regression_checked, ticker, current_df_json):
    """
    Updates the main graph based on the selected metrics and regression option.
    """
    import statsmodels.api as sm

    if not current_df_json:
        fig = go.Figure()
        fig.update_layout(
            title="No data loaded. Please enter a ticker.",
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
        )
        return fig

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
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        # Invalid JSON or empty data
        fig = go.Figure()
        fig.update_layout(
            title="Error loading data. Please try again.",
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
        )
        return fig
    
    if df.empty or "end" not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="No data available to plot.",
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
        )
        return fig

    fig = go.Figure()
    
    # Handle case where selected_metrics might be None or empty
    if not selected_metrics:
        fig.update_layout(
            title="Please select metrics to plot.",
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
        )
        return fig
    
    for metric in selected_metrics:
        if metric and metric in df.columns:
            # Create a temporary dataframe for plotting to handle potential missing values
            plot_df = df[["end", metric]].dropna()
            if not plot_df.empty:
                # Ensure 'end' is a datetime object for proper plotting
                plot_df["end"] = pd.to_datetime(plot_df["end"])
                fig.add_trace(
                    go.Scatter(x=plot_df["end"], y=plot_df[metric], mode="lines+markers", name=metric)
                )

                if "show_regression" in regression_checked:
                    # For regression on time series, it's better to use numeric values for the x-axis.
                    # We convert dates to ordinal numbers for the regression model.
                    plot_df["end_ordinal"] = plot_df["end"].apply(lambda date: date.toordinal())
                    X = sm.add_constant(plot_df["end_ordinal"])
                    model = sm.OLS(plot_df[metric], X).fit()
                    fig.add_trace(
                        go.Scatter(
                            x=plot_df["end"], # Plot against the actual dates
                            y=model.predict(X),
                            mode="lines",
                            name=f"{metric} (Regression)",
                            line=dict(dash="dash"),
                        )
                    )

    # Only update layout if we have traces
    if len(fig.data) == 0:
        fig.update_layout(
            title="No metrics selected or no data available.",
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
        )
    else:
        ticker_display = ticker.upper() if ticker else "Company"
        fig.update_layout(
            title=f"Financial Metrics for {ticker_display}",
            xaxis_title="End Date",
            yaxis_title="Value",
            hovermode="x unified",
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
        )
    return fig


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
            # The data is sorted from most recent to oldest.
            series = ratios_df[ratio_name].dropna()
            if series.empty:
                continue

            # The latest value is the first item in the series.
            try:
                latest_value = float(series.iloc[0])
            except (ValueError, IndexError, TypeError):
                continue

            # For the sparkline, we want to show time moving left-to-right (oldest to newest).
            # So, we reverse the series for plotting.
            try:
                sparkline_series = series.iloc[::-1].values.tolist()
                
                if len(sparkline_series) == 0:
                    continue

                # Create a sparkline figure
                sparkline = go.Figure(
                    go.Scatter(
                        x=list(range(len(sparkline_series))),
                        y=sparkline_series,
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
                    xaxis=dict(visible=False),
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
                cards.append(dbc.Col(dbc.Card(card_content, color="dark", outline=True), width=4))
            except Exception:
                # Skip this ratio if there's an error creating the card
                continue

        if not cards:
            return html.Div("No ratio data available.", style={"color": "#B0B0B0", "textAlign": "center"})
        
        return dbc.Row(cards)
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
def _detect_unit_scale(df: pd.DataFrame):
    """
    Detect appropriate unit scale for financial data.
    
    Analyzes numeric values to determine if they should be displayed
    in billions, millions, thousands, hundreds, or raw values.
    
    Args:
        df: DataFrame with financial data (first column is "Metric", rest are numeric)
        
    Returns:
        Tuple of (scale_factor, unit_label)
        e.g., (1e6, "millions") means divide by 1e6 and show "(millions)"
    """
    # Get all numeric values (skip "Metric" column)
    numeric_values = []
    for col in df.columns[1:]:
        numeric_values.extend(df[col].dropna().abs().tolist())
    
    if not numeric_values:
        return (1.0, "")
    
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
        return (1.0, "")


# --- UPDATED: Callback to generate the interactive income statement ---
@app.callback(
    Output("standard-is-output", "children"),
    [
        Input("standard-is-store", "data"),
        Input("is-selections-store", "data"),
    ],
    [
        State("alternatives-store", "data"),
        State("ticker-input", "value"),
    ],
    prevent_initial_call=True,
)
def display_standard_is(
    standard_is_json, selections, alternatives_json, ticker
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
        analyzer = ProfitabilityAnalyzer()
        profitability_df = analyzer.calculate_ratios(df_for_analysis)
        
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

    # Detect unit scale for the data
    scale_factor, unit_label = _detect_unit_scale(df_display)
    
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
                        style={"fontSize": "0.75em", "color": "#B0B0B0", "marginTop": "2px"},
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
                        "fontSize": "0.85em",
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
                        "fontSize": "0.85em",
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
                        # Scale the value by the detected scale factor
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
        
        # Metrics that should be bold
        bold_metrics = ["Operating Margin"]
        
        # Metrics that should have a spacer row after them
        # Note: "Revenue" removed - no spacer between Revenue growth and Expenses section
        spacer_after_metrics = ["Operating Margin"]
        
        # Iterate through profitability DataFrame and add rows
        for idx, row in profitability_df.iterrows():
            metric_name = row["Metric"]
            is_bold = metric_name in bold_metrics
            is_header = metric_name == "Expenses as % of Revenue"
            
            # Build metric name cell
            metric_cell = html.Td(
                metric_name,
                style={
                    "padding": "6px 8px",
                    "verticalAlign": "middle",
                    "minWidth": "200px",
                    "border": "1px solid #444",
                    "fontSize": "0.9em",
                    "fontWeight": "bold" if (is_bold or is_header) else "normal",
                },
            )
            
            cells = [metric_cell]
            
            # Add data cells for each date column
            for col in df_display.columns[1:]:
                value = row.get(col)
                
                # Header rows (like "Expenses as % of Revenue") should have blank cells, not dashes
                if is_header:
                    display_value = ""
                else:
                    display_value = format_percentage_display(value)
                
                cells.append(
                    html.Td(
                        display_value,
                        style={
                            "padding": "6px 8px",
                            "textAlign": "right",
                            "verticalAlign": "middle",
                            "border": "1px solid #444",
                            "fontSize": "0.9em",
                            "fontFamily": "monospace",
                            "fontWeight": "bold" if is_bold else "normal",
                        },
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
            "fontSize": "0.9em",
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


# --- Callback to update statistics display ---
@app.callback(
    Output("statistics-output", "children"),
    [
        Input("available-metrics-selector", "value"),
        Input("current-df-store", "data"),
    ],
)
def update_statistics(selected_metrics, current_df_json):
    """
    Updates the statistics display based on selected metrics.
    """
    if not selected_metrics or not current_df_json:
        return html.Div("Select metrics to view statistics.", style={"color": "#B0B0B0"})

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
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return html.Div("Error loading data.", style={"color": "#FF6B6B"})

    if df.empty:
        return html.Div("No data available.", style={"color": "#B0B0B0"})

    stats_list = []
    for metric in selected_metrics:
        if metric in df.columns:
            series = df[metric].dropna()
            if not series.empty:
                stats_list.append(
                    html.Div(
                        [
                            html.H4(metric, style={"color": "#E0E0E0"}),
                            html.P(f"Mean: ${series.mean():,.2f}", style={"color": "#B0B0B0"}),
                            html.P(f"Std Dev: ${series.std():,.2f}", style={"color": "#B0B0B0"}),
                            html.P(f"Min: ${series.min():,.2f}", style={"color": "#B0B0B0"}),
                            html.P(f"Max: ${series.max():,.2f}", style={"color": "#B0B0B0"}),
                        ],
                        style={"margin": "10px", "padding": "10px", "border": "1px solid #555555"},
                    )
                )

    if not stats_list:
        return html.Div("No statistics available for selected metrics.", style={"color": "#B0B0B0"})

    return html.Div(stats_list)


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
    [State("ticker-input", "value")],
    prevent_initial_call=True,
)
def generate_analysis_report(n_clicks, ticker):
    """
    Generate and download comprehensive financial analysis report.
    
    This callback creates a comprehensive Excel analysis report including:
    - Multi-year financial statements
    - Financial ratios
    - Trend analysis
    - Common-size statements
    - Summary metrics
    
    Args:
        n_clicks: Number of times the button was clicked
        ticker: Ticker symbol from input field
        
    Returns:
        Dash send_bytes object for Excel file download, or None
    """
    if n_clicks == 0 or not ticker:
        return None
    
    ticker_upper = ticker.strip().upper()
    
    try:
        from financial4all import Company
        from financial4all.analysis import FinancialAnalysisReport
        import io
        
        # Create company instance
        company = Company(ticker_upper)
        
        # Generate report
        report = FinancialAnalysisReport(company)
        
        # Create Excel in memory
        buffer = io.BytesIO()
        report.export_to_excel(buffer)
        buffer.seek(0)
        
        # Send to browser for download
        return dcc.send_bytes(
            buffer.getvalue(),
            f"{ticker_upper}_Financial_Analysis.xlsx"
        )
    except Exception as e:
        # Log error but don't crash the app
        print(f"Error generating analysis report for {ticker_upper}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    app.run(debug=True)
