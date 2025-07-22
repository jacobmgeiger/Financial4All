import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from data_loader import process_metrics, get_company_info
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dcc.express import send_string
import io
import json

# app.py
# This script creates a comprehensive Dash web application for financial analysis.
# It allows users to input a ticker symbol, fetch financial data from the SEC,
# visualize trends, view standardized statements with interactive formula switching, and export data.

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
                html.Hr(style={"borderColor": "#555555"}),
                dcc.Checklist(
                    id="linear-regression-checkbox",
                    options=[
                        {"label": "Show Linear Regression", "value": "show_regression"}
                    ],
                    value=[],
                    style={"marginTop": "10px", "color": "#B0B0B0"},
                ),
                html.Label(
                    "Select Metrics:",
                    style={"marginTop": "10px", "display": "block", "color": "#B0B0B0"},
                ),
                dcc.Dropdown(
                    id="available-metrics-selector",
                    options=[],
                    multi=True,
                    value=[],
                    placeholder="Select metrics to plot...",
                    style={"backgroundColor": "#333333", "color": "#E0E0E0"},
                ),
            ],
            style={"width": "80%", "margin": "auto", "padding": "10px"},
        ),
        html.Hr(style={"borderColor": "#555555"}),
        # --- Plot Output ---
        dcc.Graph(id="live-update-graph"),
        html.Hr(style={"borderColor": "#555555"}),
        # --- Action Buttons ---
        html.Div(
            [
                dbc.Button(
                    "Export to CSV",
                    id="export-excel-button",
                    className="me-2",
                    n_clicks=0,
                    style={"backgroundColor": "#007BFF", "color": "white"},
                ),
                dbc.Button(
                    "Show Standardized Income Statement",
                    id="show-standard-is-button",
                    className="me-2",
                    n_clicks=0,
                    style={"backgroundColor": "#28A745", "color": "white"},
                ),
            ],
            style={"textAlign": "center", "padding": "10px"},
        ),
        dcc.Download(id="download-dataframe-excel"),
        # --- Output Sections ---
        html.Div(id="standard-is-output", style={"marginTop": "20px"}),
        html.H2(
            "Selected Metrics Statistics:",
            style={"color": "#E0E0E0", "marginTop": "20px", "textAlign": "center"},
        ),
        html.Div(id="statistics-output", style={"color": "#E0E0E0"}),
        # --- Data Stores ---
        # dcc.Store components are used to store data in the user's browser,
        # avoiding the need for global variables and ensuring data persists between callbacks.
        dcc.Store(id="current-df-store"),
        dcc.Store(id="all-plottable-metrics-store"),
        dcc.Store(id="standard-is-store"),
        dcc.Store(id="alternatives-store"),  # NEW: Store for alternative calculations
        dcc.Store(
            id="is-selections-store", data={}
        ),  # NEW: Store for user's dropdown selections
        dcc.Store(id="standard-metrics-store"),  # NEW: Store for the list of standard metrics
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
    This callback provides immediate feedback to the user when they submit a ticker.
    It displays company info and a "Loading..." message, then triggers the main data load.
    """
    if not ticker:
        return html.P("Please enter a valid ticker symbol."), ""

    ticker_upper = ticker.strip().upper()
    info = get_company_info(ticker_upper)

    if not info:
        return html.P(
            f"Ticker '{ticker_upper}' not found.", style={"color": "orange"}
        ), ""

    status_message = html.Div(
        [
            html.P(f"Company: {info['title']} (CIK: {info['cik_str']})"),
            html.P(f"Loading data for {ticker_upper}..."),
        ]
    )
    # Pass the ticker to the hidden div to trigger the next callback
    return status_message, ticker_upper


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
        Output("standard-metrics-store", "data"),  # NEW: Output to the new store
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
        # process_metrics now returns the comprehensive df, the standard IS, and alternatives
        (
            df_all_metrics,
            standard_is_df,
            alternatives,
            standard_metrics,
        ) = process_metrics(ticker_upper)
        if df_all_metrics is None or df_all_metrics.empty:
            status_message = html.P(
                f"No data retrieved for {ticker_upper}.", style={"color": "orange"}
            )
            return status_message, [], [], None, [], None, None, {}, []

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
                "fill_rate": df_all_metrics[col].count() / len(df_all_metrics),
            }
            for col in df_all_metrics.columns
            if col != "fy"
        ]

        # The primary display df for the table is the transposed version of the standard IS
        transposed_df = standard_is_df.T.reset_index().rename(
            columns={"index": "Metric"}
        )

        info = get_company_info(ticker_upper)
        status_message = html.Div(
            [
                html.P(f"Company: {info['title']} (CIK: {info['cik_str']})"),
                html.P(f"Data loaded for {ticker_upper}. Select metrics to plot."),
            ]
        )

        filtered_options, current_selected_metrics = _apply_all_filters(
            df_all_metrics,
            all_plottable_metrics,
            "",
            ["standardized_only"],
            [],
            [],
            standard_metrics,
        )
        return (
            status_message,
            filtered_options,
            current_selected_metrics,
            df_all_metrics.to_json(date_format="iso", orient="split"),
            all_plottable_metrics,
            transposed_df.to_json(date_format="iso", orient="split"),
            alternatives,
            default_selections,
            standard_metrics,
        )
    except Exception as e:
        status_message = html.P(
            f"Error loading data for {ticker_upper}: {e}", style={"color": "red"}
        )
        return status_message, [], [], None, [], None, None, {}, []


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

        # Standardized metrics filter
        if (
            "standardized_only" in financial_checked
            and metric_name not in standard_metrics
        ):
            continue
        if text_filter and text_filter.lower() not in metric_name.lower():
            continue
        if "80_percent" in fill_rate_checked and fill_rate < 0.8:
            continue

        filtered_metrics.append({"label": metric_name, "value": metric_name})

    new_selection = [
        item
        for item in current_selection
        if item in {opt["value"] for opt in filtered_metrics}
    ]
    return filtered_metrics, new_selection


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
        State("available-metrics-selector", "value"),
        State("all-plottable-metrics-store", "data"),
        State("standard-metrics-store", "data"),
    ],
    prevent_initial_call=True,
)
def update_metric_options(
    text_filter,
    fill_rate_checked,
    financial_checked,
    current_selected_metrics,
    all_plottable_metrics,
    standard_metrics,
):
    """
    Updates the dropdown of available metrics based on the user's filter criteria.
    """
    if not all_plottable_metrics:
        return [], []

    # We pass a dummy DataFrame to _apply_all_filters as it's not needed for this filtering logic.
    filtered_options, new_selection = _apply_all_filters(
        None,
        all_plottable_metrics,
        text_filter,
        financial_checked,
        fill_rate_checked,  # Corrected order
        current_selected_metrics,
        standard_metrics,
    )
    return filtered_options, new_selection


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

    df = pd.read_json(current_df_json, orient="split")
    # THE FIX: The dataframe from the store has 'fy' as the index.
    # We must reset it to be a column for plotting.
    if "fy" not in df.columns:
        df = df.reset_index().rename(columns={"index": "fy"})

    fig = go.Figure()
    for metric in selected_metrics:
        if metric in df.columns:
            fig.add_trace(
                go.Scatter(x=df["fy"], y=df[metric], mode="lines+markers", name=metric)
            )
            if "show_regression" in regression_checked:
                df_temp = df[["fy", metric]].dropna()
                if not df_temp.empty:
                    X = sm.add_constant(df_temp["fy"])
                    results = sm.OLS(df_temp[metric], X).fit()
                    fig.add_trace(
                        go.Scatter(
                            x=df_temp["fy"],
                            y=results.predict(X),
                            mode="lines",
                            name=f"{metric} (Regression)",
                            line=dict(dash="dash"),
                        )
                    )

    fig.update_layout(
        title=f"Financial Metrics for {ticker.upper()}",
        xaxis_title="Fiscal Year",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
    )
    return fig


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


# --- UPDATED: Callback to generate the interactive income statement ---
@app.callback(
    Output("standard-is-output", "children"),
    [
        Input("show-standard-is-button", "n_clicks"),
        Input("is-selections-store", "data"),
    ],
    [
        State("standard-is-store", "data"),
        State("alternatives-store", "data"),
        State("ticker-input", "value"),
    ],
    prevent_initial_call=True,
)
def display_standard_is(
    n_clicks, selections, standard_is_json, alternatives_json, ticker
):
    """
    Generates and displays an interactive standardized income statement.
    It now includes dropdowns for metrics with alternative calculation paths.
    """
    if not standard_is_json:
        return None

    df_standard = pd.read_json(standard_is_json, orient="split")
    alternatives = alternatives_json or {}

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
                # The values are stored as a dict of {year_str: value}. Convert keys to int for matching.
                alt_series = pd.Series(
                    {int(k): v for k, v in chosen_alt["values"].items()}
                )
                # Get the integer index of the metric row to update
                row_idx = df_standard.index[df_standard["Metric"] == metric].tolist()
                if not row_idx:
                    continue
                row_idx = row_idx[0]
                # Update the numeric columns based on the alternative series
                for year_int, value in alt_series.items():
                    # THE FIX: The dataframe's columns are integers after pd.read_json.
                    # We must check for the integer year_int, not the string version.
                    if year_int in df_standard.columns:
                        df_standard.loc[row_idx, year_int] = value

    # --- Build the interactive table ---
    # Convert numeric column names to string for display, but keep original for lookup
    df_display = df_standard.copy()
    df_display.columns = ["Metric"] + [str(col) for col in df_standard.columns[1:]]

    table_header = [
        html.Tr(
            [
                html.Th(
                    col,
                    style={
                        "minWidth": "250px",
                        "padding": "12px 15px",
                        "verticalAlign": "middle",
                        "textAlign": "left",
                    }
                    if i == 0
                    else {
                        "padding": "12px 15px",
                        "textAlign": "right",
                        "verticalAlign": "middle",
                    },
                )
                for i, col in enumerate(df_display.columns)
            ]
        )
    ]

    table_body = []
    for _, row in df_display.iterrows():
        metric_name = row["Metric"]
        cells = []
        # First cell is the metric name, potentially with a dropdown
        if metric_name in alternatives:
            options = [{"label": "Primary Value", "value": "default"}] + [
                {"label": alt["label"], "value": alt["source"]}
                for alt in alternatives[metric_name]
            ]
            cells.append(
                html.Td(
                    [
                        html.Div(
                            metric_name,
                            style={"fontWeight": "bold", "marginBottom": "6px"},
                        ),
                        dcc.Dropdown(
                            id={"type": "metric-dropdown", "index": metric_name},
                            options=options,
                            value=selections.get(metric_name, "default"),
                            clearable=False,
                            style={"backgroundColor": "#333", "color": "#EEE"},
                        ),
                    ],
                    style={"padding": "10px 15px", "verticalAlign": "top"},
                )
            )
        else:
            cells.append(
                html.Td(
                    metric_name,
                    style={
                        "padding": "12px 15px",
                        "fontWeight": "bold",
                        "verticalAlign": "middle",
                    },
                )
            )

        # Other cells are the financial values
        for col_name in df_display.columns[1:]:  # Skip the 'Metric' column
            val = row[col_name]
            formatted_val = (
                f"{val:,.0f}"
                if pd.notnull(val) and isinstance(val, (int, float))
                else ""
            )
            cells.append(
                html.Td(
                    formatted_val,
                    style={
                        "textAlign": "right",
                        "padding": "12px 15px",
                        "verticalAlign": "middle",
                        "fontFamily": "monospace",
                    },
                )
            )
        table_body.append(html.Tr(cells))

    table = dbc.Table(
        table_header + table_body,
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        style={"color": "#E0E0E0", "backgroundColor": "#2E2E2E"},
    )

    return html.Div(
        [
            html.H2(
                f"Standardized Income Statement for {ticker.upper()}",
                style={"color": "#E0E0E0"},
            ),
            table,
        ]
    )


@app.callback(
    Output("statistics-output", "children"),
    [Input("available-metrics-selector", "value")],
    [State("current-df-store", "data")],
)
def update_statistics(selected_metrics, current_df_json):
    """
    Displays a table of summary statistics for the currently selected metrics.
    """
    if not current_df_json or not selected_metrics:
        return html.P(
            "Select metrics to view statistics."
            if selected_metrics
            else "No data loaded for statistics."
        )

    df = pd.read_json(current_df_json, orient="split")

    table_header = [html.Th("Statistic")] + [
        html.Th(metric) for metric in selected_metrics
    ]
    rows = []
    stats = {"Minimum": "min", "Maximum": "max", "Average": "mean", "Median": "median"}
    for stat_name, stat_func in stats.items():
        row = [html.Td(stat_name)]
        for metric in selected_metrics:
            series = df.get(metric, pd.Series(dtype=float)).dropna()
            val = getattr(series, stat_func)() if not series.empty else "N/A"
            row.append(html.Td(f"{val:,.2f}" if isinstance(val, (int, float)) else val))
        rows.append(html.Tr(row))

    return dbc.Table(
        [html.Thead(html.Tr(table_header)), html.Tbody(rows)],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        style={"color": "#E0E0E0", "backgroundColor": "#2E2E2E"},
    )


@app.callback(
    Output("download-dataframe-excel", "data"),
    [Input("export-excel-button", "n_clicks")],
    [State("standard-is-store", "data"), State("ticker-input", "value")],
    prevent_initial_call=True,
)
def export_to_excel(n_clicks, current_df_json, ticker):
    """
    Handles the logic for exporting the current financial data to a CSV file.
    This now exports the standardized income statement view.
    """
    if n_clicks == 0 or not current_df_json:
        return None

    df = pd.read_json(current_df_json, orient="split")

    # Ensure columns are correctly typed after JSON serialization
    df.columns = [str(c) if c.isdigit() else c for c in df.columns]

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    file_name = (
        f"{ticker.upper()}_income_statement.csv" if ticker else "income_statement.csv"
    )
    return send_string(buffer.getvalue(), file_name, type="text/csv")


if __name__ == "__main__":
    app.run(debug=True)
