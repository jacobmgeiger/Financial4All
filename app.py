import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd # Assuming you'll use pandas within process_metrics
from data_loader import process_metrics, CIK_dict
import dash_bootstrap_components as dbc # Import dash_bootstrap_components
import plotly.express as px # Import plotly express for easy plotting
from dash.dcc.express import send_string # Import send_string for file downloads

# Excel writer
import io


# app.py
# Global variables to store data and plottable metrics
current_df_global = None
_all_plottable_metrics_global = []

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY]) # Use a dark theme from dbc

# Define the layout of the application
app.layout = html.Div([
    html.H1("Financial 4 All", style={'color': '#E0E0E0'}),

    html.Label("Ticker Symbol:", style={'color': '#B0B0B0'}),
    dcc.Input(
        id='ticker-input',
        type='text',
        value='',
        placeholder='Enter Ticker Symbol (e.g., AAPL)',
        style={'backgroundColor': '#333333', 'color': '#E0E0E0', 'border': '1px solid #555555'}
    ),
    html.Div(id='status-output', style={'color': '#B0B0B0'}), # For messages like "Loading..."

    html.Hr(style={'borderColor': '#555555'}), # Horizontal line for separation

    html.Label("Filter Metrics:", style={'color': '#B0B0B0'}),
    dcc.Input(
        id='metric-filter-input',
        type='text',
        value='',
        placeholder='Type to filter metrics...',
        style={'backgroundColor': '#333333', 'color': '#E0E0E0', 'border': '1px solid #555555'}
    ),
    dcc.Checklist(
        id='fill-rate-checkbox',
        options=[{'label': 'Only show metrics with >= 80% data fill', 'value': '80_percent'}],
        value=[], # Empty list means unchecked
        style={'margin-top': '10px', 'color': '#B0B0B0'}
    ),
    dcc.Checklist(
        id='only-financial-checkbox',
        options=[{'label': 'Only show core financial metrics (e.g., Revenue, Assets, EPS)', 'value': 'core_financial'}],
        value=[], # Empty list means unchecked
        style={'margin-top': '5px', 'color': '#B0B0B0'}
    ),
    html.Hr(style={'borderColor': '#555555'}), # Horizontal line for separation
    dcc.Checklist(
        id='linear-regression-checkbox',
        options=[{'label': 'Show Linear Regression', 'value': 'show_regression'}],
        value=[],  # Empty list means unchecked
        style={'margin-top': '10px', 'color': '#B0B0B0'}
    ),

    html.Label("Select Metrics:", style={'color': '#B0B0B0'}),
    dcc.Dropdown( # Using Dropdown with multi=True as a SelectMultiple replacement
        id='available-metrics-selector',
        options=[], # Populated by callbacks
        multi=True,
        value=[],
        placeholder='Select metrics to plot...',
        style={'backgroundColor': '#333333', 'color': '#E0E0E0', 'border': '1px solid #555555'} # Keep border, let theme handle background and color
    ),

    html.Hr(style={'borderColor': '#555555'}), # Horizontal line for separation

    html.H2("Plot Output:", style={'color': '#E0E0E0'}),
    dcc.Graph(id='live-update-graph'), # Removed initial figure setting to allow update_graph to control it completely

    html.Hr(style={'borderColor': '#555555'}), # Horizontal line for separation
    dbc.Button("Export to Excel", id="export-excel-button", className="me-2", style={'backgroundColor': '#007BFF', 'color': 'white'}), # Add a button for exporting data
    dcc.Download(id="download-dataframe-excel"), # Component to trigger downloads

    html.H2("Selected Metrics Statistics:", style={'color': '#E0E0E0'}),
    html.Div(id='statistics-output', style={'color': '#E0E0E0'}),
    
    # Hidden Divs to store data globally instead of python global variables
    dcc.Store(id='current-df-store'),
    dcc.Store(id='all-plottable-metrics-store'),

], style={'backgroundColor': '#1E1E1E', 'color': '#E0E0E0', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'}) # Ensure backgroundColor is set here

# Callback for Ticker Input Change

@app.callback(
    [Output('status-output', 'children'),
     Output('available-metrics-selector', 'options'),
     Output('available-metrics-selector', 'value'),
     Output('current-df-store', 'data'), # Store current_df_global
     Output('all-plottable-metrics-store', 'data')], # Store _all_plottable_metrics_global
    [Input('ticker-input', 'n_submit'), # Trigger when Enter is pressed
     Input('ticker-input', 'n_blur')],   # Trigger when input loses focus
    [State('ticker-input', 'value')],    # Get the value when triggered
    prevent_initial_call=True # Prevent callback from running on initial load
)
def on_ticker_change(n_submit, n_blur, ticker):
    global current_df_global, _all_plottable_metrics_global

    # Initialize messages list for status output
    status_messages = []

    if not ticker:
        status_messages.append(html.P("Please enter a valid ticker symbol."))
        return (html.Div(status_messages), [], [], None, []) # Return None and empty list for stores

    ticker_upper = ticker.strip().upper()

    # Look up company information
    company_title = "N/A"
    company_cik = "N/A"
    if ticker_upper in CIK_dict:
        company_info = CIK_dict[ticker_upper]
        company_title = company_info.get('title', 'N/A')
        company_cik = company_info.get('cik_str', 'N/A')

    # Add initial loading message and company info to the status output
    status_messages.append(html.P(f"Loading data for {ticker_upper}..."))
    if company_title != 'N/A': # Only display if we found valid company info
        status_messages.append(html.P(f"Company: {company_title} (CIK: {company_cik})", style={'font-weight': 'bold'}))
    
    try:
        current_df_global = process_metrics(ticker_upper)

        if current_df_global is None or current_df_global.empty:
            status_messages.append(html.P(f"No data retrieved for {ticker_upper}.", style={'color': 'orange'}))
            _all_plottable_metrics_global = []
            return (html.Div(status_messages), [], [], None, []) # Return None and empty list for stores

        _all_plottable_metrics_global = []
        for col in current_df_global.columns.tolist():
            if col == 'fy':
                continue
            fill_rate = current_df_global[col].count() / len(current_df_global)
            _all_plottable_metrics_global.append({'label': col, 'value': col, 'fill_rate': fill_rate})

        status_messages.append(html.P(f"Data loaded for {ticker_upper}. Select metrics to plot. No metrics are selected by default."))
        
        filtered_options, current_selected_metrics = _apply_all_filters(
            current_df_global, _all_plottable_metrics_global,
            '', [], [], []
        )
        return (html.Div(status_messages), filtered_options, current_selected_metrics, current_df_global.to_json(date_format='iso', orient='split'), _all_plottable_metrics_global)

    except Exception as e:
        status_messages.append(html.P(f"Error loading data for {ticker_upper}: {e}", style={'color': 'red'}))
        current_df_global = None
        _all_plottable_metrics_global = []
        return (html.Div(status_messages), [], [], None, []) # Return None and empty list for stores


# Helper function to apply all filters (similar to your _apply_all_filters)
def _apply_all_filters(df, all_metrics, text_filter, fill_rate_checked, financial_checked, current_selection):
        filtered_metrics = []
        core_financial_keywords = ['revenue', 'income', 'profit', 'asset', 'liability', 'equity', 'cash flow', 'eps', 'debt']

        for metric_info in all_metrics:
            metric_name = metric_info['value'] # 'value' is the actual column name
            fill_rate = metric_info['fill_rate']

            # Apply text filter
            if text_filter and text_filter.lower() not in metric_name.lower():
                continue

            # Apply 80% fill rate filter
            if '80_percent' in fill_rate_checked and fill_rate < 0.8:
                continue

            # Apply "only core financial metrics" filter
            if 'core_financial' in financial_checked:
                is_financial = any(keyword in metric_name.lower() for keyword in core_financial_keywords)
                if not is_financial:
                    continue

            filtered_metrics.append({'label': metric_name, 'value': metric_name})

        # Preserve selected items that are still present in the filtered list.
        # This line needs correction:
        # old: new_selection = [item for item in current_selection if {'label': item, 'value': item} in filtered_metrics]
        # new:
        new_selection = [item for item in current_selection if item in [opt['value'] for opt in filtered_metrics]]

        return filtered_metrics, new_selection


# Callback to update metric options based on filter changes
@app.callback(
    [Output('available-metrics-selector', 'options', allow_duplicate=True),
     Output('available-metrics-selector', 'value', allow_duplicate=True)],
    [Input('metric-filter-input', 'value'),
     Input('fill-rate-checkbox', 'value'),
     Input('only-financial-checkbox', 'value')],
    [State('available-metrics-selector', 'value'), # Get current selections
     State('current-df-store', 'data'), # Get current_df_global from store
     State('all-plottable-metrics-store', 'data')], # Get _all_plottable_metrics_global from store
    prevent_initial_call=True
)
def update_metric_options(text_filter, fill_rate_checked, financial_checked, current_selected_metrics, current_df_json, all_plottable_metrics_json):
    # Retrieve data from stores
    if current_df_json is None or all_plottable_metrics_json is None:
        return [], []
    
    current_df_global = pd.read_json(current_df_json, orient='split')
    _all_plottable_metrics_global = all_plottable_metrics_json
    
    # Call the helper function to get filtered options and updated selection
    filtered_options, new_selection = _apply_all_filters(
        current_df_global, _all_plottable_metrics_global,
        text_filter, fill_rate_checked, financial_checked, current_selected_metrics
    )
    return filtered_options, new_selection


# Callback to update the Plotly graph
@app.callback(
    Output('live-update-graph', 'figure'),
    [Input('available-metrics-selector', 'value'),
     Input('linear-regression-checkbox', 'value')], # New input for regression checkbox
    [State('ticker-input', 'value'),
     State('current-df-store', 'data')] # Get current_df_global from store
)
def update_graph(selected_metrics, regression_checked, ticker, current_df_json):
    # Import statsmodels inside the callback to avoid global import issues if not used
    import statsmodels.api as sm

    if current_df_json is None:
        fig = go.Figure()
        fig.update_layout(title='No data loaded. Please enter a ticker.',
            template='plotly_dark', # Apply dark theme to the generated graph
            paper_bgcolor='#222222', # Set graph paper background to dark
            plot_bgcolor='#222222' # Set plot area background to dark
        )
        return fig

    current_df_global = pd.read_json(current_df_json, orient='split')

    if current_df_global is None or current_df_global.empty:
        fig = go.Figure()
        fig.update_layout(title='No data loaded. Please enter a ticker.',
            template='plotly_dark', # Apply dark theme to the generated graph
            paper_bgcolor='#222222', # Set graph paper background to dark
            plot_bgcolor='#222222' # Set plot area background to dark
        )
        return fig

    fig = go.Figure()
    for metric in selected_metrics:
        if metric in current_df_global.columns:
            fig.add_trace(go.Scatter(
                x=current_df_global['fy'],
                y=current_df_global[metric],
                mode='lines+markers',
                name=metric,
                hovertemplate=
                    '<b>Fiscal Year</b>: %{x}<br>' +
                    f'<b>{metric}</b>: %{{y}}<extra></extra>'
            ))
            # Add linear regression line if checked
            if 'show_regression' in regression_checked:
                # Prepare data for regression: x is fiscal year, y is metric value
                # Drop NaN values for regression calculation
                df_temp = current_df_global[['fy', metric]].dropna()
                if not df_temp.empty:
                    X = sm.add_constant(df_temp['fy']) # Add a constant for the intercept
                    model = sm.OLS(df_temp[metric], X)
                    results = model.fit()
                    
                    fig.add_trace(go.Scatter(
                        x=df_temp['fy'],
                        y=results.predict(X),
                        mode='lines',
                        name=f'{metric} (Regression)',
                        line=dict(dash='dash'),
                        opacity=0.7,
                        hovertemplate=
                            '<b>Fiscal Year</b>: %{x}<br>' +
                            f'<b>{metric} Regression</b>: %{{y}}<extra></extra>'
                    ))

    fig.update_layout(
        title=f'Financial Metrics for {ticker.upper()}',
        xaxis_title='Fiscal Year',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_dark', # Apply dark theme to the generated graph
        paper_bgcolor='#222222', # Set graph paper background to dark
        plot_bgcolor='#222222' # Set plot area background to dark
    )
    return fig


# New callback to update statistics
@app.callback(
    Output('statistics-output', 'children'),
    [Input('available-metrics-selector', 'value')],
    [State('current-df-store', 'data')]
)
def update_statistics(selected_metrics, current_df_json):
    if current_df_json is None:
        return html.P("No data loaded for statistics.")
    
    current_df_global = pd.read_json(current_df_json, orient='split')

    if current_df_global is None or current_df_global.empty or not selected_metrics:
        return html.P("Select metrics to view statistics.")

    # Initialize data for the single table
    # Header row will be 'Statistic' followed by selected_metrics
    table_header = [html.Th("Statistic")] + [html.Th(metric) for metric in selected_metrics]

    # Prepare rows for min, max, average, median
    min_row = [html.Td("Minimum")]
    max_row = [html.Td("Maximum")]
    avg_row = [html.Td("Average")]
    median_row = [html.Td("Median")]

    for metric in selected_metrics:
        if metric in current_df_global.columns:
            series = current_df_global[metric].dropna()
            if not series.empty:
                # Calculate statistics
                min_val = series.min()
                max_val = series.max()
                avg_val = series.mean()
                median_val = series.median()
                
                # Append formatted values to rows
                min_row.append(html.Td(f"{min_val:,.2f}"))
                max_row.append(html.Td(f"{max_val:,.2f}"))
                avg_row.append(html.Td(f"{avg_val:,.2f}"))
                median_row.append(html.Td(f"{median_val:,.2f}"))
            else:
                # Append "N/A" if no data
                min_row.append(html.Td("N/A"))
                max_row.append(html.Td("N/A"))
                avg_row.append(html.Td("N/A"))
                median_row.append(html.Td("N/A"))
        else:
            # Append "N/A" if metric column not found
            min_row.append(html.Td("N/A"))
            max_row.append(html.Td("N/A"))
            avg_row.append(html.Td("N/A"))
            median_row.append(html.Td("N/A"))

    table_body = [
        html.Tr(min_row),
        html.Tr(max_row),
        html.Tr(avg_row),
        html.Tr(median_row),
    ]

    return dbc.Table(
        [html.Thead(html.Tr(table_header)), html.Tbody(table_body)],
        bordered=True, hover=True, responsive=True, striped=True,
        style={'color': '#E0E0E0', 'backgroundColor': '#2E2E2E'}
    )


# Callback to handle data export to Excel
@app.callback(
    Output("download-dataframe-excel", "data"),
    [Input("export-excel-button", "n_clicks")],
    [State('current-df-store', 'data'),
     State('ticker-input', 'value')], # Get ticker to name the file
    prevent_initial_call=True,
)
def export_to_excel(n_clicks, current_df_json, ticker):
    # app.py
    # This callback exports the currently loaded financial data to an Excel file.
    # The data is transposed so that metrics appear on the Y-axis and years on the X-axis.
    
    if n_clicks is None or current_df_json is None:
        return None

    df = pd.read_json(current_df_json, orient='split')

    # Ensure 'fy' (Fiscal Year) is the index before transposing to have years as columns
    # And select only relevant columns (metrics and fy)
    if 'fy' in df.columns:
        df_transposed = df.set_index('fy').T # Set 'fy' as index then transpose
    else:
        # If 'fy' is not present, transpose the entire dataframe and handle column names as needed
        df_transposed = df.T
    
    # Reset index to make metric names a regular column if they were the index after transpose
    df_transposed = df_transposed.reset_index()
    df_transposed = df_transposed.rename(columns={'index': 'Metric'}) # Rename the new index column to 'Metric'

    # Use BytesIO to create an in-memory Excel file
    buffer = io.StringIO() # Use StringIO for CSV
    df_transposed.to_csv(buffer, index=False) # Write to CSV
    buffer.seek(0)

    file_name = f"{ticker.upper()}_financial_data.csv" if ticker else "financial_data.csv"

    return send_string(buffer.getvalue(), file_name, type='text/csv') # Use send_string for CSV



# Run the app
if __name__ == '__main__':
    app.run(debug=True) # debug=True allows for auto-reloading on code changes