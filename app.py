import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd # Assuming you'll use pandas within process_metrics
from data_loader import process_metrics, CIK_dict
import dash_bootstrap_components as dbc # Import dash_bootstrap_components



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
    dcc.Graph(id='live-update-graph') # Removed initial figure setting to allow update_graph to control it completely
], style={'backgroundColor': '#1E1E1E', 'color': '#E0E0E0', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'}) # Ensure backgroundColor is set here

# Callback for Ticker Input Change

@app.callback(
    [Output('status-output', 'children'),
     Output('available-metrics-selector', 'options'),
     Output('available-metrics-selector', 'value')],
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
        return (html.Div(status_messages), [], [])

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
            return (html.Div(status_messages), [], [])

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
        return (html.Div(status_messages), filtered_options, current_selected_metrics)

    except Exception as e:
        status_messages.append(html.P(f"Error loading data for {ticker_upper}: {e}", style={'color': 'red'}))
        current_df_global = None
        _all_plottable_metrics_global = []
        return (html.Div(status_messages), [], [])


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
    [State('available-metrics-selector', 'value')], # Get current selections
    prevent_initial_call=True
)
def update_metric_options(text_filter, fill_rate_checked, financial_checked, current_selected_metrics):
    if current_df_global is None:
        return [], []
    
    # Call the helper function to get filtered options and updated selection
    filtered_options, new_selection = _apply_all_filters(
        current_df_global, _all_plottable_metrics_global,
        text_filter, fill_rate_checked, financial_checked, current_selected_metrics
    )
    return filtered_options, new_selection


# Callback to update the Plotly graph
@app.callback(
    Output('live-update-graph', 'figure'),
    [Input('available-metrics-selector', 'value')],
    [State('ticker-input', 'value')]
)
def update_graph(selected_metrics, ticker):
    # This is essentially your `analyze` function logic
    if current_df_global is None or current_df_global.empty:
        fig = go.Figure()
        fig.update_layout(title='No data loaded. Please enter a ticker.',
            template='plotly_dark', # Apply dark theme to the generated graph
            paper_bgcolor='#222222', # Set graph paper background to dark
            plot_bgcolor='#222222' # Set plot area background to dark
        )
        return fig

    if not selected_metrics:
        fig = go.Figure()
        fig.update_layout(
            title=f'No Metrics Selected for {ticker.upper()}',
            xaxis_title='Fiscal Year',
            yaxis_title='Value',
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


# Run the app
if __name__ == '__main__':
    app.run(debug=True) # debug=True allows for auto-reloading on code changes