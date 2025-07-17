import pandas as pd
import numpy as np
import requests
import os
import json


EMAIL = 'your_email@example.com'

CIK_dict = pd.read_csv('CIK_dict.csv', converters={'cik_str': str})

def get_cik(ticker):
    """
    Gets the CIK for the specified ticker
    """
    row = CIK_dict[CIK_dict['ticker'].str.upper() == ticker.upper()] # In the dataframe, find the ticker that matches the input ticker
    if not row.empty:
        return row['cik_str'].iloc[0] # Returns the CIK of the ticker (string)
    else:
        raise ValueError(f"Ticker {ticker} not found in CIK_dict")




def get_filing_by_metrics(CIK):
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json'
    headers = {
        'User-Agent': EMAIL
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    return data['facts']['us-gaap']



def extract_metrics(filing_metrics):
    """
    Extracts the specified metrics from the filing
    
    Returns a dictionary of metrics with the following structure:
    {
        'metric_name': [
            {
                'entry_attributes': entry_attributes
            }
        ]
    }
    """
    
    
    
    metric_master = {}
    metrics = list(filing_metrics.keys())
    # print(metrics)
    """
    for m in metrics:
        metric_data = {}
        # print(f'10-k datapoint for{m}:')
        units = filing_metrics[m]['units'] # returns a dictionary of the entries by units i.e {'USD' : '{entry},{entry},{entry},', 'USD/Share' : '{entry},{entry},{entry},'}
        
        for u in units: # for each unit...
            entry_data = []
            metric_entries = list(units[u]) # This is the entries for a given metric
            for entry in metric_entries:
                if entry['form'] == '10-K':
                    entry_data.append(entry)
        metric_master[m] = entry_data
    """
    for metric, attributes in filing_metrics.items():
        for unit, entries in attributes['units'].items():
            for entry in entries:
                if entry['form'] == '10-K':
                    if metric + '_' + unit not in metric_master:
                        metric_master[metric + '_' + unit] = []
                    metric_master[metric + '_' + unit].append(entry)
    print(metric_master)
    
    return metric_master




def format_metrics_efficient(extracted_metrics):
    """
    Converts extracted metrics into a clean, graphable DataFrame.
    This version is both efficient and accurate, correctly handling duplicate years
    by selecting the most recent filing.
    """
    # Filter out entries with 'frame' attribute to avoid aggregated/inconsistent data
    filtered_metrics = {}
    
    for metric_name, metric_entries in extracted_metrics.items():
        # Filter out entries that have a 'frame' attribute
        filtered_entries = [
        entry for entry in metric_entries
        if 'frame' not in entry or ('frame' in entry and 'Q' not in entry['frame'] and 'q' not in entry['frame'])
        ]
        if filtered_entries:  # Only include metrics that have valid entries
            filtered_metrics[metric_name] = filtered_entries
    
    # Process each metric
    all_entries = []
    for metric_name, entries in extracted_metrics.items():
        for entry in entries:
            if 'start' in entry:
            # If 'start' exists, create a new dictionary for the entry with standardized keys.
            # This flattens the data for easier processing or conversion into a DataFrame.
                all_entries.append({
                    'fy': entry['fy'],            # Fiscal year of the entry.
                    'metric': metric_name,        # The name of the financial metric.
                    'val': entry['val'],          # The value of the metric.
                    'filed': entry['filed'],      # The date the metric was filed, useful for finding the latest entry.
                    'end': entry['end'],          # The end date/period of the entry.
                    'start': entry['start']       # The start date/period of the entry.
                })
            else:
                all_entries.append({
                    'fy': entry['fy'],            # Fiscal year of the entry.
                    'metric': metric_name,        # The name of the financial metric.
                    'val': entry['val'],          # The value of the metric.
                    'filed': entry['filed'],      # The date the metric was filed, useful for finding the latest entry.
                    'end': entry['end'],          # The end date/period of the entry.
            })

    if not all_entries:
        return pd.DataFrame()

    # Create a single DataFrame from all data points at once
    df = pd.DataFrame(all_entries)

    # Convert 'filed' to datetime for correct sorting
    df['filed'] = pd.to_datetime(df['filed'])
    df['end'] = pd.to_datetime(df['end'])
    
    if 'start' in df.columns:
        df['start'] = pd.to_datetime(df['start'])

    # Sort by filing date to ensure the latest filing comes first for each metric/year
    df.sort_values(by=['fy', 'filed', 'end', 'start'], ascending=[True, False, True, False], inplace=True)

    # Drop duplicates, keeping the last entry which is the most recent filing
    df.drop_duplicates(subset=['metric', 'fy'], keep='last', inplace=True)


    

    # Pivot the DataFrame to get fiscal years as the index and metrics as columns
    df_pivot = df.pivot(index='fy', columns='metric', values='val')

    # Standardize metric names
    metric_mapping = {
        'Revenues_USD' : 'Revenue_USD',
        
        'GrossProfit_USD' : 'GrossProfit_USD',
        
        'ResearchAndDevelopmentExpense_USD' : 'R&D_USD',
        'ResearchAndDevelopment_USD' : 'R&D_USD',
        
        'SellingGeneralAndAdministrativeExpense_USD' : 'SG&A_USD',
        'GeneralAndAdministrativeExpense_USD' : 'SG&A_USD',
        
        'OperatingIncomeLoss_USD' : 'OperatingIncome_USD',
        
        'InterestExpense_USD' : 'InterestExpense_USD',
        'NoninterestExpense_USD' : 'NonInterestExpense_USD',
        'OtherNoninterestExpense_USD' : 'NonInterestExpense(Other)_USD',
        
        'InterestIncomeExpenseNonoperatingNet_USD' : 'InterestIncome_USD',
        'InterestIncomeExpenseNet_USD' : 'InterestIncome_USD',
        'InterestExpenseDebt_USD' : 'InterestIncome_USD',
        'InvestmentIncomeInterest_USD' : 'InterestIncome_USD',
        'NoninterestIncome_USD' : 'NonInterestIncome_USD',
        
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments_USD' : 'Income_before_taxes_USD',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest_USD' : 'Income_before_taxes_USD',
        
        'IncomeTaxesPaid_USD' : 'IncomeTaxes_USD',
        
        'IncomeTaxExpenseBenefit_USD' : 'IncomeTaxes_USD(Benefit)',
        
        'NetIncomeLoss_USD': 'NetIncome_USD',
        
        'EarningsPerShareBasic_USD/shares': 'EPS_USD',
        
        'EarningsPerShareDiluted_USD/shares': 'EPS_Diluted_USD',
    }
    df_pivot.rename(columns=metric_mapping, inplace=True)

    # If renaming creates duplicate column names (e.g., two 'SG&A' columns),
    # group by the column names and take the first non-null value to consolidate them.
    df_pivot = df_pivot.groupby(by=df_pivot.columns, axis=1).first()

    # Reset index to make 'fy' a regular column
    df_pivot.reset_index(inplace=True)
    df_pivot.rename_axis(None, axis=1, inplace=True) # Clean up the column index name

    """
    
    this allows us to insert additional metadata such as filing, end, or start date etc.
    note: the filing dates are not always accurate for the time period the data is for.
    
    # Merge the pivoted DataFrame with the 'fy_end_df' to add the 'end' date as a new column.
    # The merge is performed on the 'fy' column, ensuring that each fiscal year in the
    # pivoted data gets its corresponding 'end' date.
    
    fy_filed_df = df[['fy', 'filed']].drop_duplicates(subset=['fy'], keep='first')
    
    final_df = pd.merge(df_pivot, fy_filed_df, on='fy', how='left')

    # Reorder the columns to place 'fy' and 'end' at the beginning of the DataFrame,
    # followed by the rest of the metric columns sorted alphabetically.
    cols = final_df.columns.tolist()
    if 'fy' in cols:
        cols.remove('fy')
    if 'filed' in cols:
        cols.remove('filed')
    final_df = final_df[['fy', 'filed'] + sorted(cols)]

    return final_df
    """
    return df_pivot

# just to make it easier to call the functions
def process_metrics(ticker: str):
    return format_metrics_efficient(extract_metrics(get_filing_by_metrics(get_cik(ticker))))
