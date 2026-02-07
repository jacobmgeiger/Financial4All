# Financial4All (F4A)

Financial4All, or F4A, is an open-source platform designed to democratize access to public financial data from the U.S. Securities and Exchange Commission (SEC). This project was born out of a belief that financial education and knowledge should be universally accessible, not gatekept by the financial industry.

## The Problem F4A Solves

Many existing financial data companies charge exorbitant fees or severely restrict access to crucial financial information. This creates a barrier for everyday individuals seeking to make informed investment decisions, often leading them to rely on speculative feelings rather than concrete data.

F4A aims to level the playing field by providing a transparent and accessible tool for financial analysis. Instead of relying on gut feelings, users can now back their investment strategies with verifiable data and insights directly from the source.

## Key Features

- **Direct SEC Data Fetching**: F4A directly retrieves public financial data from the SEC's API when provided with a company ticker symbol
- **Robust XBRL Parsing**: Advanced parsing of XBRL (eXtensible Business Reporting Language) filings with period-aware concept resolution and synonym detection
- **Comprehensive Standardization**: Unified synonym management system with 40+ pre-built financial concept groups, enabling consistent cross-company analysis
- **Standardized Financial Statements**: Automatically constructs standardized income statements, balance sheets, and cash flow statements
- **Profitability Analysis**: Calculates Y/Y growth, margins, expense ratios, and tax rates
- **Interactive Dashboard**: Web-based dashboard built with Dash for visualizing financial data and trends
- **Excel Export**: Export comprehensive financial analysis reports to Excel with charts and visualizations
- **Multi-year Comparisons**: Analyze financial trends across multiple reporting periods

## Architecture

Financial4All is built as a modular Python package with the following structure:

```
financial4all/
├── sec/              # SEC API client and company lookup
│   ├── client.py     # SEC EDGAR API client
│   ├── company.py    # Company class for ticker/CIK management
│   └── filings.py    # Filing retrieval and management
├── xbrl/             # XBRL parsing and processing
│   ├── facts.py      # FactSet for XBRL fact collection
│   ├── periods.py    # Period handling and normalization
│   ├── parser.py     # XBRL document parser
│   ├── standardization.py  # Unified synonym management system
│   ├── entity_info.py     # Entity information extraction
│   ├── presentation.py    # Presentation linkbase parsing
│   ├── dimensions.py      # Dimensional structure parsing
│   └── calculations.py    # Calculation formula engine
├── financials/       # Financial statement classes
│   ├── income_statement.py  # Income statement extraction and standardization
│   ├── balance_sheet.py     # Balance sheet extraction
│   ├── cash_flow.py         # Cash flow statement extraction
│   └── ratios.py            # Financial ratio calculations
└── analysis/        # Analysis and reporting
    ├── profitability_analyzer.py  # Profitability ratio calculations
    ├── trend_analyzer.py          # Trend analysis
    ├── common_size.py              # Common-size statement generation
    ├── report_generator.py        # Comprehensive report generation
    └── excel_exporter.py           # Excel export functionality
```

### Core Classes

- **`Company`**: Main entry point for accessing company financial data
  - `get_financials()`: Retrieve income statement, balance sheet, and cash flow
  - `get_filings(form)`: Get list of SEC filings for a company
  
- **`IncomeStatement`**: Extracts and standardizes income statement data
  - `to_dataframe()`: Convert to pandas DataFrame
  
- **`BalanceSheet`**: Extracts balance sheet data
  
- **`CashFlowStatement`**: Extracts cash flow statement data
  
- **`FinancialRatios`**: Calculates financial ratios from statements

- **`ProfitabilityAnalyzer`**: Calculates profitability metrics (margins, growth rates, etc.)

## How It Works

F4A utilizes the SEC's extensive public database, specifically the XBRL (eXtensible Business Reporting Language) filings, to extract key financial metrics. The platform uses a robust, multi-tier approach to XBRL parsing:

1. **Fact Discovery**: Collects all relevant XBRL facts from SEC filings
2. **Period-Aware Resolution**: Resolves the best fact for each reporting period using multi-tier filtering
3. **Concept Matching**: Matches XBRL concepts to standardized financial metrics using the comprehensive SynonymGroups system
4. **Synonym Detection**: Discovers alternative concept names when primary concepts don't yield data, leveraging pre-built synonym groups
5. **Calculation**: Applies financial formulas to derive missing values

### Standardization System

F4A includes a comprehensive standardization infrastructure inspired by EdgarTools, featuring:

- **SynonymGroups**: Unified synonym management with 40+ pre-built groups for common financial concepts
- **Concept Identification**: Reverse lookup to identify which standardized concept an XBRL tag represents
- **Category Organization**: Concepts organized by financial statement type (income_statement, balance_sheet, cash_flow, metrics)
- **User Extensibility**: Register custom synonym groups, export/import configurations
- **Multi-group Membership**: Support for tags that belong to multiple concepts in different contexts

Example:
```python
from financial4all.xbrl.standardization import get_synonym_groups

synonyms = get_synonym_groups()

# Get all synonyms for revenue
revenue_tags = synonyms.get_synonyms('revenue')
# ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', ...]

# Identify what concept a tag represents
info = synonyms.identify_concept('NetIncomeLoss')
print(info.name)  # 'net_income'
print(info.description)  # 'Net income/loss'
```

The `app.py` script powers the interactive web dashboard built with Dash, providing a user-friendly interface for exploring financial data.

## Setup and Usage

### Installation

1. **Install Dependencies**: Navigate to the project's root directory and install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure SEC API Identity**: The SEC API requires an email address in the User-Agent header. Set your identity:
   ```python
   from financial4all import set_identity
   set_identity("your_email@example.com")
   ```

### Basic Usage

#### Using the Python API

```python
from financial4all import Company, set_identity

# Set your email for SEC API requests
set_identity("your_email@example.com")

# Create a company instance
company = Company("AAPL")

# Get financial statements
financials = company.get_financials()
income_statement = financials["income_statement"]
balance_sheet = financials["balance_sheet"]
cash_flow = financials["cash_flow"]

# Convert to DataFrame
is_df = income_statement.to_dataframe()
print(is_df)
```

#### Using the Web Dashboard

1. **Run the Application**:
   ```bash
   python app.py
   ```

2. **Access the Dashboard**: Open your web browser and navigate to `http://127.0.0.1:8050/`

3. **Enter a Ticker**: Type a company ticker symbol (e.g., "AAPL", "NVDA", "MSFT") and press Enter

4. **Explore Data**: 
   - View standardized financial statements
   - Visualize trends with interactive charts
   - Export data to Excel
   - Download 10-K filings

### Advanced Usage

#### Generating Analysis Reports

```python
from financial4all import Company
from financial4all.analysis import FinancialAnalysisReport
import io

company = Company("AAPL")
report = FinancialAnalysisReport(company)

# Export to Excel
buffer = io.BytesIO()
report.export_to_excel(buffer)
buffer.seek(0)

# Save to file
with open("AAPL_Analysis.xlsx", "wb") as f:
    f.write(buffer.getvalue())
```

#### Calculating Profitability Ratios

```python
from financial4all.analysis import ProfitabilityAnalyzer
import pandas as pd

# Assuming you have an income statement DataFrame
# with periods as index and metrics as columns
is_df = income_statement.to_dataframe()
is_df_transposed = is_df.set_index("Metric").T if "Metric" in is_df.columns else is_df.T

analyzer = ProfitabilityAnalyzer()
ratios_df = analyzer.calculate_ratios(is_df_transposed)
print(ratios_df)
```

## Project Structure

```
project_3_financial4all/
├── app.py                    # Main Dash web application
├── data_loader.py            # DEPRECATED: Backward compatibility layer
├── financial4all/            # Main package directory
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── core.py               # Core utilities and logging
│   ├── exceptions.py         # Custom exceptions
│   ├── sec/                  # SEC API integration
│   ├── xbrl/                 # XBRL parsing
│   ├── financials/          # Financial statement classes
│   ├── analysis/            # Analysis and reporting
│   └── utils/                # Utility functions
├── notebooks/               # Jupyter notebooks for development/testing
├── xbrl_prep/                # XBRL taxonomy files and formula generation
│   ├── xbrl_main.py          # Script to generate calculation formulas
│   ├── income_statement_formulas.json  # Generated calculation formulas
│   └── us-gaap-2025/         # US-GAAP XBRL taxonomy files
├── CIK_dict.csv              # Company ticker to CIK mapping
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

### XBRL Preparation Directory

The `xbrl_prep/` directory contains:
- **XBRL Taxonomy Files**: Complete US-GAAP 2025 taxonomy files
- **Formula Generation Script**: `xbrl_main.py` processes calculation linkbases to extract financial formulas
- **Generated Formulas**: `income_statement_formulas.json` contains calculation relationships used by the calculation engine

To regenerate formulas (if taxonomy is updated):
```bash
python xbrl_prep/xbrl_main.py > xbrl_prep/income_statement_formulas.json
```

## Dependencies

Key dependencies include:
- **dash**: Web framework for the interactive dashboard
- **pandas**: Data manipulation and analysis
- **requests**: HTTP client for SEC API
- **lxml**: XML parsing for XBRL documents
- **openpyxl**: Excel file generation
- **plotly**: Interactive visualizations

See `requirements.txt` for the complete list.

## Notes

- **`data_loader.py`**: This file is deprecated and maintained only for backward compatibility. New code should use the `financial4all` package directly. It will be removed in a future version.

- **Notebooks**: The `notebooks/` directory contains development and testing notebooks. These are for reference and may contain experimental code.

- **CIK Dictionary**: The `CIK_dict.csv` file maps ticker symbols to Central Index Keys (CIKs) required by the SEC API. This file is required for the application to function.

## Contributing

Contributions are welcome! Please ensure that:
- Code follows the existing style and structure
- New features include appropriate documentation
- Tests are added for new functionality
- The README is updated if needed

## License

See LICENSE file for details.
