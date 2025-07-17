# README.md
# Financial4All (F4A)

Financial4All, or F4A, is an open-source platform designed to democratize access to public financial data from the U.S. Securities and Exchange Commission (SEC). This project was born out of a belief that financial education and knowledge should be universally accessible, not gatekept by the financial industry.

## The Problem F4A Solves

Many existing financial data companies charge exorbitant fees or severely restrict access to crucial financial information. This creates a barrier for everyday individuals seeking to make informed investment decisions, often leading them to rely on speculative feelings rather than concrete data.

F4A aims to level the playing field by providing a transparent and accessible tool for financial analysis. Instead of relying on gut feelings, users can now back their investment strategies with verifiable data and insights directly from the source.

## Key Features

- **Direct SEC Data Fetching**: F4A directly retrieves public financial data from the SEC's API when provided with a company ticker symbol.
- **Data Cleaning and Formatting**: The platform cleans and converts raw SEC data into a standardized, analysis-ready format.
- **Granular Data Visualization**: Users can visualize a wide range of financial metrics on a granular basis through an interactive dashboard.

## How It Works

F4A utilizes the SEC's extensive public database, specifically the XBRL (eXtensible Business Reporting Language) filings, to extract key financial metrics. The `data_loader.py` script handles the fetching and preliminary processing, while `app.py` powers the interactive web dashboard built with Dash.

## Setup and Usage

To get started with Financial4All, follow these simple steps:

1.  **Insert Your Email**: The SEC API requires an email address in the User-Agent header for all requests. Open `data_loader.py` and replace the placeholder email with your own:
    ```python
    EMAIL = 'your_email@example.com'  # Replace with your actual email
    ```

2.  **Install Dependencies**: Navigate to the project's root directory in your terminal and install the required Python packages:
    ```bash
pip install -r requirements.txt
    ```
    (Note: A `requirements.txt` file should be created if not already present, containing `dash`, `dash-bootstrap-components`, `plotly`, `pandas`, `numpy`, and `requests`.)

3.  **Run the Application**: Once dependencies are installed, run the main application script:
    ```bash
python app.py
    ```

4.  **Access the Dashboard**: The application will start a local Flask dashboard, typically accessible at `http://127.0.0.1:8050/` in your web browser. You can then enter a ticker symbol and start exploring financial data! 