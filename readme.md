# Portfolio Optimizer App

This application allows users to optimize their investment portfolios using various strategies and visualize the results. It's built with Python, Flask, and utilizes libraries like yfinance, pandas, plotly, and scipy.

**Deployed App:** [https://cionfolyo.onrender.com]

## Features

- **Portfolio Analysis:** Calculate key portfolio statistics like annualized return, volatility, Sharpe ratio, and cumulative return.
- **Data Visualization:** Generate interactive charts for portfolio returns and cumulative returns over time, as well as pie charts for portfolio weights.
- **Optimization Strategies:**
  - **Hierarchical Risk Parity (HRP):** Allocate weights based on risk contribution of each asset, aiming for diversification.
  - **Minimum Conditional Value at Risk (CVaR):** Minimize the potential for significant losses.
  - **Minimum Variance:** Construct a portfolio with the lowest possible variance.
  - **Tangency Portfolio:** Identify the portfolio with the highest Sharpe ratio for a given level of risk.
- **User-Friendly Interface:** Easy input of tickers and weights, along with options to add/remove tickers and redistribute weights equally.

## Tech Stack

- **Backend:** Python, Flask
- **Financial Data:** yfinance
- **Data Manipulation:** pandas
- **Visualization:** plotly
- **Optimization:** scipy

## Installation (for local development)

1. Clone the repository: `git clone https://[repository_url]`
2. Navigate to the project directory: `cd portfolio-optimizer`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `flask run`

## Usage

1. Access the application in your browser.
2. Enter the desired ticker symbols and their corresponding weights.
3. Choose the number of years of historical data to consider.
4. Optionally, click "Redistribute Weights Equally" to assign equal weights to all assets.
5. Click "Optimize" to analyze the portfolio and view the results.
6. Explore the various tabs to view portfolio statistics, returns charts, and pie charts for different optimization strategies.

## Additional Notes

- Ensure you have an active internet connection to fetch financial data.
- The application is for informational purposes only and does not constitute financial advice.

## Contributing

Contributions are welcome! Please feel free to fork the repository and submit pull requests.
