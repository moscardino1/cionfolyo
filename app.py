from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly, json
from scipy.optimize import minimize
import plotly.express as px
import numpy as np

app = Flask(__name__)

def create_pie_chart(weights, title):
    labels = weights.index
    values = weights.values
    fig = px.pie(values=values, names=labels, title=title)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def cumulative_return(portfolio_returns_weighted):
    return (portfolio_returns_weighted + 1).prod() - 1

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def portfolio_cvar(weights, returns, cov_matrix, alpha=0.95):
    portfolio_return = np.sum(returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    portfolio_returns = np.dot(returns, weights)
    cvar = -np.percentile(portfolio_returns, alpha * 100)
    return cvar

def negative_sharpe_ratio(weights, returns, cov_matrix):
    portfolio_return = np.sum(returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return -sharpe_ratio

def hierarchical_risk_parity(portfolio_returns):
    cov_matrix = portfolio_returns.cov()
    num_assets = len(portfolio_returns.columns)
    x0 = np.ones(num_assets) / num_assets
    bounds = ((0, 1),) * num_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    opt_result = minimize(portfolio_variance, x0, args=(cov_matrix,), bounds=bounds, constraints=constraints)
    return opt_result.x

def minimum_cvar_portfolio(portfolio_returns):
    returns = portfolio_returns.mean()
    cov_matrix = portfolio_returns.cov()
    num_assets = len(returns)
    x0 = np.ones(num_assets) / num_assets
    bounds = ((0, 1),) * num_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    opt_result = minimize(portfolio_cvar, x0, args=(returns, cov_matrix,), bounds=bounds, constraints=constraints)
    return opt_result.x

def minimum_variance_portfolio(portfolio_returns):
    cov_matrix = portfolio_returns.cov()
    num_assets = len(portfolio_returns.columns)
    x0 = np.ones(num_assets) / num_assets
    bounds = ((0, 1),) * num_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    opt_result = minimize(portfolio_variance, x0, args=(cov_matrix,), bounds=bounds, constraints=constraints)
    return opt_result.x

def tangency_portfolio(portfolio_returns):
    returns = portfolio_returns.mean()
    cov_matrix = portfolio_returns.cov()
    num_assets = len(returns)
    x0 = np.ones(num_assets) / num_assets
    bounds = ((0, 1),) * num_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    opt_result = minimize(negative_sharpe_ratio, x0, args=(returns, cov_matrix,), bounds=bounds, constraints=constraints)
    return opt_result.x

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    tickers = []
    weights = []

    for i in range(1, 5):
        ticker = request.form.get(f'ticker{i}')
        weight = request.form.get(f'weight{i}')

        if ticker:
            tickers.append(ticker)
            if weight:
                weights.append(float(weight))
            else:
                weights.append(0.0)

    years = int(request.form.get('years'))

    if not tickers:
        error_message = "Please enter at least one ticker symbol."
        return render_template('results.html', error_message=error_message)

    start_date = pd.Timestamp.now() - pd.DateOffset(years=years)
    end_date = pd.Timestamp.now()

    portfolio_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    portfolio_returns = portfolio_data.pct_change().dropna()
    portfolio_weights = pd.Series(weights, index=tickers)
    portfolio_returns_weighted = (portfolio_returns * portfolio_weights).sum(axis=1)

    portfolio_stats = {
        'Annualized Return': (portfolio_returns_weighted.mean() * 252).round(4),
        'Annualized Volatility': (portfolio_returns_weighted.std() * (252 ** 0.5)).round(4),
        'Sharpe Ratio': (portfolio_returns_weighted.mean() / portfolio_returns_weighted.std() * (252 ** 0.5)).round(4),
    }

    portfolio_returns_dates = portfolio_returns.index.strftime('%Y-%m-%d').tolist()
    portfolio_returns_values = portfolio_returns_weighted.tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_returns_dates, y=portfolio_returns_values, mode='lines+markers', name='Portfolio Returns'))
    cumulative_returns = (portfolio_returns_weighted + 1).cumprod() - 1
    fig.add_trace(go.Scatter(x=portfolio_returns_dates, y=cumulative_returns, mode='lines+markers', name='Cumulative Returns'))
    fig.update_layout(title='Portfolio and Cumulative Returns Over Time', xaxis_title='Date', yaxis_title='Returns', yaxis=dict(range=[min(portfolio_returns_values + cumulative_returns.tolist()), max(portfolio_returns_values + cumulative_returns.tolist())]))

    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    cumulative_return_value = cumulative_return(portfolio_returns_weighted)
    hierarchical_risk_parity_weights = hierarchical_risk_parity(portfolio_returns)
    minimum_cvar_weights = minimum_cvar_portfolio(portfolio_returns)
    minimum_variance_weights = minimum_variance_portfolio(portfolio_returns)
    tangency_weights = tangency_portfolio(portfolio_returns)

    hrp_pie = create_pie_chart(pd.Series(hierarchical_risk_parity_weights, index=tickers), "Hierarchical Risk Parity")
    min_cvar_pie = create_pie_chart(pd.Series(minimum_cvar_weights, index=tickers), "Minimum CVaR")
    min_var_pie = create_pie_chart(pd.Series(minimum_variance_weights, index=tickers), "Minimum Variance")
    tangency_pie = create_pie_chart(pd.Series(tangency_weights, index=tickers), "Tangency Portfolio")

    additional_stats = {
        'Cumulative Return': cumulative_return_value,
        'Hierarchical Risk Parity Portfolio Weights': hierarchical_risk_parity_weights,
        'Minimum CVaR Portfolio Weights': minimum_cvar_weights,
        'Minimum Variance Portfolio Weights': minimum_variance_weights,
        'Tangency Portfolio Weights': tangency_weights
    }

    return render_template('results.html', 
                           portfolio_stats=portfolio_stats,
                           portfolio_returns_dates=portfolio_returns_dates,
                           portfolio_returns_values=portfolio_returns_values,
                           additional_stats=additional_stats,
                            cumulative_returns=cumulative_returns.tolist()  ,
                           hrp_pie=hrp_pie, 
                           min_cvar_pie=min_cvar_pie,
                           min_var_pie=min_var_pie, 
                           tangency_pie=tangency_pie

                           )


if __name__ == '__main__':
    app.run(debug=True)
