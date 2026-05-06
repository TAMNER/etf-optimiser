"""Small, testable core helpers for the ETF optimiser notebook.

The Colab notebook remains the user-facing entry point. These functions mirror the
math-heavy parts so they can be run repeatedly in local tests without Google Drive,
Yahoo Finance, or InvestEngine network calls.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252
RISK_FREE_RATE = 0.0


def prepare_log_returns(prices_long):
    """Convert long close prices into a clean log-return matrix.

    Duplicate date/ticker rows can occur when an ETF appears more than once in
    the input universe or Yahoo returns overlapping records. Average duplicate
    closes so the reshape step remains deterministic instead of raising.
    """
    prices_wide = prices_long.pivot_table(
        index="date",
        columns="ticker",
        values="close",
        aggfunc="mean",
    ).sort_index()
    prices_wide = prices_wide.dropna(axis=1, thresh=int(len(prices_wide) * 0.8))
    prices_wide = prices_wide.ffill().dropna()
    log_returns = np.log(prices_wide / prices_wide.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("No returns available after preparing prices.")
    return log_returns


def compute_risk_metrics(returns, names=None, window_name="test", trading_days=TRADING_DAYS, risk_free_rate=RISK_FREE_RATE):
    """Compute annual return, volatility, and Sharpe ratio for each ticker."""
    names = names or {}
    annual_return = returns.mean() * trading_days
    volatility = returns.std() * np.sqrt(trading_days)
    sharpe_ratio = (annual_return - risk_free_rate) / volatility

    return pd.DataFrame({
        "window": window_name,
        "ticker": annual_return.index,
        "name": [names.get(ticker, ticker) for ticker in annual_return.index],
        "annual_return": annual_return.values,
        "volatility": volatility.values,
        "sharpe_ratio": sharpe_ratio.values,
    })


def pca_denoised_covariance(returns, n_components, names=None, trading_days=TRADING_DAYS):
    """Build a PCA-denoised annualised covariance matrix."""
    names = names or {}
    cov = returns.cov().values * trading_days
    tickers = list(returns.columns)
    n_assets = len(tickers)
    n_components = max(1, min(n_components, n_assets))

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    denoised_eigenvalues = eigenvalues.copy()
    if n_components < n_assets:
        denoised_eigenvalues[n_components:] = eigenvalues[n_components:].mean()

    denoised_cov = eigenvectors @ np.diag(denoised_eigenvalues) @ eigenvectors.T
    denoised_cov = (denoised_cov + denoised_cov.T) / 2

    explained = pd.DataFrame({
        "component": np.arange(1, n_assets + 1),
        "explained_variance": eigenvalues,
        "explained_variance_ratio": eigenvalues / eigenvalues.sum(),
        "kept": np.arange(1, n_assets + 1) <= n_components,
    })

    loadings = pd.DataFrame(
        eigenvectors[:, :n_components],
        index=tickers,
        columns=[f"PC{i}" for i in range(1, n_components + 1)],
    ).reset_index().rename(columns={"index": "ticker"})
    loadings["name"] = loadings["ticker"].map(names).fillna(loadings["ticker"])

    return denoised_cov, explained, loadings


def portfolio_stats(weights, annual_returns, cov, risk_free_rate=RISK_FREE_RATE):
    """Return annual return, volatility, and Sharpe ratio for a weight vector."""
    portfolio_return = float(weights @ annual_returns)
    portfolio_volatility = float(np.sqrt(weights @ cov @ weights))
    portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, portfolio_sharpe


def optimise_best_sharpe(
    returns,
    cov,
    names=None,
    max_weight=0.30,
    target_return_steps=50,
    trading_days=TRADING_DAYS,
    risk_free_rate=RISK_FREE_RATE,
    maxiter=300,
    ftol=1e-9,
):
    """Find a best-Sharpe long-only portfolio with one direct SLSQP solve.

    ``target_return_steps`` is kept for backward compatibility with older
    notebook/test calls, but the optimiser no longer runs a slow efficient
    frontier grid.
    """
    _ = target_return_steps
    names = names or {}
    tickers = list(returns.columns)
    n_assets = len(tickers)
    if n_assets * max_weight < 1:
        raise ValueError("max_weight is too low for the number of assets.")

    annual_returns = (returns.mean() * trading_days).values
    bounds = [(0, max_weight)] * n_assets
    initial_weights = np.ones(n_assets) / n_assets
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    def negative_sharpe(weights):
        _, volatility, sharpe = portfolio_stats(weights, annual_returns, cov, risk_free_rate)
        if not np.isfinite(sharpe) or volatility <= 0:
            return 1e6
        return -sharpe

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": ftol, "disp": False},
    )
    if not result.success:
        raise ValueError(f"No feasible portfolio found: {result.message}")

    ret, vol, sharpe = portfolio_stats(result.x, annual_returns, cov, risk_free_rate)
    best = {
        "weights": result.x,
        "annual_return": ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
    }

    weights = pd.DataFrame({
        "ticker": tickers,
        "name": [names.get(ticker, ticker) for ticker in tickers],
        "weight": best["weights"],
    })
    return weights, best
