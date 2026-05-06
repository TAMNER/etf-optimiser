import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from etf_optimiser_core import (
    compute_risk_metrics,
    optimise_best_sharpe,
    pca_denoised_covariance,
    prepare_log_returns,
)


def synthetic_prices_long(days=320, tickers=5):
    dates = pd.bdate_range("2020-01-01", periods=days)
    rows = []
    for idx in range(tickers):
        ticker = f"ETF{idx + 1}"
        drift = 0.00015 + idx * 0.00003
        wiggle = np.sin(np.arange(days) / (8 + idx)) * 0.001
        daily_returns = drift + wiggle
        prices = 100 * np.exp(np.cumsum(daily_returns))
        rows.extend(
            {"date": date.date(), "ticker": ticker, "name": f"ETF {idx + 1}", "close": price}
            for date, price in zip(dates, prices)
        )
    return pd.DataFrame(rows)


def test_prepare_log_returns_from_long_prices():
    prices = synthetic_prices_long()
    returns = prepare_log_returns(prices)

    assert returns.shape == (319, 5)
    assert list(returns.columns) == ["ETF1", "ETF2", "ETF3", "ETF4", "ETF5"]
    assert np.isfinite(returns.to_numpy()).all()


def test_prepare_log_returns_aggregates_duplicate_date_ticker_prices():
    prices = synthetic_prices_long(days=20, tickers=2)
    duplicate = prices.iloc[[0]].copy()
    duplicate["close"] = duplicate["close"] * 1.02
    prices = pd.concat([prices, duplicate], ignore_index=True)

    returns = prepare_log_returns(prices)

    assert returns.shape == (19, 2)
    assert np.isfinite(returns.to_numpy()).all()


def test_risk_metrics_have_expected_columns_and_values():
    returns = prepare_log_returns(synthetic_prices_long())
    metrics = compute_risk_metrics(returns, names={"ETF1": "ETF 1"}, window_name="full_5y")

    assert list(metrics.columns) == [
        "window",
        "ticker",
        "name",
        "annual_return",
        "volatility",
        "sharpe_ratio",
    ]
    assert len(metrics) == 5
    assert metrics["window"].eq("full_5y").all()
    assert np.isfinite(metrics[["annual_return", "volatility", "sharpe_ratio"]].to_numpy()).all()


def test_pca_denoised_covariance_is_symmetric_and_reports_outputs():
    returns = prepare_log_returns(synthetic_prices_long())
    cov, explained, loadings = pca_denoised_covariance(returns, n_components=3)

    assert cov.shape == (5, 5)
    assert np.allclose(cov, cov.T)
    assert len(explained) == 5
    assert explained["kept"].sum() == 3
    assert np.isclose(explained["explained_variance_ratio"].sum(), 1.0)
    assert list(loadings.columns) == ["ticker", "PC1", "PC2", "PC3", "name"]


def test_optimiser_respects_long_only_sum_and_max_weight_constraints():
    returns = prepare_log_returns(synthetic_prices_long())
    cov, _, _ = pca_denoised_covariance(returns, n_components=3)
    weights, stats = optimise_best_sharpe(returns, cov, max_weight=0.30, target_return_steps=20)

    assert np.isclose(weights["weight"].sum(), 1.0)
    assert (weights["weight"] >= -1e-8).all()
    assert (weights["weight"] <= 0.3000001).all()
    assert set(stats) == {"weights", "annual_return", "volatility", "sharpe_ratio"}
    assert np.isfinite([stats["annual_return"], stats["volatility"], stats["sharpe_ratio"]]).all()
