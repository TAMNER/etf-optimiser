# ETF Portfolio Optimiser

The main deliverable is `etf_portfolio_optimiser_colab.ipynb`, a Google Colab notebook that:

1. mounts Google Drive;
2. scrapes InvestEngine's ETF universe into `investengine_etfs.csv`;
3. resolves Yahoo Finance symbols and caches them in JSON;
4. downloads 5 years of dividend-adjusted close prices;
5. computes risk metrics, PCA outputs, and optimal portfolios.

## Running in Colab

Open `etf_portfolio_optimiser_colab.ipynb` in Google Colab and run the cells from top to bottom. All data is written under:

```text
/content/drive/MyDrive/etf-optimiser
```

### Colab timeout notes

The InvestEngine scrape is conservative so Colab does not appear to hang while fetching hundreds of ETF detail pages. The notebook:

- reads the ETF range page and sitemap first;
- skips per-ETF detail-page requests by default and falls back to the ticker as the ETF name when a name is not available from the listing;
- prints scraper, Yahoo symbol-resolution, and price-download progress with per-request timeouts.

If you want the scraper to try detail pages for fuller ETF names, set `SCRAPE_DETAIL_PAGES = True` in the notebook. Keep `SCRAPE_MAX_DETAIL_PAGES` low at first, then increase it only if Colab remains responsive.

## Method

- **Prices.** Five years of daily closes from Yahoo Finance through `yfinance`, adjusted for dividends and splits (`auto_adjust=True`), so distributing and accumulating share classes are compared on total return.
- **Symbols.** InvestEngine tickers are London listings, so the `.L` symbol is tried first; other exchanges are fallbacks.
- **Screen.** ETFs with less than 80% of the price history are dropped. Risk metrics are computed for the rest.
- **Universe.** The optimiser runs on at most 150 ETFs (`MAX_OPTIMISATION_ETFS`), the ones with the highest Sharpe ratio in the window.
- **Covariance.** Annualised sample covariance, denoised by keeping the top 5 principal components (`PCA_COMPONENTS`) and averaging the remaining eigenvalues.
- **Optimisation.** Long-only maximum Sharpe ratio with a 30% cap per holding (`MAX_WEIGHT`), solved with SciPy's SLSQP, on the full 5-year window and on the last 252 trading days.

## Limitations

This is a portfolio construction exercise, not a tested strategy, and not investment advice.

- **No out-of-sample test.** Every number is in-sample. The 150 ETFs are picked by their Sharpe ratio and then optimised on the same window, so the reported portfolio Sharpe ratio is biased upwards, probably by a lot.
- **Risk-free rate is zero** (`RISK_FREE_RATE`). UK Bank Rate was well above zero for much of the window, so Sharpe ratios are overstated and the maximum-Sharpe portfolio would shift with a realistic rate.
- **Log returns.** Expected returns are annualised means of daily log returns, and the portfolio return is taken as the weighted sum. Simple returns are the standard Markowitz input; log means sit below arithmetic means by roughly half the variance, which penalises volatile ETFs.
- **No currency conversion.** Prices are used in their listing currency. Returns are computed per ETF, so pounds against pence does not matter, but a fund priced in dollars or euros carries its currency moves and is not converted to sterling.
- **Symbol matching is by ticker only.** A resolved Yahoo symbol is checked for recent prices, not for being the same fund.
- **Shorter history than it says.** After aligning dates, the window starts when the youngest ETF that passed the 80% screen started, so "5 years" can be up to a year shorter. ETFs launched in the last four years are excluded.
- **Survivorship.** The universe is today's InvestEngine list; ETFs that closed during the window are missing.

If you ran an earlier version, delete `symbol_cache.json` and `prices_long.csv` from the Drive folder before running again: the old files hold symbols resolved in the old order and prices without dividends.

## Local tests

The Colab cells that mount Google Drive and call InvestEngine or Yahoo are not run in tests. The maths is mirrored in `etf_optimiser_core.py` and covered with synthetic-data tests.

```bash
python -m pip install -r requirements.txt
python -m pytest
```

The tests cover:

- notebook JSON validity and code-cell syntax;
- presence of the required end-to-end notebook workflow steps;
- long price data to log-return preparation;
- risk metric calculations;
- PCA-denoised covariance output shape and symmetry;
- long-only optimisation constraints, including sum-to-one and max-weight limits.

## Licence

MIT, see `LICENSE`. Built with AI coding assistants; I direct, review and test.
