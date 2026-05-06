# ETF Portfolio Optimiser

The main deliverable is `etf_portfolio_optimiser_colab.ipynb`, a Google Colab notebook that:

1. mounts Google Drive;
2. scrapes InvestEngine's ETF universe into `investengine_etfs.csv`;
3. resolves Yahoo Finance symbols and caches them in JSON;
4. downloads 5 years of close prices;
5. computes risk metrics, PCA outputs, and optimal portfolios.

## Running in Colab

Open `etf_portfolio_optimiser_colab.ipynb` in Google Colab and run the cells from top to bottom. All data is written under:

```text
/content/drive/MyDrive/etf-optimiser
```

## Continuous local testing

Colab-specific cells such as Google Drive mounting cannot run in this container, and live InvestEngine/Yahoo calls can be flaky or slow. To continuously test and improve the project locally, the math-heavy optimiser logic is mirrored in `etf_optimiser_core.py` and covered with synthetic-data tests.

Install dependencies and run the tests with:

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
