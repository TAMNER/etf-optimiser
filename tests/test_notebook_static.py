import ast
import json
from pathlib import Path


NOTEBOOK = Path("etf_portfolio_optimiser_colab.ipynb")


def notebook_text():
    nb = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


def test_notebook_is_valid_json_and_code_cells_parse():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4

    for index, cell in enumerate(nb["cells"], 1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
        ast.parse(source or "\n", filename=f"notebook_cell_{index}")


def test_notebook_contains_required_workflow_steps():
    text = notebook_text()

    for required in [
        "drive.mount('/content/drive')",
        "investengine_etfs.csv",
        "scrape_investengine_etfs",
        "SCRAPE_DETAIL_PAGES = False",
        "SCRAPE_REQUEST_TIMEOUT = 10",
        "YAHOO_SUFFIXES = ['.L', '', '.DE', '.MI', '.AS', '.PA', '.SW', '.IR', '.TO', '.AX']",
        "symbol_cache.json",
        "YFINANCE_REQUEST_TIMEOUT = 15",
        "timeout=YFINANCE_REQUEST_TIMEOUT",
        "period='5y'",
        "pca_denoised_covariance",
        "optimise_best_sharpe",
        "'full_5y': log_returns",
        "'last_252d': log_returns.tail(TRADING_DAYS)",
        "risk_metrics.csv",
        "pca_explained_variance.csv",
        "pca_loadings.csv",
        "optimal_portfolio.csv",
        "optimal_portfolio_stats.csv",
    ]:
        assert required in text
