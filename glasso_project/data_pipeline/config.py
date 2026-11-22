"""Shared settings for the download -> analyze -> plot chain."""
import json
from pathlib import Path

# this file lives in glasso_project/data_pipeline/; data is glasso_project/data/
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ~30 large caps across sectors, so block structure in the graph is plausible
SECTORS = {
    "Tech": ["AAPL", "MSFT", "NVDA", "ORCL", "INTC"],
    "Financials": ["JPM", "BAC", "GS", "MS", "WFC"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Health": ["JNJ", "PFE", "MRK", "UNH", "ABT"],
    "Staples": ["PG", "KO", "PEP", "WMT"],
    "Industrials": ["CAT", "BA", "GE", "UPS"],
    "Utilities": ["DUK", "SO", "NEE"],
}
DEFAULT_TICKERS = [t for ts in SECTORS.values() for t in ts]

INTERVALS = ["1d", "1wk", "1mo"]
PERIODS_PER_YEAR = {"1d": 252, "1wk": 52, "1mo": 12}
DEFAULT_WINDOW = {"1d": 60, "1wk": 26, "1mo": 12}  # rolling window, ~quarter/half-year/year


def dataset_dir(name):
    return DATA / name


def default_name(interval, start, end):
    return f"{interval}_{start[:4]}_{end[:4]}"


def load_meta(name):
    path = dataset_dir(name) / "meta.json"
    if not path.exists():
        raise SystemExit(f"No dataset '{name}' (missing {path}). Run `python -m glasso_project.data_pipeline.download` first. "
                         f"Available: {list_datasets()}")
    return json.loads(path.read_text())


def list_datasets():
    return sorted(p.parent.name for p in DATA.glob("*/meta.json"))
