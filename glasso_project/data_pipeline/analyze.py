"""Step 2: prices -> returns + summary stats.

Run from the PROJECT folder (the one containing glasso_project/).

Reads  data/<name>/prices.csv
Writes data/<name>/returns.csv, data/<name>/summary_stats.csv

    python -m glasso_project.data_pipeline.analyze 1wk_2005_2026
    python -m glasso_project.data_pipeline.analyze 1d_2019_2026 --simple
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from glasso_project.data_pipeline.config import PERIODS_PER_YEAR, dataset_dir, list_datasets, load_meta


def compute_returns(prices, log=True, max_missing=0.05):
    """(Log) returns; drop tickers with too much missing data, then align dates."""
    keep = prices.columns[prices.isna().mean() <= max_missing]
    dropped = sorted(set(prices.columns) - set(keep))
    if dropped:
        print("  Dropping (too many missing):", dropped)
    p = prices[keep].ffill().dropna()
    r = np.log(p).diff() if log else p.pct_change()
    return r.dropna()


def summary_stats(r, periods_per_year):
    return pd.DataFrame({
        "mean": r.mean(),
        "std": r.std(),
        "ann_return": r.mean() * periods_per_year,
        "ann_vol": r.std() * np.sqrt(periods_per_year),
        "skew": r.skew(),
        "excess_kurt": r.kurt(),
        "min": r.min(),
        "max": r.max(),
        "jarque_bera_p": [stats.jarque_bera(r[c])[1] for c in r],
        "lag1_autocorr": [r[c].autocorr(1) for c in r],
    })


def analyze(name, log=True, max_missing=0.05):
    meta = load_meta(name)
    d = dataset_dir(name)
    prices = pd.read_csv(d / "prices.csv", index_col=0, parse_dates=True)
    r = compute_returns(prices, log, max_missing)
    r.to_csv(d / "returns.csv")
    s = summary_stats(r, PERIODS_PER_YEAR[meta["interval"]])
    s.to_csv(d / "summary_stats.csv")

    n, p = r.shape
    ev = np.sort(np.linalg.eigvalsh(r.corr().values))[::-1]
    print(f"[{name}] {'log' if log else 'simple'} returns, interval={meta['interval']}: "
          f"n={n}, p={p}, p/n={p / n:.3f}, {r.index[0].date()} -> {r.index[-1].date()}")
    pd.set_option("display.width", 200, "display.float_format", "{:.4f}".format)
    print(s[["ann_return", "ann_vol", "skew", "excess_kurt", "lag1_autocorr"]]
          .describe().loc[["mean", "min", "max"]])
    print(f"Correlation condition number: {ev[0] / ev[-1]:.1f}  "
          f"(top eigenvalue explains {ev[0] / ev.sum():.1%})")
    print(f"Median Jarque-Bera p-value: {s.jarque_bera_p.median():.2e}")
    return r, s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help=f"dataset name; available: {list_datasets()}")
    ap.add_argument("--simple", action="store_true", help="simple instead of log returns")
    ap.add_argument("--max-missing", type=float, default=0.05)
    a = ap.parse_args()
    analyze(a.name, not a.simple, a.max_missing)


if __name__ == "__main__":
    main()
