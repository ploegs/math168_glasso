"""Step 1: download adjusted closes -> data/<name>/prices.csv + meta.json

Run from the PROJECT folder (the one containing glasso_project/).

    python -m glasso_project.data_pipeline.download                                   # daily, 2019 -> 2026-11-20
    python -m glasso_project.data_pipeline.download --interval 1wk --start 2005-01-01 # weekly, named 1wk_2005_2026
    python -m glasso_project.data_pipeline.download --tickers AAPL MSFT --name small --force
"""
import argparse
import json
from datetime import datetime

import yfinance as yf

from glasso_project.data_pipeline.config import DEFAULT_TICKERS, INTERVALS, dataset_dir, default_name


def download(tickers, start, end, interval, name, force=False):
    out = dataset_dir(name)
    if (out / "prices.csv").exists() and not force:
        raise SystemExit(f"{out}/prices.csv exists; pass --force to overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(tickers)} tickers, {interval}, {start} -> {end} ...")
    df = yf.download(tickers, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)["Close"]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    missing = sorted(set(tickers) - set(df.columns))
    if missing:
        print("  WARNING: no data for", missing)
    df.to_csv(out / "prices.csv")

    meta = dict(name=name, tickers=list(df.columns), missing=missing, start=start,
                end=end, interval=interval, rows=len(df),
                first=str(df.index[0].date()), last=str(df.index[-1].date()),
                downloaded_at=datetime.now().isoformat(timespec="seconds"))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  {len(df)} rows x {df.shape[1]} tickers, {meta['first']} -> {meta['last']}")
    print(f"  saved to {out}")
    return name


def add_args(ap):
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-11-20")
    ap.add_argument("--interval", default="1d", choices=INTERVALS)
    ap.add_argument("--name", help="dataset name (default: <interval>_<startyr>_<endyr>)")
    ap.add_argument("--force", action="store_true", help="overwrite existing dataset")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    a = ap.parse_args()
    download(a.tickers, a.start, a.end, a.interval,
             a.name or default_name(a.interval, a.start, a.end), a.force)


if __name__ == "__main__":
    main()
