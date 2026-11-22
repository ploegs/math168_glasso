"""Run the full chain: download.py -> analyze.py -> plot.py

Run from the PROJECT folder (the one containing glasso_project/).

Each step can also be run on its own (see each file's docstring):
    python -m glasso_project.data_pipeline.download --interval 1wk --start 2005-01-01
    python -m glasso_project.data_pipeline.analyze 1wk_2005_2026
    python -m glasso_project.data_pipeline.plot 1wk_2005_2026

This runner downloads (unless --skip-download), analyzes, and plots:
    python -m glasso_project.data_pipeline.data_review --interval 1wk --start 2005-01-01
    python -m glasso_project.data_pipeline.data_review --interval 1wk --start 2005-01-01 --skip-download
"""
import argparse

import glasso_project.data_pipeline.download as download
from glasso_project.data_pipeline.analyze import analyze
from glasso_project.data_pipeline.config import default_name
from glasso_project.data_pipeline.plot import plot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    download.add_args(ap)
    ap.add_argument("--skip-download", action="store_true",
                    help="reuse existing data/<name>/prices.csv")
    ap.add_argument("--simple", action="store_true", help="simple instead of log returns")
    ap.add_argument("--window", type=int, help="rolling window in periods")
    a = ap.parse_args()

    name = a.name or default_name(a.interval, a.start, a.end)
    if not a.skip_download:
        download.download(a.tickers, a.start, a.end, a.interval, name, a.force)
    analyze(name, log=not a.simple)
    plot(name, a.window)


if __name__ == "__main__":
    main()
