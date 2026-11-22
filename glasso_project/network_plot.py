"""Standardized network drawing, so graphs from different datasets / alphas /
methods can be compared by eye.

Two things are held fixed across every plot:
  * node positions: each ticker's slot depends only on its sector and its index in
    config.SECTORS -- never on the edges or on which other tickers are present.
  * edge scale: width and opacity are absolute functions of |partial corr|
    (not normalized per plot), so the same strength looks the same everywhere.

Layouts:
  "ring"     all tickers on one circle, grouped by sector with a gap between
             sectors. Within-sector edges are short chords near the rim;
             cross-sector edges cut through the middle.
  "clusters" each sector is a small circle of tickers; sectors sit on a large
             circle. Within-sector edges stay inside a cluster.

Usage:
    from network_plot import draw_network
    draw_network(edges_df, title="...", fname="net.png")       # edges: a, b, partial_corr
    draw_network(P, tickers=order, fname="net.png")            # or a partial-corr matrix
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from glasso_project.data_pipeline.config import SECTORS

LAYOUTS = ["ring", "clusters"]
WIDTH_PER_UNIT = 12.0   # linewidth (pt) for |partial corr| = 1
FULL_OPACITY_AT = 0.4   # |partial corr| at which edges become fully opaque
POS_COLOR, NEG_COLOR = "#c0392b", "#2471a3"


def _groups(tickers=()):
    """Canonical sector -> tickers, plus an 'Other' group for unknown tickers."""
    groups = {s: list(ts) for s, ts in SECTORS.items()}
    known = {t for ts in SECTORS.values() for t in ts}
    extra = sorted(t for t in tickers if t not in known)
    if extra:
        groups["Other"] = extra
    return groups


def sector_colors():
    cmap = plt.get_cmap("tab10")
    c = {s: cmap(i) for i, s in enumerate(SECTORS)}
    c["Other"] = (0.5, 0.5, 0.5, 1.0)
    return c


def fixed_layout(tickers=(), kind="ring"):
    """{ticker: (x, y)} for every ticker in config.SECTORS (+ any extras given).
    Deterministic: positions never depend on the data."""
    groups = _groups(tickers)
    pos, label_pos = {}, {}
    if kind == "ring":
        gap = 1  # empty slots between sectors
        slots = sum(len(ts) for ts in groups.values()) + gap * len(groups)
        k = 0
        for s, ts in groups.items():
            angs = [np.pi / 2 - 2 * np.pi * (k + i) / slots for i in range(len(ts))]
            for t, a in zip(ts, angs):
                pos[t] = (np.cos(a), np.sin(a))
            m = np.mean(angs)
            label_pos[s] = (1.25 * np.cos(m), 1.25 * np.sin(m))
            k += len(ts) + gap
    elif kind == "clusters":
        G = len(groups)
        for g, (s, ts) in enumerate(groups.items()):
            A = np.pi / 2 - 2 * np.pi * g / G
            cx, cy = np.cos(A), np.sin(A)
            r = 0.09 + 0.035 * len(ts)
            for i, t in enumerate(ts):
                a = A + np.pi + 2 * np.pi * (i + 0.5) / len(ts)  # start facing center
                pos[t] = (cx + r * np.cos(a), cy + r * np.sin(a))
            label_pos[s] = ((1 + r + 0.2) * cx, (1 + r + 0.2) * cy)
    else:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    return pos, label_pos


def _edges_from(edges_or_matrix, tickers, tol=1e-8):
    if isinstance(edges_or_matrix, pd.DataFrame) and {"a", "b"} <= set(edges_or_matrix):
        e = edges_or_matrix[["a", "b", "partial_corr"]]
        return list(e.itertuples(index=False, name=None))
    P = np.asarray(edges_or_matrix)
    if tickers is None:
        tickers = list(edges_or_matrix.columns)
    i, j = np.triu_indices_from(P, 1)
    m = np.abs(P[i, j]) > tol
    return [(tickers[a], tickers[b], P[a, b]) for a, b in zip(i[m], j[m])]


def draw_network(edges_or_matrix, tickers=None, layout="ring", title="",
                 fname=None, ax=None, min_abs=0.0):
    """Draw a partial-correlation graph on the fixed layout.

    edges_or_matrix: DataFrame with columns a, b, partial_corr, or a p x p
                     partial-correlation matrix (DataFrame, or array + tickers).
    min_abs:         hide edges with |partial corr| below this (display only).
    """
    edges = [e for e in _edges_from(edges_or_matrix, tickers) if abs(e[2]) >= min_abs]
    nodes = sorted({t for e in edges for t in e[:2]} |
                   set(tickers if tickers is not None else
                       getattr(edges_or_matrix, "columns", [])))
    pos, label_pos = fixed_layout(nodes, layout)
    colors = sector_colors()
    groups = _groups(nodes)
    sector_of = {t: s for s, ts in groups.items() for t in ts}

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(10, 10))

    # weakest first, so strong edges draw on top
    edges.sort(key=lambda e: abs(e[2]))
    if edges:
        segs = [(pos[a], pos[b]) for a, b, _ in edges]
        w = np.array([abs(e[2]) for e in edges])
        rgba = [matplotlib.colors.to_rgba(POS_COLOR if e[2] > 0 else NEG_COLOR,
                                          0.15 + 0.85 * min(abs(e[2]) / FULL_OPACITY_AT, 1))
                for e in edges]
        ax.add_collection(LineCollection(segs, linewidths=WIDTH_PER_UNIT * w,
                                         colors=rgba, capstyle="round", zorder=1))

    # draw every canonical slot so missing tickers show as empty circles
    for t, (x, y) in pos.items():
        present = t in nodes
        ax.scatter(x, y, s=650, zorder=2, linewidths=0.8,
                   color=colors[sector_of[t]] if present else "white",
                   edgecolors="k" if present else "0.7")
        ax.text(x, y, t, ha="center", va="center", fontsize=7, zorder=3,
                color="k" if present else "0.6")
    for s, (x, y) in label_pos.items():
        ax.text(x, y, s, ha="center", va="center", fontsize=10, fontweight="bold",
                color=colors[s])

    n_pos = sum(e[2] > 0 for e in edges)
    ref = [0.05, 0.2, 0.4]
    handles = [plt.Line2D([], [], color=POS_COLOR, lw=WIDTH_PER_UNIT * v,
                          alpha=0.15 + 0.85 * min(v / FULL_OPACITY_AT, 1)) for v in ref]
    handles.append(plt.Line2D([], [], color=NEG_COLOR, lw=WIDTH_PER_UNIT * 0.2))
    ax.legend(handles, [f"|ρ| = {v}" for v in ref] + ["negative ρ"],
              loc="lower right", fontsize=8, title="partial corr ρ", frameon=False)
    sub = f"{len(edges)} edges ({n_pos} +, {len(edges) - n_pos} −)"
    if min_abs:
        sub += f", showing |ρ| ≥ {min_abs}"
    ax.set_title(f"{title}\n{sub}" if title else sub)
    lim = 1.6
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.axis("off")

    if own:
        fig.tight_layout()
        if fname:
            fig.savefig(fname, dpi=150)
        plt.close(fig)
    return ax
