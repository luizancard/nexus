"""Figures for the Phase 2 deliverable: the accessibility heat map of Lourdes and
baseline-vs-full route overlays.

Static matplotlib/osmnx renders over the projected (EPSG:31983) graph -- no web
tiles, no folium (neither is installed and neither is needed). Two products:

- `plot_accessibility_heatmap`: every edge coloured by its full_validated
    impedance ratio (cost / length), i.e. how much extra effort a wheelchair user
    pays per metre on that segment. This is the research proposal's promised
    "heat map of Lourdes".
- `plot_route_overlay`: the baseline route vs the full_validated route for one OD
    pair, showing how the imagery data steers the path.

All figures are written as PNGs; nothing requires a display (Agg backend).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never to a screen
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import osmnx as ox

FIGURES_DIR = Path("evaluation/results/figures")


def _impedance_ratios(graph: Any) -> list[float]:
    """Per-edge cost/length ratio, in the order `graph.edges(keys=True)` yields.

    Ratio >= (surface x slope) factor product; 1.0 means the edge costs exactly
    its length (flat, good surface, ramp-discounted or penalty-free).
    """
    ratios = []
    for _u, _v, _k, d in graph.edges(keys=True, data=True):
        length = float(d.get("length", 0.0)) or 1.0
        ratios.append(float(d.get("cost", length)) / length)
    return ratios


def plot_accessibility_heatmap(
    graph: Any,
    out_path: Path = FIGURES_DIR / "lourdes_accessibility_heatmap.png",
    cmap_name: str = "viridis",
    vmax: float | None = None,
) -> Path:
    """Render the Lourdes accessibility heat map (edges coloured by impedance).

    `graph` should be the full_validated condition graph (edges carry `cost`).
    `vmax` caps the colour scale so a few very steep/penalised edges do not wash
    out the rest; defaults to the 95th percentile of the ratios.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ratios = _impedance_ratios(graph)
    if vmax is None:
        s = sorted(ratios)
        vmax = s[int(0.95 * (len(s) - 1))] if s else 2.0
    vmin = 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))
    cmap = matplotlib.colormaps[cmap_name]
    edge_colors = [cmap(norm(r)) for r in ratios]

    fig, ax = ox.plot_graph(
        graph,
        edge_color=edge_colors,
        edge_linewidth=1.4,
        node_size=0,
        bgcolor="white",
        show=False,
        close=False,
    )
    ax.set_title(
        "NEXUS accessibility impedance in Lourdes (full/validated)\n"
        "edge colour = wheelchair effort per metre (cost / length)",
        color="black",
        fontsize=10,
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("impedance ratio (1.0 = flat, unobstructed)", fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_route_overlay(
    graph: Any,
    baseline_nodes: list[Any],
    full_nodes: list[Any],
    out_path: Path,
    title: str = "",
) -> Path:
    """Overlay a baseline route and a full_validated route for one OD pair.

    Both node paths are drawn on the same graph (topology is shared across
    conditions), baseline in one colour and full in another, so the divergence
    the imagery data produces is visible directly.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if baseline_nodes == full_nodes:
        routes = [baseline_nodes]
        colors = ["tab:green"]
    else:
        routes = [baseline_nodes, full_nodes]
        colors = ["tab:red", "tab:green"]  # red = OSM-only, green = imagery-augmented

    fig, ax = ox.plot_graph_routes(
        graph,
        routes=routes,
        route_colors=colors,
        route_linewidth=3,
        node_size=0,
        edge_color="#dddddd",
        edge_linewidth=0.6,
        bgcolor="white",
        orig_dest_size=60,
        show=False,
        close=False,
    )
    ax.set_title(
        title or "Route: OSM-only (red) vs imagery-augmented (green)",
        color="black",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# Consistent colour language across every figure: OSM-only vs imagery-augmented.
C_BASE = "#c0392b"  # red  -- OSM-only / baseline
C_FULL = "#27ae60"  # green -- imagery-augmented / full_validated


def plot_cost_distribution(
    base_costs: list[float],
    full_costs: list[float],
    out_path: Path = FIGURES_DIR / "cost_distribution.png",
) -> Path:
    """ECDF of per-OD route cost, baseline vs full_validated.

    An ECDF (not a histogram) shows the whole distribution shift at once: the
    full curve sitting left of the baseline curve means routes are cheaper across
    the board, not just on average.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for costs, color, label in (
        (base_costs, C_BASE, "OSM-only (baseline)"),
        (full_costs, C_FULL, "imagery-augmented (full/validated)"),
    ):
        xs = sorted(costs)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.step(xs, ys, where="post", color=color, linewidth=2, label=label)
    ax.set_xlabel("route cost (effort-metres)")
    ax.set_ylabel("cumulative fraction of OD pairs")
    ax.set_title("Route-cost distribution shifts lower with imagery-augmented data")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_improvement_by_distance(
    by_distance: list[dict],
    out_path: Path = FIGURES_DIR / "improvement_by_distance.png",
) -> Path:
    """Grouped bars: route-difference and improvement rate per OD-distance band."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['dist_lo_m']:.0f}–{r['dist_hi_m']:.0f}" for r in by_distance]
    diff = [r["route_diff_rate"] * 100 for r in by_distance]
    imp = [r["improvement_rate_of_differ"] * 100 for r in by_distance]
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - w / 2 for i in x], diff, w, color="#7f8c8d", label="routes changed (%)")
    ax.bar(
        [i + w / 2 for i in x], imp, w, color=C_FULL, label="of changed, improved (%)"
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("OD straight-line distance band (m)")
    ax.set_ylabel("percent")
    ax.set_title("Longer trips change more; improvement quality holds across distance")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_ablation(
    comparisons: dict,
    out_path: Path = FIGURES_DIR / "ablation.png",
) -> Path:
    """Bars of improvement-of-changed per condition -- which attribute drives it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    order = [
        ("obstacle_only", "obstacle\nonly"),
        ("full_validated", "full\n(validated)"),
        ("full_raw", "full\n(raw)"),
        ("ramp_only", "ramp\nonly"),
    ]
    vals = [comparisons[k]["improvement_rate_of_differ"] * 100 for k, _ in order]
    labels = [lab for _, lab in order]
    colors = ["#27ae60", "#2e86de", "#e67e22", "#8e44ad"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=8
        )
    ax.set_ylabel("of changed routes, % genuine improvements")
    ax.set_title("Ablation: obstacle avoidance drives measured hazard reduction")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_decomposition(
    classification: dict,
    n_unchanged: int,
    out_path: Path = FIGURES_DIR / "decomposition.png",
) -> Path:
    """Single stacked bar: the complete fate of all OD pairs (nothing unexplained)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    segs = [
        ("unchanged", n_unchanged, "#bdc3c7"),
        ("improvement", classification["improvement"], C_FULL),
        ("mixed", classification["mixed"], "#f1c40f"),
        ("neutral", classification["neutral"], "#95a5a6"),
        ("regression", classification["regression"], C_BASE),
    ]
    total = sum(v for _, v, _ in segs)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    left = 0
    for name, val, color in segs:
        ax.barh(
            0,
            val,
            left=left,
            color=color,
            edgecolor="white",
            label=f"{name}: {val} ({val / total:.0%})",
        )
        # Inline label only for the two large segments; the rest go to the legend.
        if val / total > 0.15:
            ax.text(
                left + val / 2,
                0,
                f"{name}\n{val} ({val / total:.0%})",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )
        left += val
    ax.set_xlim(0, total)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel(f"OD pairs (n={total})")
    ax.set_title("Complete accounting of every route: changed ≠ improved")
    ax.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        frameon=False,
        fontsize=8,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


__all__ = [
    "plot_accessibility_heatmap",
    "plot_route_overlay",
    "plot_cost_distribution",
    "plot_improvement_by_distance",
    "plot_ablation",
    "plot_decomposition",
    "FIGURES_DIR",
]
