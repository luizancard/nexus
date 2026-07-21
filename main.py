"""NEXUS Phase 2 experiment driver: the OSM-vs-imagery accessible-routing comparison.

Runs the whole controlled experiment end-to-end on the real fused/validated data
and writes a reproducible results file. Everything is deterministic under `--seed`.

    python -m main                 # full run, 1000 stratified OD pairs, seed 42
    python -m main --n-pairs 200   # quicker smoke run

Pipeline:
  1. Build the five condition graphs (baseline, full_validated, full_raw,
     ramp_only, obstacle_only) and run the SS9.4 coverage guards.
  2. Sample a stratified OD set (shared across all conditions).
  3. Metrics 1-3 for baseline vs each condition, each with bootstrap CIs and a
     paired Wilcoxon test.
  4. Robustness: improvement-by-distance, and a weight-magnitude sensitivity grid.
  5. Dump `evaluation/results/experiment_results.json` and print a report.

No number is invented: every figure printed here is computed from the real data
in this run. A modest or null result is reported as-is.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from core import graph_manager as gm
from core.impedance_model import WHEELCHAIR_PROFILE, ExperimentConfig
from evaluation.metrics_calculator import (
    ComparisonResult,
    compare_conditions,
    improvement_rate_by_distance,
)

RESULTS_DIR = Path("evaluation/results")
RESULTS_PATH = RESULTS_DIR / "experiment_results.json"

# §9.4 precision, reported alongside every result so the detection-error bound is
# always visible next to the effect size.
PRECISION = {"ramp_present": 0.908, "fixed_obstacle_present": 0.64}


def _result_to_dict(r: ComparisonResult) -> dict[str, Any]:
    """Compact, JSON-serialisable view (drops the raw per-OD cost arrays)."""
    c = r.classification
    return {
        "n_pairs": r.n_pairs,
        "n_solved": r.n_solved,
        "n_unchanged": r.n_unchanged,
        "n_differ": r.n_differ,
        "route_diff_rate": r.route_diff_rate,
        "route_diff_ci95": r.route_diff_ci,
        "classification": c,
        "net_improvement": c["improvement"] - c["regression"],
        "net_improvement_rate_of_all": (c["improvement"] - c["regression"]) / r.n_solved
        if r.n_solved
        else 0.0,
        "improvement_rate_of_differ": r.improvement_rate_of_differ,
        "improvement_rate_of_differ_ci95": r.improvement_rate_of_differ_ci,
        "improvement_rate_of_all": r.improvement_rate_of_all,
        "regression_causes": r.regression_causes,
        "median_cost_base": r.median_cost_base,
        "median_cost_full": r.median_cost_full,
        "wilcoxon_stat": r.wilcoxon_stat,
        "wilcoxon_p": r.wilcoxon_p,
        "mean_ramp_access_base": r.mean_ramp_access_base,
        "mean_ramp_access_full": r.mean_ramp_access_full,
        "aggregate_magnitude": r.agg,
        "obstacle_wilcoxon_p": r.obstacle_wilcoxon_p,
        "n_routes_fewer_obstacles": r.n_routes_fewer_obstacles,
        "n_routes_more_obstacles": r.n_routes_more_obstacles,
    }


def _print_decomposition(r: ComparisonResult) -> None:
    """Full decomposition + tangible aggregate magnitude for one comparison."""
    c = r.classification
    n = r.n_solved
    print(f"  decomposition of {n} solved pairs:")
    print(f"    unchanged   {r.n_unchanged:4d} ({r.n_unchanged / n:.1%})")
    for k in ("improvement", "regression", "neutral", "mixed"):
        of_changed = c[k] / r.n_differ if r.n_differ else 0.0
        print(f"    {k:11} {c[k]:4d} ({c[k] / n:.1%} of all, {of_changed:.1%} of changed)")
    net = c["improvement"] - c["regression"]
    print(f"    NET (imp-reg) {net:+d} ({net / n:+.1%} of all pairs)")
    if r.regression_causes:
        print(f"    regression causes (component worsened): {r.regression_causes}")
    a = r.agg
    ob_avoided = a["obstacles_base"] - a["obstacles_full"]
    print("  tangible aggregate magnitude (summed over all routes):")
    print(
        f"    validated obstacles traversed: {a['obstacles_base']:.0f} -> {a['obstacles_full']:.0f} "
        f"(avoided {ob_avoided:.0f}, -{ob_avoided / a['obstacles_base']:.1%} ; "
        f"{r.n_routes_fewer_obstacles} routes fewer / {r.n_routes_more_obstacles} more; "
        f"Wilcoxon p={r.obstacle_wilcoxon_p})"
    )
    print(
        f"    confirmed missing-curb segments: {a['curb_barriers_base']:.0f} -> {a['curb_barriers_full']:.0f} "
        f"(net {a['curb_barriers_base'] - a['curb_barriers_full']:+.0f})"
    )
    print(
        f"    steep (>NBR) metres traversed:   {a['high_grade_len_base']:.0f} -> {a['high_grade_len_full']:.0f} "
        f"(net {a['high_grade_len_base'] - a['high_grade_len_full']:+.0f})"
    )
    print(
        f"    validated ramp access:           {a['ramp_access_base']:.0f} -> {a['ramp_access_full']:.0f} "
        f"(net {a['ramp_access_full'] - a['ramp_access_base']:+.0f})"
    )


def run_weight_sweep(
    base_graph: Any,
    gt_graph: Any,
    pairs: list[tuple[Any, Any]],
    ramp_discounts: tuple[float, ...] = (0.70, 0.80, 0.90),
    obstacle_penalties: tuple[float, ...] = (15.0, 25.0, 40.0),
) -> list[dict[str, Any]]:
    """Sensitivity grid: recompute the headline over perturbed weights.

    Baseline cost is invariant to these two weights (no ramp fires, no obstacle
    is 'present' in baseline), so the baseline graph is built once and reused;
    only full_validated is rebuilt per grid point. Shows the conclusion is not an
    artifact of any single weight choice.
    """
    out = []
    for rd in ramp_discounts:
        for op in obstacle_penalties:
            cfg = replace(WHEELCHAIR_PROFILE, ramp_discount=rd, obstacle_penalty_m=op)
            full = gm.build_condition_graph("full_validated", cfg)
            res = compare_conditions(base_graph, full, gt_graph, pairs, n_bootstrap=300)
            out.append(
                {
                    "ramp_discount": rd,
                    "obstacle_penalty_m": op,
                    "route_diff_rate": res.route_diff_rate,
                    "improvement_rate_of_differ": res.improvement_rate_of_differ,
                    "improvement_rate_of_all": res.improvement_rate_of_all,
                }
            )
    return out


def _mean_sd(xs: list[float]) -> dict[str, float]:
    """Mean, population SD, min, max of a list (empty -> zeros)."""
    if not xs:
        return {"mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0}
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return {"mean": m, "sd": var**0.5, "min": min(xs), "max": max(xs)}


def run_seed_stability(
    base_graph: Any,
    full_graph: Any,
    gt_graph: Any,
    n_pairs: int,
    seeds: tuple[int, ...] = tuple(range(1, 11)),
) -> dict[str, Any]:
    """Re-draw the OD sample under many seeds and report headline stability.

    The §12.5 headline uses one OD draw (seed=42); the bootstrap CIs quantify
    uncertainty *within* that draw but not *across* independent draws. This
    resamples the OD set under `seeds` (condition graphs are seed-independent, so
    they are reused, not rebuilt) and reports mean ± SD and range for the four
    headline quantities -- the direct answer to "did you just pick one sample?".
    """
    per_seed = []
    for s in seeds:
        pairs = gm.sample_od_pairs(base_graph, n_pairs=n_pairs, seed=s)
        r = compare_conditions(base_graph, full_graph, gt_graph, pairs, seed=s, n_bootstrap=200)
        c = r.classification
        net = (c["improvement"] - c["regression"]) / r.n_solved if r.n_solved else 0.0
        obstacles_avoided = r.agg["obstacles_base"] - r.agg["obstacles_full"]
        per_seed.append(
            {
                "seed": s,
                "route_diff_rate": r.route_diff_rate,
                "improvement_rate_of_differ": r.improvement_rate_of_differ,
                "net_improvement_rate_of_all": net,
                "obstacles_avoided": obstacles_avoided,
            }
        )
    keys = ("route_diff_rate", "improvement_rate_of_differ", "net_improvement_rate_of_all", "obstacles_avoided")
    summary = {k: _mean_sd([row[k] for row in per_seed]) for k in keys}
    return {"seeds": list(seeds), "per_seed": per_seed, "summary": summary}


def run_threshold_sensitivity(
    base_graph: Any,
    full_graph: Any,
    gt_graph: Any,
    pairs: list[tuple[Any, Any]],
    grade_margins: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0),
    high_grade_thresholds: tuple[float, ...] = (0.05, 0.0833, 0.12),
) -> list[dict[str, Any]]:
    """Vary the improvement classifier's two free parameters and report stability.

    The improvement metric draws two lines a reviewer could call arbitrary: the
    grade-noise margin (`GRADE_MARGIN_M`) and the steep-grade threshold
    (`NBR9050_MAX_RAMP_GRADE`). Both are threaded through `compare_conditions`;
    this sweeps them over a grid and reports the headline improvement rate, so the
    finding is shown not to hinge on where those lines sit.
    """
    out = []
    for gm_margin in grade_margins:
        for thr in high_grade_thresholds:
            res = compare_conditions(
                base_graph,
                full_graph,
                gt_graph,
                pairs,
                n_bootstrap=200,
                grade_margin=gm_margin,
                high_grade_threshold=thr,
            )
            out.append(
                {
                    "grade_margin_m": gm_margin,
                    "high_grade_threshold": thr,
                    "route_diff_rate": res.route_diff_rate,
                    "improvement_rate_of_differ": res.improvement_rate_of_differ,
                }
            )
    return out


def render_all_figures(graphs, gt, base, fv_cmp, by_dist, comparisons) -> list:
    """Render every deliverable figure from the current verified run."""
    from core.routing_algorithms.dijkstra import shortest_path
    from evaluation import visualizer as viz
    from evaluation.metrics_calculator import score_path_hazards

    full = graphs["full_validated"]
    paths = [
        viz.plot_accessibility_heatmap(full),
        viz.plot_cost_distribution(fv_cmp.base_costs, fv_cmp.full_costs),
        viz.plot_improvement_by_distance(by_dist),
        viz.plot_ablation({k: _result_to_dict(v) for k, v in comparisons.items()}),
        viz.plot_decomposition(fv_cmp.classification, fv_cmp.n_unchanged),
    ]
    # Landmark route overlay: first landmark pair whose full route sheds a real obstacle.
    lm = gm.resolve_landmarks(base)
    for i in range(len(lm)):
        for j in range(len(lm)):
            if i == j:
                continue
            rb = shortest_path(base, lm[i][1], lm[j][1])
            rf = shortest_path(full, lm[i][1], lm[j][1])
            if rb is None or rf is None or rb.nodes == rf.nodes:
                continue
            if score_path_hazards(rf, gt).obstacles_real < score_path_hazards(rb, gt).obstacles_real:
                paths.append(viz.plot_route_overlay(
                    full, rb.nodes, rf.nodes,
                    viz.FIGURES_DIR / "route_overlay_landmark.png",
                    title=f"{lm[i][0].name} -> {lm[j][0].name}",
                ))
                return paths
    return paths


def run_experiment(n_pairs: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Execute the full experiment and return a results dict."""
    print(f"Building condition graphs (config={WHEELCHAIR_PROFILE})...")
    graphs = gm.build_all_conditions(WHEELCHAIR_PROFILE)
    print(f"  guards passed; conditions: {list(graphs)}")

    gt = gm.load_ground_truth_graph()
    base = graphs["baseline"]

    pairs = gm.sample_od_pairs(base, n_pairs=n_pairs, seed=seed)
    print(f"Sampled {len(pairs)} stratified OD pairs (seed={seed}).\n")

    # Metrics 1-3: baseline vs each condition.
    comparisons: dict[str, ComparisonResult] = {}
    for cond in ("full_validated", "full_raw", "ramp_only", "obstacle_only"):
        res = compare_conditions(base, graphs[cond], gt, pairs, seed=seed)
        comparisons[cond] = res
        print(f"[baseline vs {cond}]\n  {res.summary()}")
        if cond == "full_validated":
            _print_decomposition(res)
        print()

    # Detection-error gap: headline honesty number.
    gap = (
        comparisons["full_validated"].improvement_rate_of_differ
        - comparisons["full_raw"].improvement_rate_of_differ
    )
    print(f"Detection-error gap (validated - raw improvement rate): {gap:+.1%}\n")

    # Robustness: effect by OD distance.
    by_dist = improvement_rate_by_distance(
        base, graphs["full_validated"], gt, pairs, lambda o, d: gm.straight_line_distance(base, o, d)
    )
    print("Improvement by OD-distance band (full_validated):")
    for row in by_dist:
        print(
            f"  {row['dist_lo_m']:.0f}-{row['dist_hi_m']:.0f} m (n={row['n']}): "
            f"route-diff {row['route_diff_rate']:.1%}, "
            f"improvement {row['improvement_rate_of_differ']:.1%}"
        )
    print()

    # Robustness: weight-sensitivity grid.
    print("Weight-sensitivity grid (full_validated headline improvement rate):")
    sweep = run_weight_sweep(base, gt, pairs)
    for row in sweep:
        print(
            f"  ramp_discount={row['ramp_discount']:.2f} "
            f"obstacle_penalty={row['obstacle_penalty_m']:.0f}m -> "
            f"route-diff {row['route_diff_rate']:.1%}, "
            f"improvement {row['improvement_rate_of_differ']:.1%}"
        )
    imp_vals = [r["improvement_rate_of_differ"] for r in sweep]
    print(f"  improvement range across grid: {min(imp_vals):.1%} - {max(imp_vals):.1%}\n")

    # Robustness: multi-seed stability -- does the headline survive a different OD draw?
    print("Multi-seed stability (baseline vs full_validated, seeds 1-10):")
    stability = run_seed_stability(base, graphs["full_validated"], gt, n_pairs)
    for row in stability["per_seed"]:
        print(
            f"  seed {row['seed']:2d}: route-diff {row['route_diff_rate']:.1%} | "
            f"improvement {row['improvement_rate_of_differ']:.1%} | "
            f"net {row['net_improvement_rate_of_all']:+.1%} | "
            f"obstacles avoided {row['obstacles_avoided']:.0f}"
        )
    for name, key in (("route-diff", "route_diff_rate"), ("improvement", "improvement_rate_of_differ"),
                      ("net", "net_improvement_rate_of_all"), ("obstacles", "obstacles_avoided")):
        s = stability["summary"][key]
        fmt = "{:.0f}" if key == "obstacles_avoided" else "{:.1%}"
        print(
            f"  {name:11} mean {fmt.format(s['mean'])} SD {fmt.format(s['sd'])} "
            f"range [{fmt.format(s['min'])}, {fmt.format(s['max'])}]"
        )
    print()

    # Robustness: metric-threshold sensitivity -- is 'improvement' an artifact of
    # where the two classifier lines (grade margin, steep-grade threshold) are drawn?
    print("Metric-threshold sensitivity (full_validated improvement rate):")
    threshold_sens = run_threshold_sensitivity(base, graphs["full_validated"], gt, pairs)
    for row in threshold_sens:
        print(
            f"  grade_margin={row['grade_margin_m']:.0f}m thr={row['high_grade_threshold']:.4f} -> "
            f"improvement {row['improvement_rate_of_differ']:.1%}"
        )
    ts_vals = [r["improvement_rate_of_differ"] for r in threshold_sens]
    print(f"  improvement range across grid: {min(ts_vals):.1%} - {max(ts_vals):.1%}\n")

    # Model-refinement check: the naive ramp (discounts the slope term too) vs the
    # slope-protected default. A first-principles correction -- a ramp does not
    # flatten a hill -- reported so the fix is measured, not assumed.
    print("Cost-model refinement (ramp discount and slope):")
    naive_cfg = replace(WHEELCHAIR_PROFILE, ramp_protects_slope=False)
    naive_base = gm.build_condition_graph("baseline", naive_cfg)
    naive_full = gm.build_condition_graph("full_validated", naive_cfg)
    naive_res = compare_conditions(naive_base, naive_full, gt, pairs, seed=seed)
    for label, res in (("naive (discounts slope)", naive_res), ("refined (slope-protected)", comparisons["full_validated"])):
        c = res.classification
        a = res.agg
        print(
            f"  {label:26} improvement(of changed) {res.improvement_rate_of_differ:.1%} | "
            f"regressions {c['regression']} | net {c['improvement'] - c['regression']:+d} | "
            f"obstacles avoided {a['obstacles_base'] - a['obstacles_full']:.0f} | "
            f"extra steep m {a['high_grade_len_full'] - a['high_grade_len_base']:+.0f}"
        )
    print()

    landmarks = [
        {"name": lm.name, "node": node, "snap_m": snap}
        for lm, node, snap in gm.resolve_landmarks(base)
    ]

    # Render every article figure from this same verified run.
    print("Rendering figures...")
    fv_cmp = comparisons["full_validated"]
    fig_paths = render_all_figures(graphs, gt, base, fv_cmp, by_dist, comparisons)
    for p in fig_paths:
        print(f"  {p}")
    print()

    return {
        "config": {
            "seed": seed,
            "n_pairs": n_pairs,
            "profile": {
                "surface_factors": WHEELCHAIR_PROFILE.surface_factors,
                "slope_factor_at_nbr_ceiling": WHEELCHAIR_PROFILE.slope_factor_at_nbr_ceiling,
                "ramp_discount": WHEELCHAIR_PROFILE.ramp_discount,
                "obstacle_penalty_m": WHEELCHAIR_PROFILE.obstacle_penalty_m,
                "steps_penalty_m": WHEELCHAIR_PROFILE.steps_penalty_m,
            },
            "precision_9_4": PRECISION,
        },
        "comparisons": {k: _result_to_dict(v) for k, v in comparisons.items()},
        "model_refinement": {
            "naive_ramp_discounts_slope": _result_to_dict(naive_res),
            "refined_slope_protected": _result_to_dict(comparisons["full_validated"]),
        },
        "detection_error_gap": gap,
        "improvement_by_distance": by_dist,
        "weight_sensitivity": sweep,
        "seed_stability": stability,
        "threshold_sensitivity": threshold_sens,
        "landmarks": landmarks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS Phase 2 routing experiment.")
    parser.add_argument("--n-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    results = run_experiment(n_pairs=args.n_pairs, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
