"""Independent verification of the Phase 2 routing results.

The routing experiment feeds numbers straight into the competition article, so
those numbers must be provably real -- not "the code looks right". This harness
re-derives the headline through a DELIBERATELY SEPARATE code path and asserts it
matches the production path, then re-checks every invariant, determinism, and the
documented claims against the serialised results. It mirrors the §7/§8 pipeline
audits: verify against the real data, trust nothing by construction.

Run: `python -m scripts.verify_experiment`
Exit 0 and "ALL CHECKS PASSED" iff every layer holds.

Four layers:
  1. Independent recomputation -- bare `nx.shortest_path` + a from-scratch hazard
     scorer/classifier (importing NONE of metrics_calculator's scoring logic),
     asserted equal to `compare_conditions` on route-diff, classification, and
     aggregate obstacles avoided. Catches bugs the production path and its author
     share.
  2. Condition-construction cross-check -- independently reconstruct
     full_validated's two imagery attributes from the validated graph and assert
     graph_manager produced the same, guarding `_apply_source`.
  3. Invariants & determinism -- guards (baseline all-absent/all-unknown, single
     component, non-negative cost, 260/294 coverage, full solvability) and
     byte-identical metrics on a repeat run.
  4. Claims ledger -- every headline number documented in METHODOLOGY §12 is
     asserted against a fresh `main.run_experiment` result, so no figure in the
     article lacks a reproducible source.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import networkx as nx

from core import graph_manager as gm
from core.impedance_model import NBR9050_MAX_RAMP_GRADE
from data_pipeline.imputation_engine import ABSENT, PRESENT
from evaluation.metrics_calculator import compare_conditions

SEED = 42
N_PAIRS = 1000
GRADE_MARGIN_M = 5.0

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    mark = "PASS" if condition else "FAIL"
    if condition:
        _passed += 1
    else:
        _failed += 1
    print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))


def approx(a: float, b: float, tol: float = 5e-3) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# Layer 1: independent recomputation (separate code path)
# --------------------------------------------------------------------------- #
def independent_route(graph, o, d):
    """Bare Dijkstra returning the exact edge list, no dijkstra.py helper."""
    try:
        nodes = nx.shortest_path(graph, o, d, weight="cost")
    except nx.NetworkXNoPath:
        return None
    edges = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        # min-cost parallel edge (there are none here, but stay honest)
        k = min(graph[u][v], key=lambda kk: float(graph[u][v][kk]["cost"]))
        edges.append((u, v, k))
    return nodes, edges


def independent_hazards(edges, gt):
    """From-scratch hazard vector; imports no metrics_calculator scoring."""
    steps = obstacles = curb = ramp = 0
    hg = 0.0
    for u, v, k in edges:
        d = gt[u][v][k]
        if d.get("steps_present") == PRESENT and d.get("steps_present_source") == "osm":
            steps += 1
        if d.get("fixed_obstacle_present_validated_value") == PRESENT:
            obstacles += 1
        if d.get("ramp_present_validated_value") == ABSENT:
            curb += 1
        if d.get("ramp_present_validated_value") == PRESENT:
            ramp += 1
        if float(d.get("grade_abs", 0.0)) > NBR9050_MAX_RAMP_GRADE:
            hg += float(d.get("length", 0.0))
    return steps, obstacles, curb, hg, ramp


def independent_classify(hb, hf):
    """From-scratch improvement classifier (5 m grade margin)."""
    signs = []
    for i in (0, 1, 2):  # steps, obstacles, curb -- discrete
        signs.append(_sgn(hf[i] - hb[i]))
    gd = hf[3] - hb[3]
    signs.append(0 if abs(gd) < GRADE_MARGIN_M else _sgn(gd))
    better = any(s < 0 for s in signs)
    worse = any(s > 0 for s in signs)
    if better and not worse:
        return "improvement"
    if worse and not better:
        return "regression"
    if not better and not worse:
        return "neutral"
    return "mixed"


def _sgn(x: float) -> int:
    return -1 if x < 0 else (1 if x > 0 else 0)


def layer1_independent_recompute(graphs, gt, pairs):
    print("Layer 1: independent recomputation (separate code path)")
    base, full = graphs["baseline"], graphs["full_validated"]
    n_differ = obstacles_base = obstacles_full = 0
    cls = {"improvement": 0, "regression": 0, "neutral": 0, "mixed": 0}
    for o, d in pairs:
        rb = independent_route(base, o, d)
        rf = independent_route(full, o, d)
        assert rb and rf, f"unreachable pair {o}->{d}"
        hb = independent_hazards(rb[1], gt)
        hf = independent_hazards(rf[1], gt)
        obstacles_base += hb[1]
        obstacles_full += hf[1]
        if rb[0] != rf[0]:
            n_differ += 1
            cls[independent_classify(hb, hf)] += 1

    prod = compare_conditions(base, full, gt, pairs, seed=SEED, n_bootstrap=1)
    check("route-diff count matches production", n_differ == prod.n_differ,
          f"indep={n_differ} prod={prod.n_differ}")
    for k in cls:
        check(f"classification[{k}] matches", cls[k] == prod.classification[k],
              f"indep={cls[k]} prod={prod.classification[k]}")
    ob_avoided_indep = obstacles_base - obstacles_full
    ob_avoided_prod = prod.agg["obstacles_base"] - prod.agg["obstacles_full"]
    check("obstacles avoided matches", ob_avoided_indep == ob_avoided_prod,
          f"indep={ob_avoided_indep:.0f} prod={ob_avoided_prod:.0f}")


# --------------------------------------------------------------------------- #
# Layer 2: condition construction cross-check
# --------------------------------------------------------------------------- #
def layer2_condition_construction(graphs, gt):
    print("Layer 2: condition-construction cross-check (independent of graph_manager)")
    full = graphs["full_validated"]
    ramp_mismatch = obs_mismatch = 0
    for u, v, k, d in full.edges(keys=True, data=True):
        gd = gt[u][v][k]
        # Independently derive the expected validated value, mirroring
        # graph_manager._validated_values' filter EXACTLY (only {absent,present}
        # count as adjudicated) so a stray third value could never mask a real
        # divergence -- else fall back to the pessimistic default.
        rv = gd.get("ramp_present_validated_value")
        ov = gd.get("fixed_obstacle_present_validated_value")
        exp_ramp = rv if rv in (ABSENT, PRESENT) else "absent"
        exp_obs = ov if ov in (ABSENT, PRESENT) else "unknown"
        if d["ramp_present"] != exp_ramp:
            ramp_mismatch += 1
        if d["fixed_obstacle_present"] != exp_obs:
            obs_mismatch += 1
    check("full_validated ramp_present matches independent reconstruction", ramp_mismatch == 0,
          f"{ramp_mismatch} mismatches")
    check("full_validated obstacle matches independent reconstruction", obs_mismatch == 0,
          f"{obs_mismatch} mismatches")


# --------------------------------------------------------------------------- #
# Layer 3: invariants & determinism
# --------------------------------------------------------------------------- #
def layer3_invariants(graphs, gt, pairs):
    print("Layer 3: invariants & determinism")
    base = graphs["baseline"]
    check("baseline ramp all 'absent'",
          all(d["ramp_present"] == ABSENT for *_, d in base.edges(keys=True, data=True)))
    check("baseline obstacle all 'unknown'",
          all(d["fixed_obstacle_present"] == "unknown" for *_, d in base.edges(keys=True, data=True)))
    for name, g in graphs.items():
        check(f"{name}: single connected component", nx.number_weakly_connected_components(g) == 1)
        check(f"{name}: all costs >= 0", all(d["cost"] >= 0 for *_, d in g.edges(keys=True, data=True)))
    fv = graphs["full_validated"]
    n_ramp = sum(1 for *_, d in fv.edges(keys=True, data=True) if d["ramp_present_imputed"] is False)
    n_obs = sum(1 for *_, d in fv.edges(keys=True, data=True) if d["fixed_obstacle_present_imputed"] is False)
    check("validated ramp coverage == 260", n_ramp == 260, f"{n_ramp}")
    check("validated obstacle coverage == 294", n_obs == 294, f"{n_obs}")
    # Determinism: two identical runs -> identical metrics + CIs.
    r1 = compare_conditions(base, fv, gt, pairs, seed=SEED, n_bootstrap=500)
    r2 = compare_conditions(base, fv, gt, pairs, seed=SEED, n_bootstrap=500)
    check("determinism: identical route-diff", r1.route_diff_rate == r2.route_diff_rate)
    check("determinism: identical improvement + CI",
          r1.improvement_rate_of_differ == r2.improvement_rate_of_differ
          and r1.improvement_rate_of_differ_ci == r2.improvement_rate_of_differ_ci)


# --------------------------------------------------------------------------- #
# Layer 4: claims ledger -- every documented §12 number vs a fresh run
# --------------------------------------------------------------------------- #
def layer4_claims_ledger():
    print("Layer 4: claims ledger (METHODOLOGY §12 numbers vs fresh run)")
    from main import run_experiment

    res = run_experiment(n_pairs=N_PAIRS, seed=SEED)
    fv = res["comparisons"]["full_validated"]
    raw = res["comparisons"]["full_raw"]
    obs_only = res["comparisons"]["obstacle_only"]
    ramp_only = res["comparisons"]["ramp_only"]
    ref = res["model_refinement"]

    # (documented value, actual value, label)
    ledger = [
        (0.497, fv["route_diff_rate"], "§12.5 route-diff 49.7%"),
        (0.634, fv["improvement_rate_of_differ"], "§12.5 improvement 63.4%"),
        (258, fv["net_improvement"], "§12.5 net +258"),
        (315, fv["classification"]["improvement"], "§12.5 improvement count 315"),
        (57, fv["classification"]["regression"], "§12.5 regression count 57"),
        (79, fv["classification"]["neutral"], "§12.5 neutral count 79"),
        (46, fv["classification"]["mixed"], "§12.5 mixed count 46"),
        (2118, fv["aggregate_magnitude"]["obstacles_base"], "§12.5 obstacles base 2118"),
        (1569, fv["aggregate_magnitude"]["obstacles_full"], "§12.5 obstacles full 1569"),
        (0.217, res["detection_error_gap"], "§12.5 detection-error gap 21.7pt"),
        (0.417, raw["improvement_rate_of_differ"], "§12.5 raw improvement 41.7%"),
        (0.903, obs_only["improvement_rate_of_differ"], "§12.5 obstacle_only 90.3%"),
        (0, obs_only["classification"]["regression"], "§12.5 obstacle_only 0 regressions"),
        (0.132, ramp_only["improvement_rate_of_differ"], "§12.5 ramp_only 13.2%"),
        (0.586, ref["naive_ramp_discounts_slope"]["improvement_rate_of_differ"], "§12.6 naive 58.6%"),
        (0.634, ref["refined_slope_protected"]["improvement_rate_of_differ"], "§12.6 refined 63.4%"),
    ]
    for doc_val, actual, label in ledger:
        if isinstance(doc_val, int):
            ok = doc_val == actual
        else:
            ok = approx(doc_val, actual)
        check(label, ok, f"doc={doc_val} actual={actual}")

    # Stability + threshold ranges present and sane.
    st = res["seed_stability"]["summary"]
    check("§12.6 stability route-diff mean ~49.5%", approx(st["route_diff_rate"]["mean"], 0.495, 0.01))
    ts = [r["improvement_rate_of_differ"] for r in res["threshold_sensitivity"]]
    check("§12.6 threshold range 63.0-70.6%", approx(min(ts), 0.630, 0.01) and approx(max(ts), 0.706, 0.01),
          f"[{min(ts):.3f}, {max(ts):.3f}]")
    return res


def main() -> int:
    print("=" * 72)
    print("NEXUS Phase 2 — independent verification of the routing results")
    print("=" * 72)
    graphs = gm.build_all_conditions()
    gt = gm.load_ground_truth_graph()
    pairs = gm.sample_od_pairs(graphs["baseline"], n_pairs=N_PAIRS, seed=SEED)

    layer1_independent_recompute(graphs, gt, pairs)
    layer2_condition_construction(graphs, gt)
    layer3_invariants(graphs, gt, pairs)
    layer4_claims_ledger()

    print("=" * 72)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    if _failed == 0:
        print("ALL CHECKS PASSED — the §12 numbers are reproducible and independently confirmed.")
        return 0
    print("VERIFICATION FAILED — do not use these numbers until resolved.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
