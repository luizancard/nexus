"""The scientific output of Phase 2: does imagery-augmented data produce
*measurably more accessible* routes than OSM alone, and how much of that effect
survives the measured detection error?

Three metrics, over a shared OD sample, comparing a baseline condition to a
"full" condition (headline: full_validated; probe: full_raw; ablations:
ramp_only / obstacle_only):

1. **Route-difference rate** -- fraction of OD pairs whose path changes. A
   necessary-but-not-sufficient signal (a different route is not automatically a
   better one), reported with a bootstrap 95% CI.

2. **Improvement rate** -- the headline. Among pairs whose route changed, how
   many are a *genuine accessibility improvement* judged on HAND-VALIDATED
   ground truth, scored entirely independently of the cost model that produced
   the routes. Each path gets a hazard vector

       H = (steps_real, obstacles_real, curb_barrier_real, high_grade_len)

   read off the validated graph; the full route improves iff it is <= the
   baseline on every component and strictly < on at least one (with a distance
   margin on the continuous grade term to ignore reroute noise). Because H never
   touches the impedance weights, a route that is cheaper *in the model* can
   still score NEUTRAL or REGRESSED -- that is the finding, not a bug, and it is
   what makes the claim non-circular.

3. **Cost-distribution comparison** -- per-condition route-cost distribution and
   a paired Wilcoxon signed-rank test on full-vs-baseline cost. Descriptive on
   magnitude (a lower full cost is partly true by construction where ramps
   discount); metric #2 is the independent validity check.

Uncertainty is first-class: uncertain / imputed edges contribute 0 to every
ground-truth count (neither credited nor blamed), and every rate carries a
bootstrap CI. Nothing here is asserted without the number behind it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from core.impedance_model import NBR9050_MAX_RAMP_GRADE
from core.routing_algorithms.dijkstra import Route, shortest_path
from data_pipeline.imputation_engine import ABSENT, PRESENT

# A grade-length change below this many metres is reroute noise, not a real
# reduction in steep-terrain exposure -- ignored when comparing the continuous
# `high_grade_len` component so it cannot manufacture spurious improvements.
GRADE_MARGIN_M = 5.0


@dataclass(frozen=True)
class Hazards:
    """Ground-truth accessibility features along one path, from the validated graph.

    The first four are *hazards* (lower is better) and are the ONLY axes the
    improvement classifier uses. `ramp_access` is a *positive* feature (higher is
    better) reported descriptively but deliberately kept OUT of the classifier:
    the full condition's cost model already optimises toward ramps via the
    discount, so crediting ramps gained as an "improvement" would be circular.
    It is surfaced separately so the high-precision curb-ramp signal still gets
    fair, transparent credit.
    """

    steps_real: int          # known OSM step edges (near-impassable)
    obstacles_real: int      # hand-validated present obstacles
    curb_barrier_real: int   # hand-validated ABSENT ramps (a confirmed missing cut)
    high_grade_len: float    # metres traversed above the NBR 9050 grade ceiling
    ramp_access: int = 0     # hand-validated PRESENT ramps (descriptive, higher better)


def score_path_hazards(
    route: Route, gt_graph: Any, high_grade_threshold: float = NBR9050_MAX_RAMP_GRADE
) -> Hazards:
    """Hazard vector for a route, read off the validated ground-truth graph.

    Only edges with actual ground truth contribute: an uncertain/imputed edge
    (no `_validated_value`, or steps not OSM-sourced) counts 0 on every axis --
    it is neither credited nor blamed. Grade is DEM-derived and objective, so it
    always contributes. `high_grade_threshold` (default the NBR 9050 ceiling) is
    exposed so §12.6 can show the finding is not an artifact of that line.
    """
    steps = obstacles = curb = ramp_access = 0
    high_grade_len = 0.0
    for u, v, k in route.edge_keys:
        d = gt_graph[u][v][k]
        # Steps: only the reliable OSM-sourced 'present' edges are ground truth.
        if d.get("steps_present") == PRESENT and d.get("steps_present_source") == "osm":
            steps += 1
        # Obstacle: hand-validated present only (raw detections are not truth).
        if d.get("fixed_obstacle_present_validated_value") == PRESENT:
            obstacles += 1
        # Curb barrier: a hand-validated ABSENT ramp is a confirmed missing cut.
        if d.get("ramp_present_validated_value") == ABSENT:
            curb += 1
        # Ramp access: a hand-validated PRESENT ramp (descriptive positive).
        if d.get("ramp_present_validated_value") == PRESENT:
            ramp_access += 1
        # Grade: objective, condition-independent; length above the threshold.
        if float(d.get("grade_abs", 0.0)) > high_grade_threshold:
            high_grade_len += float(d.get("length", 0.0))
    return Hazards(steps, obstacles, curb, high_grade_len, ramp_access)


def _sign(delta: float) -> int:
    """-1 if full is better (delta<0), +1 if worse (delta>0), 0 if equal."""
    if delta < 0:
        return -1
    if delta > 0:
        return 1
    return 0


def _component_signs(base: Hazards, full: Hazards, grade_margin: float = GRADE_MARGIN_M) -> list[int]:
    """Per-component comparison: -1 full better, +1 full worse, 0 tie.

    Discrete hazard counts compare exactly; the continuous grade term uses
    `grade_margin` (default GRADE_MARGIN_M) so sub-margin wobble reads as a tie.
    """
    signs = [
        _sign(full.steps_real - base.steps_real),
        _sign(full.obstacles_real - base.obstacles_real),
        _sign(full.curb_barrier_real - base.curb_barrier_real),
    ]
    grade_delta = full.high_grade_len - base.high_grade_len
    if abs(grade_delta) < grade_margin:
        signs.append(0)
    else:
        signs.append(_sign(grade_delta))
    return signs


def classify_improvement(base: Hazards, full: Hazards, grade_margin: float = GRADE_MARGIN_M) -> str:
    """Label the full route vs baseline: improvement / regression / neutral / mixed.

    - improvement: no component worse, at least one better.
    - regression:  no component better, at least one worse.
    - neutral:     identical ground-truth hazard profile.
    - mixed:       trades one hazard for another (drops one, adds another).
    """
    signs = _component_signs(base, full, grade_margin)
    has_better = any(s < 0 for s in signs)
    has_worse = any(s > 0 for s in signs)
    if has_better and not has_worse:
        return "improvement"
    if has_worse and not has_better:
        return "regression"
    if not has_better and not has_worse:
        return "neutral"
    return "mixed"


@dataclass
class ComparisonResult:
    """Everything metric 1-3 produce for one (baseline vs full) comparison."""

    n_pairs: int
    n_solved: int
    n_differ: int
    route_diff_rate: float
    route_diff_ci: tuple[float, float]
    # metric 2 -- classification counts over the *differing* pairs
    classification: dict[str, int]
    improvement_rate_of_differ: float
    improvement_rate_of_differ_ci: tuple[float, float]
    improvement_rate_of_all: float
    # metric 3
    base_costs: list[float] = field(default_factory=list)
    full_costs: list[float] = field(default_factory=list)
    wilcoxon_stat: float | None = None
    wilcoxon_p: float | None = None
    median_cost_base: float = 0.0
    median_cost_full: float = 0.0
    # descriptive positive: mean hand-validated ramps traversed per route
    mean_ramp_access_base: float = 0.0
    mean_ramp_access_full: float = 0.0
    # full decomposition over ALL solved pairs (not just differing)
    n_unchanged: int = 0
    # aggregate ground-truth magnitude, summed across all solved routes
    agg: dict[str, float] = field(default_factory=dict)
    # which hazard component drove each regression (component -> count)
    regression_causes: dict[str, int] = field(default_factory=dict)
    # paired Wilcoxon on per-route validated-obstacle delta (full - base)
    obstacle_wilcoxon_p: float | None = None
    n_routes_fewer_obstacles: int = 0
    n_routes_more_obstacles: int = 0

    def summary(self) -> str:
        lo, hi = self.route_diff_ci
        ilo, ihi = self.improvement_rate_of_differ_ci
        return (
            f"n={self.n_pairs} solved={self.n_solved} | "
            f"route-diff {self.route_diff_rate:.1%} [95% CI {lo:.1%}-{hi:.1%}] | "
            f"improvement (of differing) {self.improvement_rate_of_differ:.1%} "
            f"[95% CI {ilo:.1%}-{ihi:.1%}] | "
            f"class={self.classification} | "
            f"median cost {self.median_cost_base:.0f}->{self.median_cost_full:.0f} "
            f"(Wilcoxon p={self.wilcoxon_p}) | "
            f"mean ramp access {self.mean_ramp_access_base:.2f}->{self.mean_ramp_access_full:.2f}"
        )


def route_all(graph: Any, pairs: list[tuple[Any, Any]]) -> list[Route | None]:
    """Route every OD pair in one condition graph (None if unreachable)."""
    return [shortest_path(graph, o, d) for o, d in pairs]


def compare_conditions(
    base_graph: Any,
    full_graph: Any,
    gt_graph: Any,
    pairs: list[tuple[Any, Any]],
    seed: int = 42,
    n_bootstrap: int = 2000,
    grade_margin: float = GRADE_MARGIN_M,
    high_grade_threshold: float = NBR9050_MAX_RAMP_GRADE,
) -> ComparisonResult:
    """Run metrics 1-3 for baseline vs full over `pairs`.

    Routes each condition once, scores every changed route against the ground
    truth, and attaches bootstrap CIs and a paired Wilcoxon test. `gt_graph` is
    the validated graph and is read ONLY for ground-truth hazard scoring -- never
    for cost -- keeping metric #2 independent of the model. `grade_margin` and
    `high_grade_threshold` are the classifier's two free parameters, exposed for
    the §12.6 metric-threshold sensitivity sweep (defaults reproduce the headline).
    """
    routes_base = route_all(base_graph, pairs)
    routes_full = route_all(full_graph, pairs)

    solved: list[int] = []       # indices where both conditions solved
    differ_flags: list[int] = []  # 1 if the route changed, else 0 (aligned to solved)
    labels: list[str] = []       # classification for each differing pair
    base_costs: list[float] = []
    full_costs: list[float] = []

    # Aggregate ground-truth magnitude, summed across all solved routes -- the
    # tangible "N real obstacles avoided" numbers, not just classification rates.
    agg = {
        f"{axis}_{cond}": 0.0
        for axis in ("obstacles", "curb_barriers", "steps", "high_grade_len", "ramp_access")
        for cond in ("base", "full")
    }
    regression_causes: dict[str, int] = {}
    obstacle_deltas: list[float] = []
    comp_names = ("steps", "obstacles", "curb_barriers", "high_grade_len")

    ramp_access_base = 0
    ramp_access_full = 0
    for rb, rf in zip(routes_base, routes_full):
        if rb is None or rf is None:
            continue
        solved.append(1)
        base_costs.append(rb.cost)
        full_costs.append(rf.cost)
        # Score both routes' ground-truth features (for ramp-access reporting and,
        # on changed pairs, the improvement classification).
        hb = score_path_hazards(rb, gt_graph, high_grade_threshold)
        hf = score_path_hazards(rf, gt_graph, high_grade_threshold)
        ramp_access_base += hb.ramp_access
        ramp_access_full += hf.ramp_access
        agg["obstacles_base"] += hb.obstacles_real
        agg["obstacles_full"] += hf.obstacles_real
        agg["curb_barriers_base"] += hb.curb_barrier_real
        agg["curb_barriers_full"] += hf.curb_barrier_real
        agg["steps_base"] += hb.steps_real
        agg["steps_full"] += hf.steps_real
        agg["high_grade_len_base"] += hb.high_grade_len
        agg["high_grade_len_full"] += hf.high_grade_len
        agg["ramp_access_base"] += hb.ramp_access
        agg["ramp_access_full"] += hf.ramp_access
        obstacle_deltas.append(hf.obstacles_real - hb.obstacles_real)
        changed = rb.nodes != rf.nodes
        differ_flags.append(1 if changed else 0)
        if changed:
            lbl = classify_improvement(hb, hf, grade_margin)
            labels.append(lbl)
            if lbl == "regression":
                # Attribute the regression to whichever hazard component(s) worsened.
                for name, s in zip(comp_names, _component_signs(hb, hf, grade_margin)):
                    if s > 0:
                        regression_causes[name] = regression_causes.get(name, 0) + 1

    n_solved = len(solved)
    n_differ = sum(differ_flags)
    classification = {k: labels.count(k) for k in ("improvement", "regression", "neutral", "mixed")}
    n_improve = classification["improvement"]

    route_diff_rate = n_differ / n_solved if n_solved else 0.0
    imp_of_differ = n_improve / n_differ if n_differ else 0.0
    imp_of_all = n_improve / n_solved if n_solved else 0.0

    rng = random.Random(seed)
    route_diff_ci = _bootstrap_ci(differ_flags, rng, n_bootstrap)
    # Improvement-of-differing CI resamples the differing pairs' labels.
    improve_flags = [1 if lbl == "improvement" else 0 for lbl in labels]
    imp_ci = _bootstrap_ci(improve_flags, rng, n_bootstrap)

    wstat, wp = _wilcoxon(base_costs, full_costs)
    # Paired test on per-route obstacle reduction (the tangible safety effect).
    zeros = [0.0] * len(obstacle_deltas)
    _, obs_wp = _wilcoxon(zeros, obstacle_deltas)
    n_fewer = sum(1 for x in obstacle_deltas if x < 0)
    n_more = sum(1 for x in obstacle_deltas if x > 0)

    return ComparisonResult(
        n_pairs=len(pairs),
        n_solved=n_solved,
        n_differ=n_differ,
        route_diff_rate=route_diff_rate,
        route_diff_ci=route_diff_ci,
        classification=classification,
        improvement_rate_of_differ=imp_of_differ,
        improvement_rate_of_differ_ci=imp_ci,
        improvement_rate_of_all=imp_of_all,
        base_costs=base_costs,
        full_costs=full_costs,
        wilcoxon_stat=wstat,
        wilcoxon_p=wp,
        median_cost_base=_median(base_costs),
        median_cost_full=_median(full_costs),
        mean_ramp_access_base=ramp_access_base / n_solved if n_solved else 0.0,
        mean_ramp_access_full=ramp_access_full / n_solved if n_solved else 0.0,
        n_unchanged=n_solved - n_differ,
        agg=agg,
        regression_causes=regression_causes,
        obstacle_wilcoxon_p=obs_wp,
        n_routes_fewer_obstacles=n_fewer,
        n_routes_more_obstacles=n_more,
    )


def improvement_rate_by_distance(
    base_graph: Any,
    full_graph: Any,
    gt_graph: Any,
    pairs: list[tuple[Any, Any]],
    distance_fn: Any,
    n_bands: int = 4,
) -> list[dict[str, Any]]:
    """Route-difference and improvement rate per straight-line distance band.

    Tests whether imagery matters more for longer trips. Bands are equal-count
    quantiles of the OD straight-line distances.
    """
    dists = sorted(distance_fn(o, d) for o, d in pairs)
    if not dists:
        return []
    cuts = [dists[min(int(q / n_bands * len(dists)), len(dists) - 1)] for q in range(1, n_bands)]

    banded: list[list[tuple[Any, Any]]] = [[] for _ in range(n_bands)]
    for o, d in pairs:
        dist = distance_fn(o, d)
        idx = next((i for i, c in enumerate(cuts) if dist < c), n_bands - 1)
        banded[idx].append((o, d))

    out = []
    lo = 0.0
    edges = [0.0] + cuts + [max(dists)]
    for i, band_pairs in enumerate(banded):
        if not band_pairs:
            continue
        res = compare_conditions(base_graph, full_graph, gt_graph, band_pairs, n_bootstrap=500)
        out.append(
            {
                "band": i,
                "dist_lo_m": edges[i],
                "dist_hi_m": edges[i + 1],
                "n": res.n_solved,
                "route_diff_rate": res.route_diff_rate,
                "improvement_rate_of_differ": res.improvement_rate_of_differ,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Small statistical helpers (kept explicit so assumptions are visible)
# --------------------------------------------------------------------------- #
def _bootstrap_ci(
    flags: list[int], rng: random.Random, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a 0/1 vector."""
    if not flags:
        return (0.0, 0.0)
    m = len(flags)
    means = []
    for _ in range(n):
        s = sum(flags[rng.randrange(m)] for _ in range(m))
        means.append(s / m)
    means.sort()
    lo = means[int((alpha / 2) * n)]
    hi = means[min(int((1 - alpha / 2) * n), n - 1)]
    return (lo, hi)


def _wilcoxon(base: list[float], full: list[float]) -> tuple[float | None, float | None]:
    """Paired Wilcoxon signed-rank test on full-vs-baseline cost.

    Returns (statistic, p-value), or (None, None) if the paired differences are
    all zero (scipy raises) or scipy is unavailable. Non-parametric: makes no
    normality assumption about the cost differences.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return (None, None)
    diffs = [f - b for b, f in zip(base, full)]
    if all(abs(x) < 1e-9 for x in diffs):
        return (None, None)
    try:
        res = wilcoxon(diffs)
        return (float(res.statistic), float(res.pvalue))
    except ValueError:
        return (None, None)


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s)
    return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])


__all__ = [
    "Hazards",
    "ComparisonResult",
    "score_path_hazards",
    "classify_improvement",
    "compare_conditions",
    "improvement_rate_by_distance",
    "route_all",
    "GRADE_MARGIN_M",
]
