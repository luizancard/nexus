"""The NEXUS impedance (cost) model: turns one fused-graph edge into a single
scalar traversal cost for a mobility-impaired (wheelchair) pedestrian.

This is the instrument of the Phase 2 experiment, not its finding. It extends
the research proposal's skeleton -- `Custo = Distancia x Fator_superficie +
Penalidade_obstaculo` -- with a slope term (the dominant biomechanical cost for
a wheelchair) and a *ramp reducer* (the high-leverage, hand-validated imagery
signal). The whole point of the controlled OSM-vs-imagery comparison is that the
SAME cost model runs over the same graph under different data conditions, so any
route difference is attributable to the data, never to the formula. Every weight
therefore lives in one `ExperimentConfig` dataclass: nothing is hand-tuned to a
desired result, and the sensitivity sweep varies these fields programmatically.

Design rules this module enforces, each traceable to `docs/METHODOLOGY.md`:

- **Non-negative costs only.** Dijkstra requires it; every term here is >= 0 and
  the ramp reducer is a bounded multiplicative factor, never a subtraction, so
  no edge can go negative.
- **3-state hazards are handled explicitly** (steps, obstacle): `present` /
  `absent` / `unknown`. `unknown` is NEVER silently treated as safe *nor*
  penalised -- we can only route around a hazard we actually detected (SS4). Over
  half the graph is `unknown` on these attributes.
- **The ramp discount fires on real evidence only** (`ramp_present == 'present'`
  AND NOT `ramp_present_imputed`), so the pessimistic imputation default can
  never trigger it, and ramp *absence* stays cost-neutral -- no fabricated curb
  penalty on the ~440 edges that have no curb at all (SS11.3).
- **Fully/mostly-imputed attributes are excluded** (weight 0): width, handrail,
  smoothness, tactile, marked-crossing. Their imputation rates are reported so
  the reader sees why they carry no weight (SS11.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data_pipeline.geometric_attribute_extractor import NBR9050_MAX_RAMP_SLOPE_PCT
from data_pipeline.imputation_engine import PRESENT

# NBR 9050 (Brazilian accessibility standard) caps a compliant ramp at 8.33%
# grade. Expressed as a rise/run fraction to match the graph's `grade_abs`.
NBR9050_MAX_RAMP_GRADE = NBR9050_MAX_RAMP_SLOPE_PCT / 100.0  # 0.0833
# Below this the terrain reads as effectively flat for a wheelchair; no penalty.
FLAT_GRADE = 0.03


@dataclass(frozen=True)
class ExperimentConfig:
    """Every tunable weight of the wheelchair impedance model, in one place.

    Frozen so a config can be safely shared across the three routing conditions
    and hashed into a sweep grid. All defaults are justified in METHODOLOGY SS12.1;
    the sensitivity analysis re-runs the experiment over perturbations of these
    exact fields to show the conclusion is not an artifact of any single choice.
    """

    # --- Surface (Fator_superficie): multiplicative rolling-resistance factor.
    # Modest by design so surface never dominates raw distance; imputed 'fair'
    # (~42% of edges) is the neutral middle and cannot inflate cost.
    surface_factors: dict[str, float] = field(
        default_factory=lambda: {"good": 1.0, "fair": 1.15, "poor": 1.4}
    )

    # --- Slope (topographic term): piecewise-linear on |grade|, anchored to the
    # NBR 9050 8.33% accessible-ramp ceiling. Flat below FLAT_GRADE; linear rise
    # up to the ceiling; steep escalation beyond it (a non-compliant grade).
    slope_factor_at_nbr_ceiling: float = 1.5
    slope_escalation_per_grade: float = 8.0
    slope_factor_cap: float = 2.3

    # --- Ramp reducer (R_ramp): a validated curb ramp discounts the edge's
    # effort. Fires only on real, non-imputed 'present' evidence. < 1.0 => cheaper.
    ramp_discount: float = 0.80

    # A curb ramp helps a wheelchair user mount a curb; it does NOT flatten a
    # hill. When True (default), the discount applies ONLY to the flat-ground
    # traversal effort (length x surface), leaving the slope surcharge intact, so
    # a ramp can never make a steep segment look cheaper than a flat one. When
    # False, the naive model discounts the whole slope-inflated effort -- which
    # SS12.5 found pulls routes onto steeper terrain (grade drove 79% of that
    # model's route regressions). Both are runnable so the correction is measured,
    # not assumed.
    ramp_protects_slope: bool = True

    # --- Obstacle penalty: additive, SOFT. A false positive over-avoids a clear
    # path (suboptimal, not unsafe), so it must stay recoverable -- never a block.
    obstacle_penalty_m: float = 25.0

    # --- Steps penalty: additive, STRONG (near-impassable for a wheelchair) but
    # still finite, so the graph stays connected and a step-only destination is
    # reachable at high cost rather than unreachable. Only the 8 reliable OSM
    # step edges ever trigger it.
    steps_penalty_m: float = 500.0


# The single primary profile for the headline experiment.
WHEELCHAIR_PROFILE = ExperimentConfig()


def surface_factor(edge: dict[str, Any], config: ExperimentConfig) -> float:
    """Multiplicative surface-quality factor for one edge (>= 1.0)."""
    tier = edge.get("surface_material_tier")
    # An unrecognised/missing tier falls back to the neutral middle ('fair'),
    # matching the imputation policy rather than silently assuming the best case.
    return config.surface_factors.get(tier, config.surface_factors["fair"])


def slope_factor(edge: dict[str, Any], config: ExperimentConfig) -> float:
    """Multiplicative topographic factor from the edge's absolute grade (>= 1.0).

    `grade_abs` is a rise/run fraction restored as a real float by
    `load_fused_graph`; it is DEM-derived and therefore identical across all
    data conditions, so this term shapes routes but never drives the
    OSM-vs-imagery comparison.
    """
    grade = _as_float(edge.get("grade_abs"), default=0.0)
    if grade <= FLAT_GRADE:
        return 1.0
    if grade <= NBR9050_MAX_RAMP_GRADE:
        # Linear from 1.0 at FLAT_GRADE to slope_factor_at_nbr_ceiling at the ceiling.
        span = NBR9050_MAX_RAMP_GRADE - FLAT_GRADE
        frac = (grade - FLAT_GRADE) / span
        return 1.0 + frac * (config.slope_factor_at_nbr_ceiling - 1.0)
    # Beyond the accessible ceiling: escalate steeply, capped.
    excess = grade - NBR9050_MAX_RAMP_GRADE
    factor = config.slope_factor_at_nbr_ceiling + config.slope_escalation_per_grade * excess
    return min(factor, config.slope_factor_cap)


def ramp_factor(edge: dict[str, Any], config: ExperimentConfig) -> float:
    """Bounded ramp discount (<= 1.0). Fires ONLY on real, non-imputed evidence.

    Keying on `ramp_present_imputed is False` (a real bool after
    `load_fused_graph`) guarantees the pessimistic imputation default can never
    trigger the discount; a ramp that was never actually observed is neutral,
    not rewarded.
    """
    if edge.get("ramp_present") == PRESENT and edge.get("ramp_present_imputed") is False:
        return config.ramp_discount
    return 1.0


def obstacle_penalty(edge: dict[str, Any], config: ExperimentConfig) -> float:
    """Additive soft obstacle penalty in effort-metres (>= 0).

    Only a detected `present` obstacle is penalised. `unknown` (the imputed
    hazard default, > 55% of edges) contributes 0: we cannot route around a
    hazard we never observed, and treating every unknown as an obstacle would
    render the under-mapped graph impassable (SS4). `absent` is also 0.
    """
    if edge.get("fixed_obstacle_present") == PRESENT:
        return config.obstacle_penalty_m
    return 0.0


def steps_penalty(edge: dict[str, Any], config: ExperimentConfig) -> float:
    """Additive strong steps penalty in effort-metres (>= 0).

    `present` (only the 8 reliable OSM step edges here) => strong avoidance.
    `unknown` (454 edges) contributes 0 -- never assumed safe, but never
    penalised on a mere measurement gap either: we avoid only *known* steps.
    """
    if edge.get("steps_present") == PRESENT:
        return config.steps_penalty_m
    return 0.0


def edge_cost(edge: dict[str, Any], config: ExperimentConfig = WHEELCHAIR_PROFILE) -> float:
    """Total wheelchair traversal cost for one edge, in effort-metres (>= 0).

        effort = length x surface_factor x slope_factor
        cost   = ramp_factor x effort + obstacle_penalty + steps_penalty

    `edge` is a fused-graph edge's attribute dict (as returned by
    `G.edges(..., data=True)`), loaded via `edge_attribute_fusion.load_fused_graph`
    so that `grade_abs` is a float and `ramp_present_imputed` a real bool.

    Ramp handling depends on `config.ramp_protects_slope`:
      - True  (default): discount the flat effort only; slope surcharge untouched.
                 cost_traversal = R_ramp*(len*surf) + (len*surf)*(slope-1)
                                = len*surf*(R_ramp + slope - 1)
      - False (naive):   discount the whole slope-inflated effort.
                 cost_traversal = R_ramp*(len*surf*slope)
    Both agree exactly on flat edges (slope factor == 1) and are non-negative
    (R_ramp + slope - 1 >= 0.8).
    """
    length = _as_float(edge.get("length"), default=0.0)
    flat_effort = length * surface_factor(edge, config)
    slope = slope_factor(edge, config)
    r = ramp_factor(edge, config)
    if config.ramp_protects_slope:
        cost = flat_effort * (r + slope - 1.0)
    else:
        cost = r * flat_effort * slope
    cost += obstacle_penalty(edge, config)
    cost += steps_penalty(edge, config)
    return cost


def _as_float(value: Any, default: float) -> float:
    """Coerce a graph attribute to float, tolerating the string round-trip.

    `load_fused_graph` already restores `length`/`grade_abs` as real floats, but
    this stays defensive so the cost model is correct even if handed an edge
    from a graph loaded without the fusion dtypes -- exactly the string-`'False'`
    class of bug the fusion module warns about.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "ExperimentConfig",
    "WHEELCHAIR_PROFILE",
    "edge_cost",
    "surface_factor",
    "slope_factor",
    "ramp_factor",
    "obstacle_penalty",
    "steps_penalty",
    "NBR9050_MAX_RAMP_GRADE",
    "FLAT_GRADE",
]
