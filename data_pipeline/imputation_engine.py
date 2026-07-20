"""Pessimistic Safe Fallback imputation for missing accessibility attributes.

The README and research proposal both name this heuristic conceptually
("Pessimistic Safe Fallback") but never define it precisely enough to
implement directly -- "missing data gets the conservative default" is not
one rule, it is (at least) three, depending on what kind of attribute is
missing:

1. Binary INFRASTRUCTURE presence (tactile paving, handrail, curb ramp,
   marked crossing): these are specific, rare, positive investments. If
   neither OSM nor imagery shows one, "probably absent" is a fair inference,
   not an overreaction -- default to absent, no credit without evidence.

2. Binary HAZARD presence (steps, fixed obstacles): the two extremes are
   both wrong. Defaulting to "absent" risks routing someone into a real
   staircase (the exact failure this project exists to prevent). Defaulting
   to "present" would make most of an under-mapped graph look artificially
   impassable. This stays a genuine third state -- `unknown` -- carrying a
   moderate uncertainty signal instead of collapsing into either extreme.

3. Graded/continuous QUALITY (surface material, smoothness, sidewalk
   width): every real street has *some* actual value here; "unknown" is a
   measurement gap, not evidence of the worst case. Given OSM tag coverage
   for these is empirically ~0% in Lourdes (see osm_extractor's
   `diagnosticar_cobertura_tags`), defaulting to the worst tier would mark
   nearly the entire graph as critically narrow/poor-surface -- a fabricated
   claim, not caution. Default to the *middle* tier instead.

This module only resolves a concrete value and flags whether it was
measured or imputed -- it does not decide how much an imputed value should
cost. That is the future impedance model's job (see core/impedance_model.py,
out of scope here); keeping the two separated means this schema survives
whatever the eventual cost function turns out to be.
"""

from __future__ import annotations

import math
from typing import Any

ABSENT = "absent"
PRESENT = "present"
UNKNOWN = "unknown"

# attribute -> (category, ordered tiers if graded, default value when missing)
IMPUTATION_POLICY: dict[str, dict[str, Any]] = {
    # Category 1: infrastructure presence -- unknown -> absent
    "tactile_paving_present": {"category": "infrastructure", "unknown_default": ABSENT},
    "handrail_present": {"category": "infrastructure", "unknown_default": ABSENT},
    "ramp_present": {"category": "infrastructure", "unknown_default": ABSENT},
    "marked_crossing_present": {"category": "infrastructure", "unknown_default": ABSENT},
    # Category 2: hazard presence -- unknown stays unknown, never collapses
    "steps_present": {"category": "hazard", "unknown_default": UNKNOWN},
    "fixed_obstacle_present": {"category": "hazard", "unknown_default": UNKNOWN},
    # Category 3: graded quality -- unknown -> middle tier, not best or worst
    "surface_material_tier": {
        "category": "graded",
        "tiers": ["poor", "fair", "good"],
        "unknown_default": "fair",
    },
    "smoothness_tier": {
        "category": "graded",
        "tiers": ["poor", "fair", "good"],
        "unknown_default": "fair",
    },
    "width_bucket": {
        "category": "graded",
        # Intentionally a local literal, not imported from
        # geometric_attribute_extractor.WIDTH_BUCKET_ORDER: this module is
        # designed to be a generic, standalone policy table with no
        # dependency on the heavier rasterio/numpy-based geometric module,
        # even though the values happen to coincide today.
        "tiers": ["under_50cm", "50_to_90cm", "over_90cm"],
        "unknown_default": "50_to_90cm",
    },
}


def _is_missing(value: Any) -> bool:
    """True if `value` should be treated as not-yet-known.

    Every current call site (OSM tag extraction, segmentation fusion) only
    ever passes `None`/`"unknown"`/a clean string literal, never a raw
    `NaN` -- but this module is meant to be a generic, reusable policy
    table, and a future caller feeding it a raw pandas/GeoDataFrame row
    directly (very plausible given how pandas-heavy the rest of this
    pipeline is) could easily pass a float NaN. `nan == "unknown"` and
    `nan is None` are both False, so without this explicit check a NaN
    would silently pass through as if it were a legitimate measured value.
    """
    if value is None or value == UNKNOWN:
        return True
    return isinstance(value, float) and math.isnan(value)


def impute_missing_attributes(known: dict[str, Any]) -> dict[str, Any]:
    """Fill in missing accessibility attributes per the pessimistic policy.

    A value counts as "missing" if the key is absent from `known`, its
    value is `None`/NaN, or the literal string "unknown" -- callers
    upstream (OSM tag extraction, segmentation fusion) are expected to pass
    one of those for anything they could not determine.

    Args:
        known: Attribute name -> value for whatever OSM/imagery already
            established. Unlisted or None/"unknown" entries are imputed.

    Returns:
        A copy of `known` with every `IMPUTATION_POLICY` attribute resolved
        to a concrete value, plus a companion `<attr>_imputed` boolean for
        each one so downstream consumers can distinguish measured from
        assumed data.
    """
    result = dict(known)
    for attr, policy in IMPUTATION_POLICY.items():
        value = result.get(attr)
        was_missing = _is_missing(value)
        if was_missing:
            result[attr] = policy["unknown_default"]
        result[f"{attr}_imputed"] = was_missing
    return result


def summarize_imputation_rate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute what fraction of rows needed imputation, per attribute.

    Direct input to the methodology doc's coverage reporting -- quantifies
    how much of the final graph is measured vs. assumed, rather than
    leaving that as an implicit, unverified claim.

    Args:
        rows: Attribute dicts already processed by `impute_missing_attributes`.

    Returns:
        attribute name -> percentage (0-100) of rows where it was imputed.
    """
    if not rows:
        return {}
    rates: dict[str, float] = {}
    for attr in IMPUTATION_POLICY:
        flag = f"{attr}_imputed"
        imputed_count = sum(1 for row in rows if row.get(flag))
        rates[attr] = round(100.0 * imputed_count / len(rows), 2)
    return rates
