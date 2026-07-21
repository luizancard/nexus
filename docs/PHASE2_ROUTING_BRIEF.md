# NEXUS Phase 2 — Accessible Routing + OSM-vs-Imagery Scientific Comparison (planning brief)

You are being brought in to plan and then build **Phase 2** of NEXUS. Read
this whole brief, then read the referenced sections of `docs/METHODOLOGY.md`
before proposing anything.

## Why this project exists

NEXUS is a research-competition project (32º Prêmio Jovem Cientista, "AI &
Community" track, final delivery **2026-07-31 — about 10 days out**),
building an accessible pedestrian routing system for people with reduced
mobility, piloted in the **Lourdes** neighborhood of Belo Horizonte, Brazil.
Repo: `/Users/luizancard/Developer/nexus_project` (Python `.venv` at root;
run modules as `python -m ...` from the repo root).

Phase 1 — the full data pipeline (Mapillary imagery → segmentation → geometric
heuristics; OSM tags; DEM/slope; fusion into one per-edge schema) — is
**complete, audited three times** (two adversarial passes plus one
independent cross-model audit), then **hand-validated edge-by-edge**. It is
documented exhaustively in `docs/METHODOLOGY.md`. Your phase consumes that
data and runs the experiment that produces the paper's actual finding.

## The scientific goal — this is what everything serves

Not "a router that runs." The finding the article needs:

> **Does adding AI/imagery-derived accessibility attributes to OSM produce
> measurably more accessible routes for mobility-impaired users than OSM
> alone — by how much, and how much of that effect survives the data's known,
> measured imperfection?**

A modest or null result, honestly measured, is a valid finding. The goal is a
**defensible scientific conclusion** that adds something real to the
literature — not a soulless demo. The literature usually assumes crowdsourced
accessibility data is either complete or uniformly sparse; an honestly-bounded,
imagery-augmented routing comparison on a real neighborhood is the
contribution. Rigor and honesty are the product.

## Read first, in order

1. `docs/METHODOLOGY.md` **§11** — the routing-experiment specification. This
   is your primary spec; implement it.
2. **§6** — the OSM-vs-imagery coverage table your experiment operationalizes.
3. **§9.4 and §9.5** — the per-edge validation results and the go/no-go
   verdict: exactly what each attribute can be trusted for.
4. **§10** — the claims the article may make and the caveats it MUST state.
5. **§4** — the 3-category imputation policy (absent / unknown / middle-tier).
6. `README.md` "Project Architecture" — intended module boundaries.

## Scope — build these (currently 0-byte stubs)

- `core/impedance_model.py` — the cost function.
- `core/routing_algorithms/` (`dijkstra.py`, `d_star_lite.py` exist as stubs)
  — the router(s). A correct Dijkstra over the impedance graph is enough for
  the thesis; a second algorithm is optional polish.
- `core/graph_manager.py` — graph loading/management.
- `evaluation/metrics_calculator.py`, `evaluation/visualizer.py` — the
  experiment metrics and the "heat map of Lourdes" deliverable.

**Do NOT** modify `data_pipeline/` or re-run expensive pipeline stages — the
fused data is final. **Do NOT** re-introduce the Vistas `Barrier` class as an
obstacle source (removed with evidence in §9.4; it was 3% precision).

## The data contract — read carefully, these are load-bearing

- **Load with `edge_attribute_fusion.load_fused_graph(path)`, NEVER
  `carregar_grafo`.** The latter leaves every `_imputed`/`_validated` flag as
  the *string* `'False'`, which is truthy in Python — silently inverting every
  flag (this was real bug #10; see §7). This single mistake would corrupt the
  whole experiment.
- `data_files/lourdes_graph_fused.graphml` — 278 nodes, 832 edges. Every edge
  carries all 9 canonical attributes, each with an `<attr>_imputed` flag, an
  `<attr>_source` (`osm`/`imagery`), plus `length` and `grade`/`grade_abs`.
- `data_files/lourdes_graph_validated.graphml` — the same graph plus
  `<attr>_validated` / `<attr>_validated_value` for the hand-validated subset:
  **260 `ramp_present` edges** (236 present / 24 absent) and **294
  `fixed_obstacle_present` edges** (120 present / 174 absent). Regenerate via
  `python -m scripts.edge_validation apply`.
- **`steps_present` and `fixed_obstacle_present` are 3-STATE**
  (`present`/`absent`/`unknown`). Never write `if present / else` — `unknown`
  must be handled explicitly. Over half the graph is `unknown` on these;
  treating `unknown` as "safe" is the exact failure §4 exists to prevent
  (routing someone into an unconfirmed staircase).
- **Per-attribute trust levels (from §9.4), design the cost model around these:**
  - `ramp_present`: hand-validated **90.8%** precision — trust it; it is the
    high-leverage imagery attribute.
  - `fixed_obstacle_present`: **64%** after the Barrier fix — use as a **soft
    penalty**, not a hard block (a false positive over-avoids a clear path;
    that must be recoverable, not fatal).
  - `steps_present`: 8 OSM-derived edges are reliable; strong penalty / avoid
    for wheelchair profiles; `unknown` stays `unknown`.
  - `surface_material_tier`, `grade`/slope: cost multipliers.
  - `width_bucket`, `smoothness_tier`, `handrail_present`,
    `tactile_paving_present`: mostly or fully imputed — include if useful but
    do not let a fully-imputed attribute dominate a routing decision; report
    their imputation rates (§4, §6) so their weight is honest.

## The experiment — operationalize §11

**Two conditions, same router, same graph** (this isolation is what lets you
attribute any difference to the imagery):

- **Baseline:** OSM + DEM only, no imagery. Reproducible today by running the
  fusion CLI with an empty predictions file (`echo '[]' > empty.json`),
  already verified to reproduce the "OSM alone" column of §6.
- **Full:** the fused graph with imagery.

**Ground truth:** for the **headline** result, route on the **validated
subset** — use `<attr>_validated_value` where present; treat the `uncertain`
segments (no validated flag) as **imputed** (the pessimistic default), never
as confirmed features. Then run a **secondary** condition on the raw
(post-Barrier) imagery. **The gap between the validated and raw runs is your
measurement of how much detection error affects the conclusion** — this run is
non-negotiable; it is what makes the finding measured rather than assumed.

**Cost model** (`Custo = Distância × Fator_superfície + Penalidade_obstáculo`,
plus a slope/grade term): map attributes to terms per the trust levels above.
State every weight and justify it; do not hand-tune weights to produce a
desired result.

**Metrics** (`evaluation/`), over a sample of origin-destination pairs:
1. What fraction of routes **differ** between baseline and full.
2. Among differing routes, how many **avoid a real (validated) obstacle or a
   steeper ramp** the OSM-only route would have taken — i.e. the difference is
   an *improvement* for a mobility-impaired user, not just a change.
3. How the **cost distributions** compare between conditions.
Report the §9.4 precision numbers alongside every result.

**OD sampling:** use real, stated OD pairs (key origins — a metro station, a
hospital, a plaza — to a spread of destinations) or a random sample of node
pairs stratified by straight-line distance. State the sampling rule; never
cherry-pick pairs that flatter the result.

## Rigor standard — this is the project's entire identity

Every claim must be confirmed by **executing real code against the real data**,
never by "the code looks correct." No unverified numbers, no invented results,
no silently-skipped scope, and be explicit about every limitation. The graph is
already known routable (single connected component; 50/50 random OD pairs solve
under plain Dijkstra) — sanity-check your own router the same way. **If the
result is null or modest, report it honestly** — that is still a valid finding
and far better than an overstated one.

## Article framing you must preserve (§10)

The paper must state plainly: this is **not** field-surveyed ground truth (no
imagery-only method can be); **26% (105/399) of imagery-flagged segments are
unconfirmable** from the data source; validation is cross-model visual
adjudication of thumbnails, not a field survey. Credibility comes from
reporting uncertainty, not hiding it. Every routing effect size must be
reported **with its precision bound**.

## Deliverables & working style

1. **First, produce a detailed implementation + experiment PLAN for my
   approval** (use plan mode): module design, the cost model with justified
   weights, the experiment procedure, the metrics, and a realistic sequence
   that fits ~10 days. Do not start building until I approve the plan.
2. **After approval:** implement the router + experiment, run it against the
   real data, and write the results into `docs/METHODOLOGY.md` as a new
   **§12**, matching the rigor and honesty of §7–§11 (what you did, the real
   numbers, what's uncertain, what you verified vs. merely assume).
3. The `evaluation/` outputs (metrics + the Lourdes heat map) the article needs.

## Constraints

- ~10 days to final delivery (2026-07-31). Prioritize the **correct, honestly
  measured comparison** over a sophisticated router — the comparison is the
  thesis, the router is the instrument.
- Ask before any destructive or irreversible action. Commit only when asked.
