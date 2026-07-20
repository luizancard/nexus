# NEXUS Methodology

Living implementation log for the accessibility data pipeline, written to be
lifted directly into the project's scientific article (32º Prêmio Jovem
Cientista). Updated as each stage is built and run against real data, not
written retroactively. Every number below came from an actual run against
the real Lourdes dataset, not an estimate.

## Scope and sequencing

The project's cost model (`Custo = Distância × Fator_superfície +
Penalidade_obstáculo`) needs an accessibility signal per street segment that
neither of its two source datasets provides alone: OpenStreetMap (OSM) tags
are official but sparse, and street-level imagery (Mapillary) is dense but
requires a computer-vision pipeline to extract structured attributes from.
This document covers building both sources and fusing them, in that order,
followed by the digital elevation model (DEM) pipeline that both depend on
for slope-related attributes. The routing algorithm itself
(`core/impedance_model.py`, `core/routing_algorithms/`) consumes this
pipeline's output but is a separate, later phase.

## 1. A pre-existing data bug: the DEM was empty

Before any new work, `data_files/lourdes_dem_1m.tif` (the elevation raster,
believed complete going into this phase) was found to contain **zero real
data**: all 277,168,816 pixels held the exact same value, 0.0. `validate_dem.py`
already had the correct check in place and printed `ALERTA: A altitude média
está fora do esperado` on this file -- the bug was real, not a gap in
validation.

Root cause: the source contour shapefile (`data_files/curvas_lourdes_Isolado.shp`)
is genuine, high-quality data -- 131 contour lines, elevation attribute
column `COTA_CURVA`, range 825–932m, consistent with Belo Horizonte's known
elevation (the file's own `.prj` confirms SIRGAS 2000 / UTM zone 23S,
EPSG:31983). The external process that had generated the `.tif` from this
shapefile (git history: "created the .TIF ... using cKDTree") most likely
queried a column named `Z`, which does not exist in this shapefile -- the
real elevation field is `COTA_CURVA` -- causing every interpolated point to
silently default to 0.

Fix: `scripts/generate_dem_from_contours.py`, a from-scratch contour-to-raster
interpolation using only tools already in the project (`geopandas`, `scipy`,
`rasterio` -- no new heavy GIS dependency):
1. Clip contour lines to the Lourdes neighborhood + 400m buffer (of 131 total
   lines, 108 fall in this area; the full shapefile's bounding box is much
   larger than the neighborhood, evidently other municipal contour coverage under the same file).
2. Densify each line into a point cloud at 2m spacing along its length
   (172,373 points), each carrying its line's real `COTA_CURVA` elevation.
3. Interpolate onto a regular 1m grid via `scipy.interpolate.griddata`
   (linear, with nearest-neighbor fill for small gaps inside the convex hull).
4. Mask any pixel farther than 150m from the nearest real contour point as
   proper `NoData` (-9999, declared in the GeoTIFF header) rather than a
   silent, misleading interpolated guess.

Result: 2,562 × 2,073 pixels (right-sized to the neighborhood, vs. the
previous file's 15,209 × 18,224 -- most of which was outside any area this
project needs), 85.5% valid coverage, elevation range 838.0–928.0m, mean
875.3m. `validate_dem.py` now reports `SUCESSO`. Every downstream
elevation/slope computation in `osm_extractor.py` was silently wrong before
this fix (e.g. one sample edge's grade read 0.00% before, 0.95% after) --
worth stating plainly in the article as a caught-and-corrected error, not
glossed over.

## 2. OSM accessibility tag extraction

`data_pipeline/osm_extractor.py`'s pedestrian-graph download already existed
but only retained OSMnx's default tag set (routing-topology tags, not
accessibility ones). Extended `ox.settings.useful_tags_way` /
`useful_tags_node` to also retain: `surface`, `smoothness`, `width`, `kerb`,
`tactile_paving`, `incline`, `lit`, `handrail`, `ramp`, `step_count`,
`wheelchair`, `barrier`, `footway`, `crossing`.

Measured tag coverage across Lourdes's 278-node, 832-edge pedestrian graph
(`diagnosticar_cobertura_tags`, run 2026-07-20):

| Tag | Coverage |
|---|---|
| `highway` | 100.0% |
| `surface` | 58.2% |
| `footway` | 9.6% |
| `tactile_paving` | 1.9% |
| `smoothness`, `lit` | 0.5% each |
| `width`, `kerb`, `incline`, `handrail`, `ramp`, `step_count`, `wheelchair`, `barrier` | 0.0% |
| `highway=steps` count | 0 edges |

Two things worth noting for the article: (1) `surface` coverage (58.2%) is
substantially better than the near-total sparsity commonly assumed for
crowdsourced OSM data in this literature (see Neis & Zielstra 2014) -- worth
citing as a positive, area-specific finding rather than assuming the worst
case applies uniformly. (2) Every other descriptive tag is at or near 0%,
which is exactly the gap the Mapillary imagery pipeline exists to fill, and
`highway=steps` returning zero matches means OSM currently offers *no*
signal on stairs for this neighborhood -- the segmentation pipeline (and,
longer-term, the stretch-goal fine-tune) is the only source for that
specific hazard here, not a redundant second opinion.

The pedestrian graph itself is also never persisted anywhere before this
work (`injetar_topografia_e_calcular_esforco`'s output only existed inside a
`__main__` demo). Added `salvar_grafo`/`carregar_grafo` (GraphML) so it's a
reusable artifact.

### Weekly refresh

OSM is community-edited, and the project's "VGI feedback loop" concept only
means something if edits are actually re-absorbed over time. `data_pipeline/refresh.py`
wraps extraction + DEM fusion + tag diagnostics into one idempotent
`refresh_lourdes_graph()` call: each run writes a dated snapshot
(`data_files/graph_snapshots/lourdes_graph_YYYY-MM-DD.graphml`), updates a
stable `lourdes_graph_latest.graphml` pointer, and diffs the new snapshot
against the previous one edge-by-edge (matching by OSM's stable (u, v, key)
identity), logging every tag change to `data_files/graph_snapshots/changelog.md`.
First run: "Initial snapshot: 832 edges, nothing to compare against." Second
run (same day, no real-world edits in between): 0 changes, as expected --
confirms the diff logic is precise, not just always reporting activity.
Actual weekly scheduling (cron / the `/schedule` skill) is intentionally a
separate, explicit follow-up action, not silently wired up as a side effect
of writing this script.

## 3. Mapillary imagery pipeline

### 3.1 Data acquisition

`data_pipeline/mapillary_client.py`, built against the live Mapillary Graph
API v4. Two undocumented-until-tested constraints shaped the design:

- The API's documented `bbox` limit (< 0.01° square) forced tiling Lourdes's
  ~1.27km × 1.76km extent into a grid (16 tiles at 0.004° cells).
- More surprising: **tile density affects total results even for tiles well
  under the documented 2,000-result cap.** A coarse 4-tile fetch (2 tiles
  near the cap) returned 6,579 unique images; the same area at 16 tiles
  returned 22,943; at 56 tiles, 39,035 and still climbing. The API's `limit`
  parameter is not simply "return everything up to N" -- something in
  Mapillary's bbox search appears to sample/thin results in a way that
  scales with query area, not just result count. Chosen operating point:
  16 tiles (0.004°), 22,943 raw images -- a pragmatic balance, not a proven
  completeness guarantee; worth stating as a limitation rather than implying
  exhaustive coverage.

Raw image count (22,943) is dominated by near-duplicate frames from repeated
drive/walk-throughs of the same streets -- expected for a well-mapped urban
area, but impractical to run a 216M-parameter segmentation model against
directly within the project timeline. Added a representative-downselection
step: bin images into a 20m × 20m grid (in the DEM's projected CRS, so cell
size is true meters) crossed with a 4-way compass quadrant, keep the most
recent image per (cell, quadrant). Result: 22,943 → 3,697 images (6.2×
reduction) while preserving spatial coverage and multiple viewing angles per
location.

### 3.2 Segmentation model

Used **`facebook/mask2former-swin-large-mapillary-vistas-semantic`**
(HuggingFace, verified live: ~216M params, `transformers`-native, Mapillary
Vistas v1.2's real 65-class taxonomy read directly from the checkpoint's
`config.json`) for inference -- no training, consistent with the project's
timeline (delivery 2026-07-31) and the "hybrid" strategy decision (pretrained
now, YOLO26 fine-tune later as a scoped stretch goal). Runs locally via
Apple Silicon MPS acceleration (~1.3–2 img/s), no cloud GPU needed for this
stage.

Two load-time warnings needed verification, not just dismissal, given the
project's own standard for "no black-box code":
- `UNEXPECTED` keys for `relative_position_index` in every Swin attention
  block: standard, well-documented Swin Transformer behavior -- these are
  deterministic non-trainable buffers regenerated at init, routinely
  excluded from saved checkpoints.
- `MISSING` key for `swin.layernorm.{weight,bias}`: verified against
  `transformers`' own source (`modeling_swin.py`,
  `SwinBackbone._keys_to_ignore_on_load_missing = [r"swin.layernorm.*"]`) --
  the backbone-mode Swin uses per-stage layer norms for multi-scale
  features and never calls this particular final layernorm; the library
  itself declares this specific key expected-missing. Confirmed not a sign
  of degraded weights.

Class coverage, checked against the real 65-class list: **8 of 11**
requested obstacle categories map directly (street, sidewalk, curb,
ramp/curb-cut, crossing, lighting, surface obstacles, general road surface).
**3 have no equivalent in any available pretrained checkpoint anywhere** --
verified, not assumed: `steps`, `handrail`, `tactile_paving`. These stay
genuinely undetected (no fabricated negative) until the stretch-goal
fine-tune, which can now be scoped narrowly to just these 3 classes instead
of the original full 10-class plan.

The existing `MAPILLARY_TO_ACCESSIBILITY` label map (written before the real
Vistas class list was available) used placeholder-style keys (`"curb-cut"`,
`"crosswalk-zebra"`) that do not match Vistas' actual names (`"Curb Cut"`,
`"Crosswalk - Plain"`). Corrected using the verified class list, and the
label-normalization function was rewritten from a naive `.replace(" ",
"-")` (which leaves broken multi-hyphen keys on any class name containing
its own punctuation, e.g. `"Crosswalk - Plain"` -> `"crosswalk---plain"`) to
a proper regex normalization. Also removed the original map's
`"guard-rail" -> "handrail"` entry: Vistas' Guard Rail class is a roadside
vehicle barrier, not pedestrian stair/ramp support, and the two are visually
and functionally distinct enough that conflating them would have fabricated
a false accessibility-positive signal on ordinary street guardrails, which
are extremely common in this terrain.

A real bug was caught by inspecting actual output rather than trusting the
pipeline by construction: per-pixel confidence, computed by replicating
HF's internal query-fusion (softmax class probabilities × sigmoid mask
probabilities, summed over queries -- the same computation
`post_process_semantic_segmentation` uses internally to pick the winning
class, which the convenience function does not expose), can exceed 1.0 when
multiple queries agree on the same pixel (observed up to 1.34 in early
testing). Since `summarize_barriers_per_image` multiplies this value
directly against `BARRIER_WEIGHTS` and assumes a [0, 1] range, values above
1.0 would silently overweight some detections. Fixed with an explicit clamp.

Visual QA (`overlay_*.png` renders, matplotlib polygons over source images,
3 sample images) confirmed sidewalk/curb/pole masks align accurately with
real features. One incidental finding worth keeping: several sample images
carry a "BLACKVUE DR900M" watermark -- i.e. sourced from car dashcams, not
pedestrian phones -- which have a different mounting height and pitch than
the pedestrian-capture assumption the width heuristic uses (see §3.3). This
is concrete evidence for a limitation that would otherwise have been
speculative.

### 3.3 Geometric attribute heuristics

No monocular depth model (per project decision, categorical heuristics only,
matching the cost formula's own coarse thresholds like the <50cm rule).
`data_pipeline/geometric_attribute_extractor.py`:

- **Sidewalk width**: pinhole ground-plane projection from a *stated, fixed*
  assumption -- camera height 1.5m, vertical field of view 65° -- rather
  than Mapillary's real per-image `camera_parameters` (not fetched; varies
  by contributor device and was judged not worth the added complexity given
  the project's chosen scope). Bucketed into `<50cm / 50-90cm / >90cm`.
- **Ramp/curb-cut declivity**: *not* estimated from images at all. Samples a
  slope-percent raster (`generate_slope_raster`, a finite-difference
  gradient over the real DEM from §1) directly at the ramp's coordinate.
  This is finer-grained than the existing node-to-node edge `grade` --
  important because a short, steep curb ramp inside an otherwise-flat
  street segment is invisible to that coarse edge-level average. Classified
  against NBR 9050's general ramp-slope guidance (~8.33% max for longer
  runs).
- **Curb type/curvature**: deliberately *not* a separate geometric
  computation. Mapillary Vistas already distinguishes "Curb" (raised) from
  "Curb Cut" (the beveled accessibility ramp) as distinct trained classes --
  re-deriving "sharp vs. beveled" from contour geometry would be a strictly
  less reliable proxy for information the classifier already provides
  directly.
- **Step height, camber**: explicitly out of scope. Step height cannot be
  estimated for a class with no detector (§3.2); camber needs
  cross-sectional 3D reconstruction that monocular heuristics handle poorly.

Slope raster generation surfaced its own data-quality issue, same family as
§1: `np.gradient` over the interpolated DEM produced a small number of
physically-implausible extreme values (max observed: 2,418%) at boundaries
between the linear-interpolated region and the nearest-neighbor hull-gap
fill. Distribution check showed this is a small tail, not a systemic
problem (median 4.85%, 95th percentile 30.94%, only 0.44% of pixels exceed
100%) -- added a 150% sanity bound (same defensive pattern `osm_extractor.py`
already uses for implausible elevation values), consistent with "no real
Lourdes street approaches a 150% grade."

### 3.4 Validation

No physical field access is available for independently-measured ground
truth. Implemented `scripts/validate_geometric_heuristics.py` as an honest
substitute: a **cross-method agreement check**, not a gold-standard
comparison. For every image with both a Curb and Sidewalk detected, width is
estimated two independent ways -- the primary pinhole-projection method, and
a second method using the detected curb's own pixel height as a local scale
reference (assuming a standard ~15cm curb height per NBR 9050). Agreement
between the two is evidence of real signal; disagreement quantifies the
primary method's fixed-camera assumption cost directly, rather than leaving
it a guess.

**Full-sample result** (3,579 images with both methods producing an
estimate, run 2026-07-20): **57.8% agreement** on the 3-way width bucket.
This is meaningfully above the chance-agreement baseline implied by each
method's own marginal distribution (~38.2%, computed from the confusion
matrix's row/column totals) -- Cohen's kappa ≈ 0.32, conventionally "fair"
agreement: real signal, not noise, but far from precise. The confusion
matrix shows two specific weaknesses worth stating plainly rather than
averaging away: (1) the middle bucket (50-90cm) is the least reliable for
both methods -- of 646 images the primary method placed there, only 118
(18%) were confirmed by the curb-reference method; (2) a meaningful
minority of comparisons (463/3,579, ~13%) show *complete* disagreement
between opposite extremes (one method says under 50cm, the other over
90cm), which is a more serious failure mode than adjacent-bucket
disagreement.

One further caveat for the article: even perfect agreement between these
two methods would not by itself prove absolute accuracy against physical
ground truth, since both share the same class of assumption -- a fixed,
stated real-world reference size (assumed camera height/FOV for the
primary method, assumed curb height for the reference method) rather than
a measured one. Agreement mainly rules out gross methodological errors
(e.g. a sign error or wildly wrong FOV assumption), not fine calibration
error in either shared assumption.

**Decision gate, as planned**: agreement is above chance, so the width
bucket is kept as a 3-way graded attribute rather than collapsed to binary
presence/absence -- but given the specific failure modes above, it should
be treated as a lower-confidence attribute in any future cost function
(e.g. weighted down, or re-validated against field measurements or real
per-image camera calibration) rather than trusted at the same level as
directly-tagged OSM attributes or high-confidence class detections like
Curb Cut presence.

## 4. Imputation policy

The README and research proposal both name a "Pessimistic Safe Fallback"
heuristic conceptually but never define it precisely enough to implement.
Implementing it directly surfaced a real tension the one-line description
glosses over -- "missing data gets the conservative default" is not one
rule, it is (at least) three:

1. **Binary infrastructure presence** (tactile paving, handrail, ramp,
   marked crossing): rare, specific, positive investments. If neither OSM
   nor imagery shows one, "probably absent" is a fair inference. Default:
   absent.
2. **Binary hazard presence** (steps, fixed obstacles): both extremes are
   actively wrong here. Defaulting to absent risks routing someone into a
   real staircase -- the exact failure this project exists to prevent.
   Defaulting to present would make most of an under-mapped graph look
   artificially impassable. Kept as a genuine third state, `unknown`, not
   collapsed into either extreme.
3. **Graded/continuous quality** (surface material, smoothness, sidewalk
   width): every real street has *some* actual value; "unknown" is a
   measurement gap, not evidence of the worst case. Given OSM coverage for
   these is empirically ~0% in Lourdes (§2), defaulting to the worst tier
   would mark nearly the entire graph as critically narrow/poor-surface --
   a fabricated claim, not caution. Default: the *middle* tier.

`data_pipeline/imputation_engine.py` implements this as a policy table plus
one pure function, deliberately *not* deciding how much an imputed value
should cost -- that stays the future impedance model's job, keeping this
schema independent of whatever the eventual cost function turns out to be.
Every imputed attribute carries a companion `<attr>_imputed` flag, so
`summarize_imputation_rate` can report exactly what fraction of the graph is
measured vs. assumed, per attribute, rather than leaving that an implicit
and unverified claim.

## 5. Fusion schema

`data_pipeline/edge_attribute_fusion.py` produces the final per-edge
attribute table: OSM tags (converted to the canonical schema via
`osm_tags_to_canonical`) take priority where present; Mapillary imagery
(spatially joined via `osmnx.distance.nearest_edges`, images snapped to
their nearest edge after reprojection into the graph's own CRS) fills gaps;
`imputation_engine` resolves whatever remains. Every edge additionally
records `<attr>_source` (`"osm"` or `"imagery"`) and `n_images_observed`,
so downstream consumers -- including this document -- can audit exactly
where any given value came from rather than treating the fused graph as an
opaque ground truth.

Output: a persisted `.graphml` (`data_files/lourdes_graph_fused.graphml`,
ready for `core/impedance_model.py`) and a flat GeoDataFrame/CSV export
(`data_files/lourdes_edges_fused.csv` -- feeds the research proposal's
promised "heat map of Lourdes" deliverable, and general inspection/debugging).

**Full run result** (2026-07-20, all 832 Lourdes edges, all 3,697 manifest
images): **432/832 edges (52%)** have at least one Mapillary image directly
snapped to them. Per-attribute imputation rate on the final fused graph --
i.e. the fraction of edges where *neither* OSM nor imagery contributed
evidence, so the pessimistic default from §4 was used:

| Attribute | Imputed | Resolved from real data |
|---|---|---|
| `handrail_present` | 100.0% | 0% -- confirmed total gap (§3.2 + §2's 0% OSM handrail coverage) |
| `smoothness_tier` | 99.5% | 0.5% -- matches OSM smoothness coverage exactly |
| `tactile_paving_present` | 98.1% | 1.9% -- matches OSM tactile_paving coverage exactly |
| `ramp_present` | 60.0% | 40.0% -- entirely from imagery (OSM `kerb` coverage is 0%) |
| `fixed_obstacle_present` | 60.2% | 39.8% -- entirely from imagery |
| `marked_crossing_present` | 62.9% | 37.1% -- mostly from imagery |
| `steps_present` | 56.0% | 44.0% -- from OSM `highway` values that positively rule out steps |
| `width_bucket` | 48.6% | 51.4% -- entirely from imagery (OSM `width` coverage is 0%) |
| `surface_material_tier` | 41.8% | 58.2% -- matches OSM surface coverage exactly |

The exact matches between resolved-rate and previously-measured OSM tag
coverage (surface, smoothness, tactile_paving) are a useful internal
consistency check on the fusion logic itself: those three attributes have
no current imagery-derived source in `CLASS_TO_PRESENCE_ATTR`, so 100% of
their real (non-imputed) values should trace back to OSM alone, which is
exactly what the numbers show. `ramp_present`, `fixed_obstacle_present`,
and `width_bucket`, conversely, have 0% OSM tag coverage for their
corresponding raw tags (`kerb`, `barrier`, `width`) -- so their entire
non-imputed fraction is attributable to the Mapillary pipeline, which is
the clearest quantified demonstration in this document of why the imagery
pipeline is necessary rather than redundant with OSM.

## Open items

- Stretch goal (YOLO26 fine-tune, `steps`/`handrail`/`tactile_paving` only):
  not started. Cost estimate from planning: ~$2-8 on Azure/GCP T4 spot
  pricing for a full 100-epoch run; a 3-class focused run should land at or
  below that.
- Mapillary's bbox search non-convergence (§3.1) is based on this project's
  own testing, not documented API behavior -- worth independent replication
  before citing as a general claim in the final article.
- Width bucket carries known, quantified uncertainty (§3.4) -- treat as
  lower-confidence than directly-tagged attributes in any future cost
  function design.
- This pipeline's output (`lourdes_graph_fused.graphml`) is now ready for
  the next phase: `core/impedance_model.py` and the routing algorithms,
  which are out of scope for this document.
