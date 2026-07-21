# NEXUS Methodology

Living implementation log for the accessibility data pipeline, written to be
lifted directly into the project's scientific article (32º Prêmio Jovem
Cientista). Updated as each stage is built and run against real data, not
written retroactively. Every number below came from an actual run against
the real Lourdes dataset, not an estimate -- including several corrections
made after independent, adversarial re-verification of earlier drafts of
this very document (see Section 7).

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
pipeline's output but is a separate, later phase -- see Section 6 for why
that phase, not more data work, is what the project's scientific thesis
actually depends on next.

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
(`diagnosticar_cobertura_tags`):

| Tag | Coverage |
|---|---|
| `highway` | 100.0% |
| `surface` | 58.2% |
| `footway` | 9.6% |
| `tactile_paving` (edge) | 1.9% -- rises to 12.7% once node-level tags are also checked, see below |
| `smoothness`, `lit` | 0.5% each |
| `width`, `kerb`, `incline`, `handrail`, `ramp`, `step_count`, `wheelchair`, `barrier` | 0.0% |
| `highway=steps` count | **8 edges** -- see correction below |

Two things worth noting for the article: (1) `surface` coverage (58.2%) is
substantially better than the near-total sparsity commonly assumed for
crowdsourced OSM data in this literature (see Neis & Zielstra 2014) -- worth
citing as a positive, area-specific finding rather than assuming the worst
case applies uniformly. (2) Most other descriptive tags are at or near 0%,
which is exactly the gap the Mapillary imagery pipeline exists to fill.

**Correction, found by an independent audit (Section 7): `highway=steps`
was not actually zero.** An earlier draft of this document claimed OSM had
"no signal on stairs for this neighborhood." That was wrong, and the error
was a code bug, not a data gap. OSMnx's `simplify=True` (used in
`extrair_malha_pedestres`) merges consecutive original way segments into
one edge; when merged segments disagree on a tag, OSMnx stores it as a
**list**, not a scalar string. Both `diagnosticar_cobertura_tags` and
`edge_attribute_fusion.osm_tags_to_canonical` compared `highway` with plain
`==`/`in` against string literals -- which a list never equals or is
"in", so both checks silently missed every one of 8 real edges in the
Lourdes graph carrying `highway=['steps', 'footway']`-style tags. There
genuinely are staircases mapped in OSM here; the code just couldn't see
them. Fixed with `osm_extractor.highway_values()`, a small normalizer used
everywhere a `highway` value is compared, and re-verified: `highway=steps`
count is now correctly 8, and those 8 edges now resolve `steps_present`
from OSM instead of falling through to imputation. This is the single most
consequential bug found in this project to date, given steps/stairs are
the project's own paradigmatic example of a routing hazard.

The pedestrian graph itself is also never persisted anywhere before this
work (`injetar_topografia_e_calcular_esforco`'s output only existed inside a
`__main__` demo). Added `salvar_grafo`/`carregar_grafo` (GraphML) so it's a
reusable artifact.

### Weekly refresh

OSM is community-edited, and the project's "VGI feedback loop" concept only
means something if edits are actually re-absorbed over time.
`data_pipeline/refresh.py` wraps extraction + DEM fusion + tag diagnostics
into one idempotent `refresh_lourdes_graph()` call: each run writes a dated
snapshot (`data_files/graph_snapshots/lourdes_graph_YYYY-MM-DD.graphml`),
updates a stable `lourdes_graph_latest.graphml` pointer, and diffs the new
snapshot against the previous one edge-by-edge (matching by OSM's stable
`(u, v, key)` identity), logging every tag change to
`data_files/graph_snapshots/changelog.md`. The `old is None` first-run path
(logging `"Initial snapshot: N edges, nothing to compare against."`) was
verified directly via `diff_snapshots(None, graph)`, but the actual
committed `changelog.md` only shows a same-day, zero-change second entry
(`"No accessibility tag changes since previous snapshot."`) -- because a
`lourdes_graph_latest.graphml` already existed from an earlier manual run
before `refresh.py` was first exercised end to end, so the true first-run
path never got a chance to write its own line to this particular file. The
code path is verified correct; the changelog's contents just reflect the
order things were actually run in, worth being precise about rather than
implying the file contains something it doesn't. Actual weekly scheduling
(cron / the `/schedule` skill) remains an explicit follow-up action, not
silently wired up as a side effect of writing this script.

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
  exhaustive coverage. (This finding is based on this project's own testing,
  not documented Mapillary behavior -- worth independent replication before
  citing as a general claim.) `DEFAULT_TILE_DEGREES` in the shipped code is
  set to this validated 0.004° operating point specifically so a future
  re-run doesn't silently regress to the coarser, under-collecting default
  that an earlier draft of the code shipped with.

Raw image count (22,943) is dominated by near-duplicate frames from repeated
drive/walk-throughs of the same streets -- expected for a well-mapped urban
area, but impractical to run a 216M-parameter segmentation model against
directly within the project timeline. Added a representative-downselection
step: bin images into a 20m × 20m grid (in the DEM's projected CRS, so cell
size is true meters) crossed with a 4-way compass quadrant, keep the most
recent image per (cell, quadrant). Result: 22,943 → 3,697 images (6.2×
reduction) while preserving spatial coverage and multiple viewing angles per
location. Of these, 3,689 produced at least one prediction after inference
(8 images failed or had zero relevant detections).

### 3.2 Segmentation model

Used **`facebook/mask2former-swin-large-mapillary-vistas-semantic`**
(HuggingFace, verified live: ~216M params, `transformers`-native, Mapillary
Vistas v1.2's real 65-class taxonomy read directly from the checkpoint's
`config.json`) for inference -- no training, consistent with the project's
timeline (delivery 2026-07-31) and the "hybrid" strategy decision (pretrained
now, YOLO26 fine-tune later as a scoped stretch goal). Runs locally via
Apple Silicon MPS acceleration (~0.8-2 img/s depending on system load), no
cloud GPU needed for this stage.

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

Class coverage, checked against the real 65-class list and against this
project's own 10-class `ACCESSIBILITY_CLASSES` taxonomy: **7 of 10**
requested obstacle categories map directly (street, sidewalk, curb,
ramp/curb-cut, crossing, lighting, surface obstacles). **3 have no
equivalent in any available pretrained checkpoint anywhere** -- verified,
not assumed: `steps`, `handrail`, `tactile_paving`. These stay genuinely
undetected by imagery (no fabricated negative) until the stretch-goal
fine-tune, which can now be scoped narrowly to just these 3 classes instead
of the original full 10-class plan. (Note: `steps` now has a real, working
OSM-derived source instead -- see Section 2's correction -- so imagery's gap
here is now partially, not wholly, uncovered.)

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
pipeline by construction: per-pixel confidence, computed via the same
query-fusion HF's `post_process_semantic_segmentation` uses internally
(softmax class probabilities × sigmoid mask probabilities, summed over
queries -- the convenience function computes this but doesn't expose it),
can exceed 1.0 when multiple queries agree on the same pixel (observed up
to 1.34 in early testing). Since `summarize_barriers_per_image` multiplies
this value directly against `BARRIER_WEIGHTS` and assumes a [0, 1] range,
values above 1.0 would silently overweight some detections. Fixed with an
explicit clamp -- confirmed in the full delivered dataset, all 63,843
detections in `lourdes_predictions.json` have confidence in [0.0003, 1.0].

Separately, this computation was later found (Section 7's audit) to not be
a byte-identical replication of HF's method after all: HF interpolates mask
logits to a fixed 384x384 size *before* the query-fusion step, then
interpolates again after; this code fuses first at native resolution and
interpolates once. Compared directly against HF's own method on a real
image, this disagrees on 1.38% of pixels, concentrated at class-transition
boundaries (80.8% of disagreements are on a boundary, vs. boundaries being
only 4.1% of the image). Not fixed -- the deviation is small and the
correct-vs-HF question doesn't have an obviously "more correct" answer
given both are reasonable orderings of the same two operations -- but the
code's docstring was corrected to state this precisely rather than claim an
exact replication it doesn't deliver.

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
  the project's chosen scope). Bucketed into `<50cm / 50-90cm / >90cm`. See
  §3.4 for why this estimate is not currently trusted.
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
  estimated for a class with no imagery detector (§3.2); camber needs
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
estimate): **57.8% agreement** on the 3-way width bucket. This is
meaningfully above the chance-agreement baseline implied by each method's
own marginal distribution (~38.2%, computed from the confusion matrix's
row/column totals) -- Cohen's kappa ≈ 0.32, conventionally "fair"
agreement: real signal, not noise, but far from precise. The confusion
matrix shows two specific weaknesses worth stating plainly rather than
averaging away: (1) the middle bucket (50-90cm) is the least reliable for
both methods -- of 646 images the primary method placed there, only 118
(18%) were confirmed by the curb-reference method; (2) opposite-extreme
disagreement (one method says under 50cm, the other over 90cm) happens in
**555/3,579 (15.5%)** of all comparisons in total -- 463 where the primary
method said over_90cm and the reference said under_50cm, plus 92 in the
other direction -- a more serious failure mode than adjacent-bucket
disagreement, and one that doesn't wash out in either direction.

One further caveat for the article: even perfect agreement between these
two methods would not by itself prove absolute accuracy against physical
ground truth, since both share the same class of assumption -- a fixed,
stated real-world reference size (assumed camera height/FOV for the
primary method, assumed curb height for the reference method) rather than
a measured one. Agreement mainly rules out gross methodological errors
(e.g. a sign error or wildly wrong FOV assumption), not fine calibration
error in either shared assumption.

**Decision gate, as planned -- revised after a full fusion spot-check.** The
57.8%-agreement result above was initially read as "fair agreement, keep as
a graded attribute, just lower-confidence." That was an incomplete read:
cross-method agreement measures *consistency* between two methods, not
*calibration* against reality, and two methods can agree with each other
while sharing the same directional bias. A post-fusion check across all 832
edges found exactly that: **90.9% of every real (non-imputed) width
estimate (389/428) landed in the single most extreme bucket, `under_50cm`**
-- including on well-known wide boulevards (Avenida do Contorno, Rua da
Bahia) that are not plausibly sub-50cm almost everywhere. Combined with
dashcam watermarks observed during the §3.2 visual QA (a common
wide-angle-lens source, roughly 140-160deg FOV vs. the 65deg assumed here),
the likely mechanism is direct: an assumed FOV that's too narrow
overestimates the pinhole focal length, which underestimates real-world
size per pixel, which shrinks every width estimate in the same direction,
regardless of which of the two (correlated) methods computes it.

Guessing a corrected FOV without real per-image calibration data would just
swap one unvalidated assumption for another. Instead, the raw estimate is
computed under a deliberately different key,
`width_bucket_uncalibrated_estimate` (not `width_bucket`), specifically so
a future caller can't accidentally merge it into the canonical schema by
using the "obvious" name. The canonical `width_bucket` falls through to OSM
(0% tag coverage for `width`) and then to imputation's neutral default, so
100% of edges honestly report `width_bucket_imputed = True` rather than a
confidently wrong distribution. The raw estimate is preserved (present on
428 edges at the time of this decision; 504 after the §8 audit's re-fusion,
where the bias re-measured essentially unchanged at 91.3% `under_50cm` --
confirming the gate remains warranted) so the recalibration work isn't
lost, just not trusted yet. An
honestly-unknown value is safer to route on than a confident wrong one --
this is the same "Pessimistic Safe Fallback" principle from §4, applied to
a miscalibration failure mode rather than a coverage gap.

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
`osm_tags_to_canonical`, checking both edge-level tags and both endpoint
nodes' tags -- `crossing`/`kerb`/`tactile_paving`/`barrier` are frequently
node-level in OSM, e.g. at the point a footway meets a road) take priority
where present; Mapillary imagery (spatially joined via
`osmnx.distance.nearest_edges`, images snapped to their nearest edge after
reprojection into the graph's own CRS -- since the §8 audit, only when that
edge is within 25m, and mirrored onto both directed twins of the segment)
fills gaps; `imputation_engine` resolves whatever remains. Every edge additionally records `<attr>_source`
(`"osm"` or `"imagery"`) and `n_images_observed`, so downstream consumers --
including this document -- can audit exactly where any given value came
from rather than treating the fused graph as an opaque ground truth.

Output: a persisted `.graphml` (`data_files/lourdes_graph_fused.graphml`,
ready for `core/impedance_model.py`) and a flat GeoDataFrame/CSV export
(`data_files/lourdes_edges_fused.csv` -- feeds the research proposal's
promised "heat map of Lourdes" deliverable, and general inspection/debugging).
Both are reproducible from scratch via `python -m data_pipeline.edge_attribute_fusion`
(a CLI driver added after an audit noted the fused graph previously had no
committed way to regenerate it).

**Full run result, all 832 Lourdes edges**: **508/832 edges (61.1%)** have at
least one Mapillary image snapped to them. (An earlier draft reported
432/832 (52%) -- that number was real but produced by a fusion run that
both leaked evidence from outside-neighborhood imagery onto 81 boundary
edges *and* dropped evidence on the unobserved directed twin of every
observed two-way segment; see §8's findings 1 and 2. The current figure is
after discarding 1,510 far images and mirroring evidence across twins --
net coverage went up because mirroring adds more than the discard removes.)

## 6. OSM vs. imagery: quantifying the core comparison

This is the table the project's thesis actually depends on: not "does the
pipeline run," but "does imagery add real information beyond what OSM
already provides." For each canonical attribute, where does its real
(non-imputed) value actually come from:

| Attribute | OSM alone | Imagery (net) | Combined | Imputed |
|---|---|---|---|---|
| `ramp_present` | 0.0% | 47.1% | 47.1% | 52.9% |
| `fixed_obstacle_present` | 0.0% | 44.5% | 44.5% | 55.5% |
| `marked_crossing_present` | 25.0% | 34.4% | 59.4% | 40.6% |
| `steps_present` | 45.4% | 0.0% | 45.4% | 54.6% |
| `surface_material_tier` | 58.2% | 0.0% | 58.2% | 41.8% |
| `tactile_paving_present` | 12.7% | 0.0% | 12.7% | 87.3% |
| `smoothness_tier` | 0.5% | 0.0% | 0.5% | 99.5% |
| `handrail_present` | 0.0% | 0.0% | 0.0% | 100.0% |
| `width_bucket` | 0.0% | 0.0%* | 0.0% | 100.0% |

(Imagery percentages here are post-§8-audit values -- the pre-audit table
showed 40.0%/39.8%/28.5% for the first three rows, computed from a fusion
run with the two spatial-join defects described in §8. `fixed_obstacle_present`
imagery further dropped from 48.8% to 44.5% in §9.4, when the 3%-precision
Vistas `Barrier` class was removed from the obstacle mapping -- fewer edges,
but far more of them real.)

\* imagery produces a raw estimate for 504 edges, but it is excluded from
the canonical schema -- see §3.4's decision gate.

**Reading this honestly, attribute by attribute, not as one blended
number:**
- **`ramp_present` and `fixed_obstacle_present` are entirely dependent on
  imagery.** OSM's `kerb` and `barrier` tags have exactly 0% coverage in
  this graph. Without the Mapillary pipeline, these two safety-relevant
  attributes would not exist at all for any edge in Lourdes.
- **`marked_crossing_present` more than doubles**: OSM alone covers 25.0%
  of edges (via node-level `crossing` tags, 41/278 nodes); imagery adds a
  net 34.4 percentage points on edges OSM says nothing about. One
  precision the earlier draft got wrong: because OSM takes priority in the
  fusion, the "Imagery (net)" column *by construction* counts only edges
  where OSM was silent, so Combined always equals the sum of the two
  columns -- summing to ~the total is not evidence of non-overlap, as the
  earlier text claimed. Measured independently (§8), imagery actually
  observes crossings on 46.2% of all edges, overlapping OSM on 11.8% --
  genuinely complementary, but the overlap is real, not near-zero.
- **`surface_material_tier`, `smoothness_tier`, `tactile_paving_present`,
  and `steps_present` currently gain nothing from imagery.** The first
  three because no imagery-derived signal for them exists in the current
  pipeline (surface/smoothness would need a texture classifier this project
  doesn't have; tactile paving has no pretrained detector at all -- §3.2).
  `steps_present` is a genuinely interesting case: OSM alone already covers
  45.4% of edges here (after the highway-tag list-bug fix in §2), and
  imagery contributes nothing on top of that today, purely because Vistas
  has no steps class. This is exactly why the stretch-goal fine-tune,
  narrowly scoped to `steps`/`handrail`/`tactile_paving`, is the single
  highest-leverage remaining piece of *data* work -- everything else
  already has some real signal from at least one source.
- **`width_bucket` and `handrail_present` currently have zero real signal
  from either source** -- both are honest, fully-imputed gaps carried
  forward rather than papered over.

**What this table does and does not prove.** It demonstrates, with real
numbers, that Mapillary imagery contains accessibility information that is
genuinely absent from OSM in this specific, under-mapped context -- not
redundant with it. That is a real, defensible, quantifiable claim. It does
**not** by itself demonstrate that this additional information changes
routing outcomes, reduces estimated biomechanical cost, or helps a
mobility-impaired user in practice -- that requires an algorithm that
consumes this data and produces routes, which does not exist yet
(`core/impedance_model.py` and `core/routing_algorithms/` are still empty).

**The concrete next step, and the actual scientific finding this project
needs**, is a controlled comparison: build the cost function and Dijkstra
router in two configurations over the *same* graph -- a baseline using only
OSM + DEM (no imagery), and the full fused version above -- and run both
against a real sample of origin-destination pairs across Lourdes. Measure
(a) what fraction of routes differ between the two conditions, (b) among
those that differ, how many avoid a real obstacle/steeper ramp the
OSM-only version would have missed, and (c) how the estimated cost
distributions compare. The `ramp_present`/`fixed_obstacle_present` rows
above -- 0% OSM, ~47-49% imagery, and both are high-impact terms in the cost
formula (obstacle penalties, not minor tie-breakers) -- are the concrete,
data-level reason to expect this comparison will show a real, non-trivial
effect, not just a hopeful guess. But the actual effect size is unmeasured
until that comparison is built and run; a modest or null result, honestly
measured, would still be a valid finding for this project's purposes,
just not yet the one hoped for.

## 7. Two full-codebase audits (this session)

Beyond ad hoc testing during development, two dedicated correctness passes
were run before considering this phase complete, given how many real bugs
had already turned up from just building the pipeline once.

**First pass**, informal: re-loading the saved fused graph and inspecting
real edges rather than trusting the in-memory fusion result by
construction. Found and fixed:
1. **GraphML round-trip silently stringifies typed attributes.**
   `ox.save_graphml`/`load_graphml` serialize every attribute as a string
   by design and require the caller to declare `edge_dtypes` to restore
   real types on load. `n_images_observed` came back as the string `'9'`
   instead of `int(9)`, and every `<attr>_imputed` flag came back as the
   *string* `'False'` -- truthy in Python (`bool('False') is True`),
   silently inverting every imputation flag for any code trusting the
   saved graph's own type system. Fixed via `edge_attribute_fusion.load_fused_graph()`.
2. The `width_bucket` systematic bias described in §3.4.
3. `osm_tags_to_canonical` only checked edge-level OSM tags, missing
   `crossing`/`kerb`/`tactile_paving`/`barrier` tags that live on nodes.
   Fixed by also checking both endpoint nodes -- this is what corrected
   `marked_crossing_present`'s OSM coverage from an (incorrectly measured)
   0% to the real 25.0%, and `tactile_paving_present` from 1.9% to 12.7%.

**Second pass**, deliberately adversarial: three independent review agents,
each with no visibility into the others' work, tasked respectively with
(a) tracing every function for correctness bugs and verifying suspicions by
executing real code against real data, (b) auditing for dead code,
duplication, and quality issues, and (c) independently re-deriving every
quantified claim in this document against the real data files and flagging
any mismatch. This is what found, beyond the three bugs above:

- **The `highway=steps` list-vs-string bug** (§2) -- the most consequential
  finding of the whole session, given it silently hid the one unambiguous
  OSM signal for the project's most safety-critical hazard.
- **`requirements.txt` was missing `scipy`** -- `scripts/generate_dem_from_contours.py`
  (the exact script that fixed the critical DEM bug in §1) directly imports
  it, but nothing else in the dependency tree pulls it in transitively; a
  fresh `pip install -r requirements.txt` would have made that fix
  unreproducible. Added, along with `numpy` (directly imported in four
  files but previously only present transitively).
- **`mapillary_client.py`'s default tile size (0.009°) contradicted this
  document's own §3.1 finding** that 0.004° is the validated operating
  point -- found independently by both review agents. The shipped data was
  fetched with the correct value passed explicitly, so today's files
  aren't affected, but any future re-fetch without an explicit override
  would have silently regressed to materially worse coverage. Fixed.
- Several stale numbers in earlier drafts of this document, corrected in
  place above rather than listed separately here: `marked_crossing_present`
  and `tactile_paving_present`'s resolved rates (both were computed before
  the node-tag fix above was reflected in the doc text), the segmentation
  class-coverage fraction (7/10, not 8/11), the width-disagreement
  direction (15.5% total, not 13% one-directional), and the weekly-refresh
  changelog narration.
- Smaller code-quality fixes applied: a dead `PRESENT` constant now wired
  up instead of every call site hardcoding the string `"present"`; the
  width-bucket tier order centralized into one shared constant
  (`geometric_attribute_extractor.WIDTH_BUCKET_ORDER`) instead of four
  independent literal copies; `extract_geometric_attributes`'s raw width
  estimate renamed at its source to `width_bucket_uncalibrated_estimate`
  (previously it was returned as `"width_bucket"` and only renamed by its
  one current caller -- a landmine for any future direct caller); a
  duplicated "bottom band of mask" pixel-selection helper unified into one
  shared function; the DEM's elevation sanity bounds named as constants;
  `imputation_engine.impute_missing_attributes` hardened against a raw NaN
  being passed in (not currently triggered by any real call site, but a
  real gap for a pandas-heavy codebase); a missing CLI driver added to
  `edge_attribute_fusion.py` so the delivered fused graph can actually be
  regenerated from a committed script instead of only existing as the
  product of interactive development.
- Findings the audit specifically checked for and did **not** find: CRS
  argument-order bugs, segmentation class-name string mismatches between
  code and real model output, broad exception handling hiding real errors,
  commented-out/dead debug code, or reimplementations of coordinate
  conversion outside `spatial_utils.py` (the centralization principle was
  followed correctly everywhere it applies).

Neither audit pass would have been possible from static reading alone --
every finding here was confirmed by executing real code against the real
project data, not inferred from inspection. Worth keeping full-pipeline,
adversarial audits like this as a standing step before any future phase
change, not a one-time exercise.

### 7.3 Fixed-issue index, for future independent reviewers

Every bug found and fixed in this project to date, in one place, so a new
reviewer (human or model, and specifically including any future independent
cross-model audit of this pipeline) can confirm these are actually closed
rather than re-discovering them from zero. "Verified by" states how each fix
was confirmed -- all by executing real code against real project data, none
by inspection alone.

| # | Issue | Where | Verified by |
|---|---|---|---|
| 1 | DEM raster entirely empty (all 277M pixels = 0.0) | `data_files/lourdes_dem_1m.tif` (§1) | `validate_dem.py` ALERTA→SUCESSO; a real edge's grade changed 0.00%→0.95% |
| 2 | `highway=steps` silently undetectable (list-vs-string comparison) | `osm_extractor.py`, `edge_attribute_fusion.py` (§2) | `highway=steps` count measured 0→8 after fix, re-checked against raw OSM tags |
| 3 | Pedestrian graph never persisted to disk | `osm_extractor.py` (§2) | `salvar_grafo`/`carregar_grafo` round-trip tested |
| 4 | `MAPILLARY_TO_ACCESSIBILITY` keys didn't match real Vistas class names | `semantic_segmentation_training.py` (§3.2) | Diffed against the checkpoint's real `config.json` id2label |
| 5 | Naive label normalization broke multi-hyphen class names | `semantic_segmentation_training.py` (§3.2) | Regex normalizer tested against real class strings incl. `"Crosswalk - Plain"` |
| 6 | False `"guard-rail" -> "handrail"` mapping | `semantic_segmentation_training.py` (§3.2) | Manual review of Vistas class semantics; entry removed |
| 7 | Per-pixel confidence could exceed 1.0 (observed up to 1.34) | `run_segmentation_inference.py::segment_image` (§3.2) | Full 63,843-detection dataset checked post-fix: range confirmed [0.0003, 1.0] |
| 8 | Width heuristic systematic bias (90.9% pinned to most extreme bucket) | `geometric_attribute_extractor.py` (§3.4) | Post-fusion distribution check across all 832 edges, cross-checked against dashcam-FOV evidence from visual QA |
| 9 | Slope raster produced physically-implausible extremes (max 2,418%) | `geometric_attribute_extractor.py::generate_slope_raster` (§3.3) | Full distribution check (median 4.85%, p95 30.94%, 0.44% > 100%) before bounding |
| 10 | GraphML round-trip silently stringifies typed attributes (`'False'` is truthy) | `edge_attribute_fusion.py` (§7) | Loaded a real saved graph, inspected raw types before/after `load_fused_graph()` |
| 11 | `osm_tags_to_canonical` only checked edge-level tags, missing node-level `crossing`/`kerb`/`tactile_paving`/`barrier` | `edge_attribute_fusion.py` (§7) | `marked_crossing_present` OSM coverage re-measured 0%→25.0%, `tactile_paving_present` 1.9%→12.7% |
| 12 | `requirements.txt` missing `scipy`/`numpy` (used directly, only present transitively) | `requirements.txt` (§7) | `pip show scipy`/`numpy` had empty "Required-by" before fix |
| 13 | `mapillary_client.py` default tile size (0.009°) contradicted the validated 0.004° operating point | `mapillary_client.py` (§7) | Cross-checked against §3.1's own tiling experiment results |
| 14 | Dead code / duplication cluster: unused `PRESENT` constant, width-tier order duplicated 4x, duplicated "bottom band" pixel helper, unnamed magic numbers, `imputation_engine` NaN gap, no CLI driver for `edge_attribute_fusion.py` | Multiple files (§7) | Each confirmed individually; CLI driver confirmed by running the fusion stage end-to-end from the command line |

**What this table is not**: a claim that the pipeline is now bug-free. It is
a record of what has specifically been checked and fixed, so a new audit's
time goes toward genuinely new ground -- verifying these fixes still hold
under real execution, and probing code paths and edge cases the two passes
above did not specifically target -- rather than re-finding the same 14
issues from zero. One deliberate omission from this table:
`run_segmentation_inference.py::segment_image`'s 1.38%-pixel-deviation from
HF's exact post-processing (§3.2) is *not* listed as "fixed" because it
wasn't a bug -- it's an accepted, documented, unresolved methodological
deviation. A reviewer re-examining that finding should know the code was
deliberately left as-is, not that it was overlooked.

## 8. Independent cross-model audit (Fable)

Third full audit pass, run 2026-07-20 -- by a different model (Claude
Fable) in a fresh session with no memory of building this pipeline,
specifically because both prior passes (§7) were run by the same model
family that wrote the code, in sessions descended from the same
conversation. Same standard as §7: every finding below was confirmed by
executing real code against the real project data files, none by
inspection alone. §7.3's fixed-issue index was used as directed -- prior
fixes were re-verified under execution rather than re-discovered, and this
pass's time went to code paths the earlier audits did not target.

### 8.1 Confirmed defects, found and fixed

**1. The spatial join had no maximum snap distance, attaching evidence
from other neighborhoods' streets to Lourdes boundary edges.** The
Mapillary fetch (§3.1) covers Lourdes's *rectangular* geocoded bbox; the
graph covers the neighborhood *polygon*. `snap_images_to_edges` assigned
every image to its nearest edge unconditionally, however far away.
Measured with `ox.distance.nearest_edges(..., return_dist=True)` over all
3,689 images against the delivered fused graph: the snap-distance
distribution is bimodal -- 2,179 images within 25m (median 13.7m: real
on-street captures), a thin trough at 25-50m (233 images), then a second
population of **1,277 images (34.6%) snapped from 50-501m away**, of which
83% lie entirely outside the graph's convex hull. 966 of those far images
carry presence-class detections, and 81 edges had received evidence from
them -- i.e. photos of streets in adjacent neighborhoods were marking
Lourdes boundary edges as having ramps/obstacles/crossings. Fixed with a
25m maximum snap distance (`MAX_SNAP_DISTANCE_M`, chosen at the measured
trough between the two populations: GPS error plus half a street width
from an edge centerline); the re-run discards 1,510 images (the 50m+
population plus the trough band).

**2. Image evidence landed on only one of a two-way segment's two
directed edges.** Every one of the 832 edges has a reverse twin (416
pairs; OSMnx's walk network stores each two-way segment as two directed
edges with identical geometry), and `nearest_edges` arbitrarily returns
one of the two equidistant twins. Measured on the delivered fused graph:
**245 of 416 twin pairs disagreed on `n_images_observed`**, and 87/69/99
pairs disagreed on `ramp_present`/`marked_crossing_present`/
`fixed_obstacle_present` respectively -- the same physical street reported
"obstacle present" walking one direction and "unknown" walking the other.
That this was unintended (not a directional-evidence design decision) is
shown by the OSM-derived attributes, which are symmetric: all 8
`highway=steps` edges form 4 clean twin pairs. Fixed by mirroring each
image list onto the reverse twin in `snap_images_to_edges`; re-measured
after the fix: 0 of 416 pairs disagree, on any attribute.

Both fixes shipped together in one re-run of the committed fusion CLI.
Net effect on the delivered data: image coverage rose from 432/832 (52%)
to **508/832 edges (61.1%)** -- mirroring adds more than the far-image
discard removes -- and §6's imagery columns rose accordingly
(`ramp_present` 40.0%→47.1%, `fixed_obstacle_present` 39.8%→48.8%,
`marked_crossing_present` 28.5%→34.4%). §5 and §6 above now show the
corrected numbers with the old values noted in place.

**3. Latent list-valued-tag crash in `osm_tags_to_canonical` -- the §2
`highway` bug's exact mechanism, alive for every other tag.** OSMnx's
`simplify=True` stores disagreeing merged-segment tags as lists (12 real
list-valued `highway` edges exist in this graph today, and osmnx 2.1.0's
`load_graphml` was verified to restore them as real Python lists, not
strings). `highway` comparisons were fixed in §2, but executing the other
tag paths with list inputs showed: `surface`/`smoothness` raise
`TypeError: unhashable type: 'list'` (a list can't be tested for dict
membership) -- killing the entire fusion run -- and
`tactile_paving`/`kerb`/`handrail`'s scalar `==` comparisons silently drop
the evidence. Not triggered by today's data (the tag-type census over all
832 edges shows zero list-valued instances of those tags right now), but
any future OSM edit absorbed by `refresh.py`'s weekly re-pull could
produce one, and the failure would be either a crash or a silent evidence
loss. Fixed with `_tag_values()` normalization on every tag comparison,
resolving mixed lists pessimistically (worst tier / narrowest width /
"no" over "yes" / "raised" over "lowered" -- the same worst-case-wins rule
`aggregate_image_evidence` applies to imagery). Regression-checked by
running old and new logic side-by-side over all 832 real edges with their
real node data: 0 differences on scalar inputs; list inputs verified to
resolve pessimistically instead of crashing.

**4. §6's non-overlap claim was structurally unmeasurable from its own
table.** The earlier prose read combined-≈-sum as evidence that OSM and
imagery crossing coverage were "largely non-overlapping." But because OSM
takes priority in the fusion, the imagery column only ever counts edges
where OSM was silent -- Combined *always* equals the sum, so the claim was
circular. Measured independently (running `osm_tags_to_canonical` and
`aggregate_image_evidence` separately per edge): imagery observes
crossings on 46.2% of all edges, OSM on 25.0%, overlapping on 11.8% --
complementary, but with real overlap. §6's text is corrected above.

### 8.2 Verified correct (checked specifically, found no defect)

Re-verification of §7.3's fixed-issue index -- all 14 hold under real
execution: #1 `validate_dem.py` re-run on the shipped DEM prints SUCESSO
(5,311,026 px = 2,562×2,073, 85.5% valid, 838.0-928.0m, mean 875.3m --
every §1 number exact); #2 the `highway=steps` fix holds through the real
CLI load path (8 edges, `steps_present=present`, `source=osm` in the
delivered fused graph); #3 implicitly via every graph load in this pass;
#4/#5/#6 via a class-name census of all 63,843 real detections -- every
class name the fusion and geometry code compares against ("Curb Cut",
"Crosswalk - Plain", "Lane Marking - Crosswalk", "Sidewalk", "Pedestrian
Area", "Curb", ...) appears verbatim in the delivered predictions, no
mismatches, no "Guard Rail" mapping; #7 confidence recomputed over all
63,843 detections: range exactly [0.0003, 1.0]; #8 width bias re-measured
after this pass's re-fusion: 91.3% `under_50cm` (460/504) -- the §3.4 gate
remains warranted; #9 the unbounded slope distribution recomputed from the
shipped DEM reproduces every claimed figure exactly (median 4.85%, p95
30.94%, 0.44% > 100%, max 2,418%) and the shipped bounded raster contains
nothing above 150.0%; #10 `load_fused_graph` restores `n_images_observed`
as int, every `_imputed` flag as real bool, `ramp_declivity_pct` as float;
#11 node-tag coverage re-measured (crossing on 41/278 nodes → 25.0%,
tactile 12.7%); #12 `scipy`/`numpy` present in requirements.txt; #13
`DEFAULT_TILE_DEGREES = 0.004` confirmed in shipped code; #14 the CLI
driver exercised end-to-end by this pass's own re-fusion run, and
`impute_missing_attributes(NaN)` verified to impute rather than pass NaN
through.

Beyond re-verification, this pass specifically checked for and did **not**
find problems in:

- **Every §2 tag-coverage number**, recomputed from the delivered
  `lourdes_graph_latest.graphml`: surface 58.2% (484/832), footway 9.6%
  (80), tactile edge-level 1.9% (16), smoothness and lit 0.5% (4 each),
  highway 100%, all the 0.0% rows, steps count 8 -- all exact.
- **Every §3 dataset count**: 3,697 metadata rows, 3,689 predicted images,
  63,843 detections -- exact.
- **Every §3.4 validation number**, by re-running
  `scripts/validate_geometric_heuristics.py` against the delivered
  predictions: 3,579 compared, 57.8% agreement, 38.2% chance agreement,
  Cohen's kappa 0.32, middle-bucket 646→118 confirmed, opposite-extreme
  463+92=555 (15.5%) -- all exact.
- **Pessimistic aggregation semantics**, on a real 45-image edge whose
  images genuinely disagree (some show ramps/obstacles/crossings, some
  none): OR-semantics for presence, narrowest width bucket among mixed
  per-image buckets, steepest declivity -- aggregate output matches the
  stored edge attributes.
- **Imputation policy coverage**: all 9 canonical attributes covered; an
  attribute not in the policy table passes through untouched with no
  spurious flag and no KeyError path; `summarize_imputation_rate` output
  matches §6's Imputed column.
- **CRS centralization and argument order**: grep confirms no
  pyproj/manual reprojection outside `spatial_utils.py` (only
  `ox.project_graph`/`gdf.to_crs` library calls); every `to_dem_crs`
  call site passes (lat, lon) in the documented order.
- **Routability of the delivered fused graph** (the §6 experiment's
  precondition): single connected component, 50/50 random
  origin-destination pairs route via plain networkx Dijkstra
  (weight=length), zero NaN elevations across 278 nodes, zero NaN grades
  across 832 edges, |grade| median 2.87% / max 17.8% (plausible for
  Lourdes's terrain).
- **The CSV export's type integrity**: `pandas.read_csv` on the delivered
  `lourdes_edges_fused.csv` parses every `_imputed` column as real bool
  and `n_images_observed` as int64 -- the GraphML `'False'`-is-truthy trap
  (§7 #10) does not recur through the CSV path for pandas consumers.

### 8.3 Open items from this pass (reported, not silently resolved)

- **The 25m snap threshold is a measured judgment call, not ground
  truth.** It sits in the clear trough of a bimodal distribution, but the
  233 discarded images in the 25-50m band are mixed -- mostly inside the
  graph hull, plausibly including some legitimate captures on wide
  boulevards. If per-image calibration or field checks ever happen,
  re-examine this band.
- **§3.1's raw-fetch counts (6,579 / 22,943 / 39,035 images at 4/16/56
  tiles) are not re-verifiable from disk** -- only the downselected 3,697
  rows are committed, and re-fetching is out of audit scope. Treated as
  recorded history: plausible, consistent with the committed artifacts,
  but unverified by this pass.
- **`crossing=unmarked`/`crossing=no` would incorrectly register as
  `marked_crossing_present`** -- the check is `is not None`, not a value
  test. Zero instances in today's data (all 41 crossing nodes carry
  `traffic_signals`/`marked`/`uncontrolled`), so this was left unfixed as
  unexercised, but a future OSM refresh could import such a value; the
  right mapping for `unmarked` (absent vs. unknown) is a semantics
  decision worth making deliberately, not in passing.
- **`ramp_compliance` is computed per image but never reaches any edge**
  -- `extract_geometric_attributes` returns it, `aggregate_image_evidence`
  doesn't propagate it (only `ramp_declivity_pct`). Not a correctness bug
  (nothing downstream expects it yet), but the future impedance model
  should either consume it or the dead output should be removed.
- **`aggregate_image_evidence` can only ever produce `PRESENT`** -- an
  image with a clear, unobstructed view of a sidewalk and no ramp
  detection contributes nothing toward "ramp absent." All negative
  evidence currently comes from OSM or imputation. This is a conservative
  design (a missed detection isn't evidence of absence), not a bug, but
  it caps how much imagery can ever reduce the imputation rate for
  infrastructure attributes; worth stating as a known asymmetry in the
  article.

## 9. Detection precision of the imagery-only presence attributes (Fable)

The §8 audit closed the pipeline's *mechanical* correctness but flagged a
substantive gap it could not close by execution alone: `ramp_present`
(47.1%) and `fixed_obstacle_present` (48.8%) are the only two attributes in
the fused schema whose real signal comes *entirely* from imagery -- 0% OSM
coverage -- and they are high-impact obstacle terms in the cost formula,
not tie-breakers. Yet unlike every other attribute (OSM cross-check for
crossings/surface/steps; cross-method agreement for width, §3.4), these two
had never been checked against anything beyond the 3-image visual QA in
§3.2. They rest entirely on the pretrained Mask2Former's "Curb Cut" and
obstacle-class outputs being right. If it systematically over-detects, the
OSM-vs-fused comparison in §6 inflates in exactly the hoped-for direction,
and nothing else in the repo would catch it. This section measures that.

### 9.1 Method

`scripts/validate_detection_precision.py` renders a fixed, seeded sample of
real detections as full-frame + zoomed-inset overlays on the real Mapillary
thumbnails (the same images inference ran on -- URL liveness and
stored-vs-downloaded dimension match confirmed first). Adjudication is
visual, per detection: **TRUE** (a curb cut / the named object is clearly
there), **PLAUSIBLE** (right location -- a corner/crossing with a curb, or
a vertical object where a bin/hydrant could be -- but the frame can't
confirm it), or **FALSE** (the polygon lies on open road surface, crosswalk
stripes, a vehicle, or the dark dashcam hood/foreground). This is a
precision check only; it shares the honest limits of §3.4 -- no
field-measured ground truth, single monocular thumbnails, adjudication by a
vision-capable model (Fable), not a surveyor. Verdicts are recorded in the
script's companion sample so the numbers below can be regenerated and
re-judged rather than taken on faith.

Sampling is deliberately two-pronged: an unbiased random sample of
detections (the precision estimate) plus the smallest-area detections,
because in `aggregate_image_evidence` **any** Curb Cut detection -- even a
24px sliver at confidence 0.014 -- flips `ramp_present` to `present`, so
tiny false positives do maximum damage and deserve separate scrutiny.

### 9.2 Results

**Curb Cut → `ramp_present` (30 random detections):** 6/30 clear TRUE, 6/30
PLAUSIBLE, 18/30 FALSE. **Precision 20% strict (TRUE only) to 40% lenient
(TRUE + PLAUSIBLE all counted correct).** The 10 smallest-area detections
(24-29px): **0/10** -- every one a thin sliver on open road, a crosswalk
stripe, or beside a car wheel. The FALSE cases cluster on three recurring
artifacts: the dark dashcam hood/foreground (BLACKVUE-watermarked frames,
consistent with the dashcam-source limitation already noted in §3.2),
crosswalk paint mistaken for the ramp, and generic road-surface slivers.
The clear-TRUE cases are all larger detections (≥0.09% of frame) at genuine
street corners with visible ramps.

**A confidence threshold does not rescue this -- checked explicitly.**
Restricting to conf ≥ 0.85 keeps strict precision at just 25% (8 detections
survive, 2 clear-TRUE); several of the highest-confidence detections (0.90+)
are unambiguous false positives on the dashcam hood or road surface.
Confidence and correctness are largely decorrelated here, so raising the
bar mostly discards true and false detections together. (For reference, the
fusion currently applies *no* confidence floor at all -- inconsistent with
the 0.30 threshold `summarize_barriers_per_image` declares for the scoring
path -- but that gap is nearly irrelevant to Curb Cut: only 2% of Curb Cut
detections fall below 0.30. It matters more for `Barrier`, below.)

**Obstacle classes → `fixed_obstacle_present` (22 detections, stratified
across classes):** 7/22 clear TRUE, 3/22 PLAUSIBLE, 12/22 FALSE.
**Precision 32% strict to 45% lenient.** Two class-specific findings: (1)
`Barrier` is the noisiest class and the largest contributor -- its FALSE
cases are dashcam hood, motion blur, and generic dark foreground, and 39%
of *all* `Barrier` detections fall below even the pipeline's own stated
0.30 confidence threshold yet are all used by fusion; (2) three of the
clear-TRUE detections are `Manhole`s correctly identified but located *in
the roadway*, not on the pedestrian path -- correct class, but not a
mobility obstacle, so even true positives over-count sidewalk obstruction
somewhat. Real, correctly-detected obstacles do exist (a bench, a bike
rack, a roadside fence were all confirmed), so this is genuine signal --
just low-precision signal.

**Recall (under-detection) could not be adjudicated from thumbnails.** A
sample of 10 images that have both a crossing and a curb but *no* Curb Cut
detection showed no blatant missed ramps -- but confirming the *absence* of
a small feature at street-corner distance in a single thumbnail is
inherently unreliable, so this is reported as inconclusive, not as evidence
of good recall. The clearly-demonstrated error mode is over-detection, not
under-detection.

### 9.3 Implication for the routing experiment -- a decision, not a bug

This is not a defect to patch. It is a measured property of the pretrained
model, and it directly shapes what the §6 routing comparison can honestly
claim: the imagery-derived presence attributes carry real information (both
are 0% in OSM, so any true detection is information OSM lacks) but at
roughly 20-45% precision, which means a substantial fraction of the edges
currently marked `ramp_present`/`fixed_obstacle_present` are false
positives. A routing comparison run on this data as-is would likely
overstate imagery's benefit.

No code was changed in response, deliberately -- the same reasoning as the
§3.4 width gate: imposing an uncalibrated confidence/area threshold to
"clean up" the detections would swap one unvalidated assumption for
another, and the threshold sensitivity above shows it wouldn't even work.
The three honest paths forward, for the project owner to choose between
before the routing phase, are:

1. **Fine-tune** (the existing stretch goal, currently scoped to
   `steps`/`handrail`/`tactile_paving`) -- widen it to also improve Curb
   Cut / obstacle precision, which §9 now shows is the higher-leverage
   target. This is the only path that raises precision rather than trading
   it against recall.
2. **Proceed with the limitation stated**, and design the routing
   experiment to be robust to it -- e.g. report results as a function of a
   detection-confidence sweep, or frame the imagery condition as an upper
   bound on effect size rather than a point estimate, with §9's precision
   band cited explicitly.
3. **Manually correct** the ~500 imagery-touched edges against their source
   images (feasible at this graph size: 508 edges, and the overlay tooling
   now exists) to produce a validated subset for the experiment.

What is *not* defensible is running the comparison, reporting a headline
effect size, and omitting that its two dominant terms are ~20-45%-precision
signals. §9 exists so that can't happen by omission.

### 9.4 Per-edge manual validation (option 3) -- COMPLETE

The project owner chose option 3: hand-validate every imagery-touched edge
into a trusted subset. `scripts/edge_validation.py` implements this as a
resumable workflow -- the unit of work is one undirected physical segment
(the two directed OSMnx twins collapsed) per attribute, and it renders the
*single largest-area* triggering detection for that segment as the
representative to judge (an edge is imagery-`present` iff at least one image
triggers, so confirming one strong detection confirms the segment; §9.2
established that large detections are the reliable-true ones). Verdicts
(`present`/`absent`/`uncertain`) are recorded in a tracked JSON label store
(`data_files/edge_validation_labels.json`, force-tracked past the
`data_files/` ignore because it is hand-curated ground truth, not generated
data), keyed by `segment_id|attribute` so the work is fully resumable and
`apply` writes the results back onto the graph as `<attr>_validated` /
`<attr>_validated_value` without touching the original imagery value.

**All 399 (segment, attribute) judgments were completed** -- 196
`ramp_present` + 203 `fixed_obstacle_present` segments, each adjudicated
against its real source image by a vision-capable model (Fable). The two
attributes gave sharply different results, and reporting them as one blended
number would hide the single most important fact this validation produced:

| Attribute | present | absent | uncertain | **precision on resolvable** |
|---|---|---|---|---|
| `ramp_present` | 118 | 12 | 66 | **118/130 = 90.8%** |
| `fixed_obstacle_present` | 60 | 104 | 39 | **60/164 = 36.6%** |

**`ramp_present` is trustworthy.** At 90.8% precision on the 130 resolvable
segments, the imagery-derived curb-ramp signal is solid -- and it is the
attribute the routing thesis leans on hardest (0% OSM coverage, high cost
weight). The 90.8% figure is dramatically higher than the 20-40%
*detection*-level precision in §9.2, and that gap is itself a finding: a
segment with a genuine curb ramp almost always has at least one clean corner
capture among its several images even though most individual detections on
it are false positives, so aggregating to the segment and judging its best
evidence recovers signal that per-detection precision badly understates.
This is direct empirical support for the fusion's pessimistic-OR aggregation
being the right design for presence attributes. Most confirmed ramps were
corner crossing ramps and driveway/garage lowered curbs (both genuine
lowered curbs a wheelchair can use).

**`fixed_obstacle_present` was NOT trustworthy as-shipped, and §9.4 found
exactly why.** At 36.6% precision it was little better than a coin flip --
but the failure is almost entirely one class. Broken down by the
representative detection's Vistas class:

| Class | present | absent | precision |
|---|---|---|---|
| **Barrier** | 2 | 71 | **3%** |
| Manhole | 26 | 4 | 87% |
| Fire Hydrant | 3 | 0 | 100% |
| Trash Can | 26 | 24 | 52% |
| Bench | 2 | 2 | 50% |
| all non-`Barrier` | 58 | 33 | **64%** |

Vistas' `Barrier` class is effectively noise for this purpose: it fired on
passing vehicles, the dashcam hood/foreground, raised road medians, and
night motion-blur -- 71 of 73 resolvable Barrier segments were false
positives. **Fix applied:** `Barrier` was removed from
`edge_attribute_fusion.CLASS_TO_PRESENCE_ATTR`, an evidence-based change
(73 hand-validated segments, not a guessed threshold), and the fused graph
regenerated. This raises segment-level obstacle precision from 36.6% to
**64%** and drops `fixed_obstacle_present`'s imagery coverage modestly from
48.8% to 44.5% (only segments whose *sole* evidence was a Barrier lose their
flag; segments also seeing a manhole/bin/hydrant keep it). The reliable
true positives that remain are concrete, checkable objects -- manholes,
construction dumpsters, trash bins, fire hydrants, a portable toilet, café
furniture -- genuinely obstructing the pedestrian path.

**The uncertain fraction is a real, reported ceiling.** 66/196 ramp (33.7%)
and 39/203 obstacle (19.2%) segments could not be adjudicated from the
available thumbnails -- night captures, heavy motion blur, distance, or
occlusion. This is a limitation of the Mapillary source for this
neighborhood (many captures are night-time dashcam drive-throughs), not only
of the model, and it bounds how far any *imagery-only* validation can ever
go. `apply` deliberately writes a validated value only for present/absent
verdicts, leaving `uncertain` segments unflagged so the routing phase can
decide per experiment whether to treat them as imagery-present, imputed, or
excluded.

**Committed artifacts.** `data_files/edge_validation_labels.json` (399
verdicts, tracked ground truth). `data_files/lourdes_graph_validated.graphml`
(regenerated, git-ignored like other derived data) carries validated flags
on **260 `ramp_present` edges** (236 present / 24 absent) and **294
`fixed_obstacle_present` edges** (120 present / 174 absent) -- the resolvable
segments times their directed twins. Regenerate anytime via
`python -m scripts.edge_validation apply`.

### 9.5 Verdict: is the accuracy enough to proceed?

**Yes for `ramp_present`, conditionally yes for `fixed_obstacle_present`
after the Barrier fix, with a stated caveat -- and the routing experiment
should run on the validated subset, not the raw imagery.** Concretely:

- **`ramp_present` (90.8%): proceed.** This clears any reasonable bar for a
  routing input, and it is the highest-leverage imagery attribute. Route on
  the validated `present`/`absent` values; treat the 34% `uncertain`
  segments as imputed (the existing pessimistic default) rather than
  asserting a ramp that was never confirmed.
- **`fixed_obstacle_present` (64% post-fix): proceed with a stated
  limitation, or restrict to the validated subset for the headline result.**
  64% is usable for a *penalty* term (a false obstacle over-avoids a clear
  path -- suboptimal, not unsafe) but should not be presented as a precise
  measurement. For the scientific claim, use the 294 hand-validated obstacle
  edges as ground truth and report the model's precision alongside, rather
  than treating every raw detection as fact.
- **What would make it flawless (not required to proceed, but the honest
  path to a stronger claim):** (1) the `uncertain` segments are the binding
  constraint now -- resolving them needs *better imagery* (daytime,
  pedestrian-height, higher-resolution captures) or a light field check on a
  sample, not more model work; (2) a narrowly-scoped fine-tune on
  `steps`/`handrail`/`tactile_paving` (still zero imagery signal) plus
  Curb-Cut/obstacle precision would lift both coverage and precision; (3)
  the routing experiment itself should report results as a function of the
  validated-vs-raw imagery condition, so the effect of detection error on
  the conclusion is measured, not assumed.

The bottom line for the competition article: the pipeline's central claim --
that street-level imagery contributes accessibility information OSM lacks --
now rests on a **hand-validated** curb-ramp signal at 90.8% precision and a
**cleaned, class-audited** obstacle signal, with every unconfirmable edge
honestly marked as such. That is a defensible empirical footing. It is not
"flawless" in the sense of field-surveyed ground truth -- no imagery-only
method can be -- and the article should say so plainly, which is itself the
kind of honesty that makes the result credible.

## 10. Writing the scientific article: defensible claims and mandatory caveats

Consolidated guidance for the person writing the 32º Prêmio Jovem Cientista
article, so the paper claims neither too much nor too little. Every item
here traces to a number computed from real data elsewhere in this document.

### 10.1 Claims the data supports -- state these with confidence

- **Street-level imagery contributes accessibility information that OSM does
  not have, in this under-mapped neighborhood.** `ramp_present` and
  `fixed_obstacle_present` both have **0% OSM coverage**; without imagery
  they would not exist for any edge in Lourdes (§6). This is the central,
  defensible thesis.
- **The curb-ramp signal is hand-validated at 90.8% precision** on 130
  resolvable segments (§9.4) -- not asserted from model output, but checked
  against the real source photographs one segment at a time.
- **The obstacle signal, after a data-driven class audit, is ~64%
  precision** on reliable object classes (manhole, fire hydrant, trash bin,
  dumpster); the one systematically bad class (Vistas `Barrier`, 3%
  precision) was identified and removed with evidence, not intuition (§9.4).
- **Every quantified claim in this document was independently recomputed
  from the real data across three audit passes** (§7, §8), two of them by a
  different model. Several real bugs were caught, fixed, and re-verified --
  the pipeline is not trusted by construction.

### 10.2 Limitations you MUST state -- omitting these is the "slop" failure

- **This is NOT field-surveyed ground truth, and no imagery-only method can
  be.** State this explicitly. **105 of 399 imagery-flagged segments (26%)
  are genuinely unconfirmable** from the available Mapillary imagery --
  night-time dashcam captures, motion blur, distance, occlusion. They are
  honestly marked `uncertain` and never asserted as fact. This is a
  limitation of the **data source**, not the model or the method; more model
  work cannot fix it. Putting this *in the paper* is what separates a
  credible result from one that merely looks rigorous.
- **The precision figures come from a vision-model adjudication (Claude
  Fable) of single monocular thumbnails, not a surveyor with a tape
  measure.** Good enough to flag and rank accessibility features; not a
  calibrated physical measurement. Say so.
- **`width_bucket` is fully imputed** -- a systematic camera-FOV bias makes
  the raw estimate untrustworthy (§3.4), so it carries zero real signal in
  the delivered graph. `handrail`, `tactile_paving`, and `steps` have zero
  imagery signal at all (no Vistas class); `steps` has a reliable but
  partial OSM signal (8 edges, §2).
- **The Mapillary tile-density / coverage findings are this project's own
  testing, not documented API behavior** (§3.1), and the raw-fetch counts
  are not re-verifiable from the committed data. Present them as observed,
  not as general claims.
- **The routing effect size is still unmeasured** until the §11 experiment
  runs. A modest or null result, honestly measured, is still a valid finding.

### 10.3 The single most important framing

The credibility of this project does not come from the numbers being
perfect -- they are not, and the paper should not pretend otherwise. It
comes from **every uncertainty being measured and reported rather than
hidden**: the 26% unconfirmable fraction, the per-class precision, the
imputed attributes, the caught-and-fixed bugs. Report those *as findings*.
An honestly-bounded accessibility map is a genuine contribution to a
literature that usually assumes crowdsourced data is either complete or
uniformly sparse; a map that overclaims precision it cannot defend is not.

## 11. Routing experiment specification (for the next phase)

The next session builds `core/impedance_model.py` and the routing
algorithms and runs the OSM-vs-imagery comparison. This section specifies
that experiment so it produces a defensible scientific finding, not merely a
router that runs. **This is a specification, not an implementation -- the
router is deliberately still out of scope for the data-pipeline phase.**
Hand this section (plus §6, §9.4, §9.5) to whoever builds the routing phase.

### 11.1 The two conditions being compared

The finding is a *controlled comparison of the same router over the same
graph* under two data conditions:

- **Baseline (OSM + DEM only):** cost graph built from OSM tags + slope
  alone, no imagery. Reproducible today: run the fusion CLI with an empty
  predictions file (`echo '[]' > empty.json`), already verified to produce
  exactly the "OSM alone" column of §6.
- **Full (fused):** the delivered fused graph, imagery included.

### 11.2 Ground truth: route on the validated subset, not raw model output

For the **headline** result, the two imagery-only presence attributes should
take their **hand-validated** values (§9.4), not raw model output:

- `ramp_present`: 260 validated edges (236 `present` / 24 `absent`) in
  `lourdes_graph_validated.graphml` (`ramp_present_validated_value`).
- `fixed_obstacle_present`: 294 validated edges (120 `present` / 174
  `absent`).
- The 105 `uncertain` segments carry no validated flag -- treat them as
  **imputed** (the existing pessimistic default: `absent` for ramp as
  infrastructure, `unknown` for obstacle as hazard). Do not assert a feature
  that was never confirmed.

Then run a **secondary** condition using the raw (post-Barrier) imagery
values, and report both. The gap between "validated" and "raw imagery"
routing outcomes *is* the measurement of how much detection error changes
the conclusion -- that is the number that makes the result honest instead of
assumed.

### 11.3 How each attribute should feed the cost model

Cost formula (from the proposal): `Custo = Distância × Fator_superfície +
Penalidade_obstáculo`, plus a slope/grade term from the DEM.

- `ramp_present` (trustworthy, 90.8%): the high-leverage positive term --
  a validated curb ramp should *reduce* the cost of a curb transition /
  enable a crossing. This is where imagery earns its place.
- `fixed_obstacle_present` (64% post-fix, or use validated): an **obstacle
  penalty**. A false positive here over-avoids a clear path -- suboptimal
  but not unsafe -- so a penalty (not a hard block) is the right treatment.
  **`Barrier` is already excluded at the data level (§9.4); the router must
  not re-introduce any Barrier-derived obstacle.**
- `steps_present` (hazard, 3-state): `unknown` must stay `unknown` -- never
  silently route someone into a staircase. The 8 OSM-derived steps edges are
  reliable (§2) and should carry a strong penalty or hard avoidance for
  wheelchair profiles.
- `surface_material_tier` and slope/grade: cost multipliers (`Fator_superfície`
  and the topographic term).
- `width_bucket`, `smoothness_tier`, `handrail_present`, `tactile_paving_present`:
  mostly or fully imputed -- include them if the cost model wants them, but
  do not let a fully-imputed attribute dominate a route decision, and report
  their imputation rate (§4, §6) so their weight is honest.

### 11.4 Metrics -- the actual scientific output

Over a sample of origin-destination (OD) pairs across Lourdes, compute in
both conditions:

1. **What fraction of routes differ** between baseline and full.
2. **Among differing routes, how many avoid a real (validated) obstacle or a
   steeper ramp** the OSM-only route would have taken -- i.e. is the
   difference an *improvement* for a mobility-impaired user, not just a
   change.
3. **How the estimated cost distributions compare** between conditions.

Report the §9.4 precision numbers alongside every result, so a reader sees
the detection-error bound on the effect, not just the point estimate.

### 11.5 OD-pair sampling

Use real, stated OD pairs -- e.g. key origins (a metro station, a hospital,
a plaza) to a spread of destinations, or a random sample of node pairs
stratified by straight-line distance. State the sampling rule; do not
hand-pick pairs that flatter the result.

### 11.6 Traps this audit already found -- do not re-introduce them

- **Load the fused graph with `edge_attribute_fusion.load_fused_graph`, never
  `carregar_grafo`** -- the latter leaves every `_imputed`/validated flag as
  the string `'False'`, which is truthy, silently inverting them (§7 #10).
- **`steps_present` and `fixed_obstacle_present` are 3-state, not boolean.**
  A two-branch `if present / else` treats `unknown` as safe -- the exact
  failure §4 exists to prevent. Over half the graph is `unknown` on these.
- **Do not re-add `Barrier` as an obstacle source** (§9.4).
- **Do not report a headline effect size without the precision caveat** -- a
  routing difference driven by a false detection is not a real improvement.

## Open items

- **`uncertain` segments (105 total: 66 ramp + 39 obstacle) are the binding
  limit on imagery-only validation** -- resolvable only with better imagery
  (daytime/pedestrian-height/higher-res) or a sampled field check, not more
  model work. This is now the top data-quality follow-up.
- Stretch goal (YOLO26 fine-tune): not started. Cost estimate from
  planning: ~$2-8 on Azure/GCP T4 spot pricing for a full 100-epoch run.
  Originally scoped to the 3 gap classes (`steps`/`handrail`/
  `tactile_paving`); §9 shows Curb Cut / obstacle *precision* is the
  higher-leverage target, so a re-scope to include those is worth
  considering. Per §6 this remains downstream of the routing comparison in
  sequencing, but §9's precision decision is now the immediate gate before
  that comparison can produce a defensible number.
- Width bucket carries known, quantified uncertainty (§3.4) and is
  currently fully imputed rather than trusted -- revisit if real per-image
  camera calibration or field measurements become available.
- `core/impedance_model.py` and the routing algorithms remain out of scope
  for this document -- see §6 for exactly what they need to do and why.
