# AnaMorph Python technical reference

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).

This folder is an **incomplete** independent Python implementation. It reads SWC trees,
partitions neurites into paths, builds clamped cubic splines, and generates
an **experimental** analytic triangular OBJ surface by default. It also includes
a greedy preconditioning implementation, experimental edge-collapse cleanup,
HC smoothing, experimental isotropic remeshing, approximate sampled geometry
warnings, basic SWC diagnostics, and OBJ mesh statistics.

`generate` uses canal surfaces and red-blue
unions; `generate-redblue` is an alias. A voxel-based approximation is available
only through the explicitly selected `generate-voxel` command. There is no
automatic fallback from the analytic method to voxel meshing.
Existing scripts that used `generate ... --spacing ...` for voxel meshes
must switch to `generate-voxel ... --spacing ...`.

This is not yet a replacement for `am_cellgen`: the voxel union is an alternative
algorithm, not AnaMorph's canal-surface triangulation and red-blue stitching.
The original's full network-level analytical consistency analysis and neuron
meshing pipeline have not been ported. A restricted red-blue mesh union and
curve-level analytic checks are available as development building blocks. The
greedy edge-collapse implementation has different safety
checks and update order from the C++ version. Its watertightness,
non-self-intersection, and mesh-quality claims do not apply to these outputs.
A primary distance diagnostic alone is not a consistency analysis. Grid spacing
can affect topology, miss thin processes, and alter morphology; inspect output
before using it for any simulation. Voxelization can be costly for large cells.

Unlike the C++ input routine, this reader retains the SWC soma samples as-is
and accepts non-consecutive IDs. `generate-redblue` first consolidates a connected
soma-section tree using the source centroid and maximum-radius rule. The
preconditioner operates on the input
nodes without the C++ soma/root adjustment when called directly. Analytic
preparation retains root and descendant edge order through soma consolidation;
some C++ edge-ID allocation and tie ordering remain approximate. The
voxel mesher represents soma samples by spheres and connects soma and neurites
with straight, constant-radius segments. Its radii are linearly interpolated over
short sampled spline chords, not the original arc-length canal-surface model.
The analytic pipeline uses stable summation for a multisample soma centroid.
For C++ comparisons, `--source-soma-centroid` instead accumulates section
coordinates in the original order; the corresponding Python option is
`source_soma_centroid=True`. This is available on analytic analysis, source-job
planning and tracing, MorphView export, and both analytic mesh generators.
Sequential addition can lose small contributions when large values cancel;
it raises an error if the sum overflows. The option requires preparation and
cannot be combined with `--raw`.
Uniform parameterization follows the original C++ implementation's unusual
interior-knot formula. The remaining differences affect
geometry and mean output cannot be assumed equivalent to the C++ version.
For a neurite root exactly on its soma sphere, Python keeps the existing root
instead of inserting the coincident sample created by the C++ strict-boundary
rule. Pass `--source-surface-roots` to use the original rule; coincident
samples can prevent spline construction when preconditioning is disabled.
Likewise, when an interior root is moved exactly onto its child's surface
position, the default preparation absorbs that child and keeps its branches;
the source-compatible option retains the coincident samples.
An isolated neurite root on the sphere has no segment to render; analytic
meshing reports an error instead of silently omitting it, and analysis marks
it as unresolved.

Python 3.10 or later is required. The implementation has no runtime dependencies.
Install from this folder with `python -m pip install .`, or run it without
installing:

For a one-command analytic mesh and comparison report:

```sh
python -m anamorph workflow path/to/cell.swc --output-dir cell_results
```

The result folder contains `input.swc`, `raw.obj`, `remeshed.obj`, a readable
`report.md`, and a detailed `report.json` with settings and SHA-256 checksums.
The report compares triangle sizes, topology, numerical intersection checks,
area and volume. Existing result files are never overwritten; choose another
output folder for a new run. Without `--output-dir`, the folder is named
`<input name>_results` in the current directory.

By default `remeshed.obj` is produced by isotropic remeshing, ten Taubin
smoothing iterations, and the experimental mortar soma-junction fill, in that
order. The default target edge length is the median input neurite radius, a
heuristic. Compare the raw and processed meshes and inspect the report.

Use `--standard-postprocess` to select source-style greedy edge collapse, a
local soma-junction blend, and HC smoothing. The collapse implementation has a
Python work limit and can differ numerically from the C++ implementation.

To set an explicit isotropic target:

```sh
python -m anamorph workflow path/to/cell.swc --output-dir cell_results --target-edge-length 0.4
```

The analytic workflow also accepts a soma-free SWC component whose root is a
neurite (type 2, 3, or 4) with parent `-1`. The component must contain at
least one neurite segment. A root with one outgoing path gets a flat closed
cap at its first sample. For a root that branches immediately, a rounded
junction is built at that sample using its radius; no soma node is added to
the SWC. Forests can mix soma-containing and soma-free components. For example:

```sh
python -m anamorph workflow examples/no_soma.swc
```

The root cap and junction are part of the generated surface. They do not infer
a missing soma or reconstruct the morphology beyond the recorded root. A
short terminal branch numerically contained inside the rounded junction adds
no separate exterior surface. The existing `--soma-refinements` setting also
controls the tessellation of a rounded soma-free root junction.

For `--isotropic`, the target is measured in the SWC coordinate units. If omitted, the workflow
uses the median input neurite radius as a starting heuristic; inspect the
report and choose a target appropriate for your morphology. Use `--iterations`
to change the number of remeshing passes. By default, no pre-meshing analysis
is run; the workflow checks both output meshes afterward. Add `--analyze` for
an optional, non-blocking Python geometric analysis in the report. That analysis
is incomplete. `--skip-intersection-check` omits the potentially expensive
numerical mesh checks and records that they were not performed. Neither
analysis nor numerical checking certifies the mesh, and remeshing changes the
surface without guaranteeing uniformity.
Python callers can use `anamorph.workflow.run_workflow`; set
`target_edge_length=0.4` to override the default heuristic, or select
`postprocess_method="standard"` for source-style postprocessing.

Other commands are also available:

```sh
python -m anamorph inspect path/to/cell.swc
python -m anamorph inspect path/to/cell.swc --alpha 3.0
python -m anamorph analyze path/to/cell.swc
python -m anamorph analyze-analytic path/to/cell.swc
python -m anamorph analyze-analytic path/to/cell.swc --job-mode source
python -m anamorph plan-source-jobs path/to/cell.swc
python -m anamorph trace-source-jobs path/to/cell.swc
python -m anamorph analyze-analytic path/to/cell.swc --check-adjacent-end-start
python -m anamorph analyze-analytic path/to/cell.swc --scale-radius 0.7
python -m anamorph generate path/to/cell.swc cell.obj
python -m anamorph generate-voxel path/to/cell.swc voxel.obj --spacing 0.25
python -m anamorph generate-individual path/to/cell.swc individual.obj
python -m anamorph generate-redblue path/to/cell.swc cell.obj --flush-face-limit 100000
python -m anamorph generate-voxel path/to/cell.swc voxel.obj --spacing 0.25 --adjust-soma-roots --precondition --edge-collapse --smooth-iterations 10
python -m anamorph meshstat path/to/mesh.obj
python -m anamorph meshquality path/to/mesh.obj
python -m anamorph meshcheck path/to/mesh.obj
python -m anamorph meshcheck path/to/mesh.obj --include-adjacent
python -m anamorph mesh-boolean red.obj blue.obj difference.obj --operation difference
python -m anamorph remesh path/to/mesh.obj processed.obj
python -m anamorph remesh path/to/mesh.obj uniform.obj --isotropic --target-edge-length 0.09
python -m anamorph postprocess path/to/mesh.obj processed.obj --edge-collapse --hc-iterations 10
python -m unittest discover -s tests
```

For a small included example:

```sh
python -m anamorph generate examples/branched.swc branched.obj
python -m anamorph remesh branched.obj uniform.obj --isotropic --target-edge-length 0.09
python -m anamorph meshstat branched.obj
python -m anamorph analyze-analytic examples/parallel_overlap.swc --raw --max-rectangles 100
```

The experimental analytic path can be exercised with
`python -m anamorph generate-redblue examples/branched.swc redblue.obj --angular-segments 8`.
To save the stitched mesh and an independently remeshed version in one run:

```sh
python -m anamorph generate-redblue examples/branched.swc raw.obj --angular-segments 8 --soma-refinements 2 --uniform-edge-length 0.14 --uniform-iterations 8 --postprocess-output uniform.obj --check-output
```

The second OBJ is derived from the raw red-blue result. Isotropic remeshing is
an experimental Python post-processing method, not AnaMorph's original
stitching algorithm; it changes the surface and does not guarantee
quasi-uniformity. The command reports edge-length percentiles and area and
volume changes. Without `--postprocess-output`, the requested output is the
remeshed mesh. `--uniform-max-operations` controls the remeshing work limit.
If collapse or HC smoothing is also enabled, it runs before isotropic
remeshing, and the reported changes cover all enabled stages.
Analytic generation does not run pre-meshing analysis unless requested. It has
been tested on synthetic and two real examples and compared with a patched
Windows C++ build. Those limited comparisons do not establish full parity;
failures and poor triangles remain possible.
The mesher selects a seeded random angular offset for each outer attempt,
and a new random render vector when the preferred z-axis is unsuitable.
`--random-seed` changes the seed (default 0). `--random-generator windows-c`
selects the 15-bit C runtime stream used by the tested Windows C++ build; the
default is Python's generator. The same option is available for individual
surface generation. C++ random streams are platform-dependent, and the
Python and C++ triangulators consume different numbers of random values, so
matching the stream does not guarantee matching later frames or meshes.
`--source-curve-evaluation` on red-blue and individual-surface generation
constructs the source-order Bézier controls and evaluates the spine and its
derivative with de Casteljau's algorithm when rendering canal rings. The
option also uses those controls for frame selection and future-path flush
bounds. It follows the local Windows reference's square-root behavior during
spline construction and may reject inputs whose source-style unscaled
arithmetic is not representable. Frame selection, meshing, and the rest of the
pipeline remain numerical and are not guaranteed to match every C++ build.
`--arc-length-mode source-x87` adds the local Windows reference's sequential
distance arithmetic and ring-height operation order to the source sampling
rule. Combine it with `--source-curve-evaluation` or
`--source-render-vector` to sample source-order Bézier curves. It rejects
distances whose unscaled squared length overflows or underflows to zero; the default
`source` mode uses scale-safe distances instead. When an interval contains no
full sample, the C++ routine reads an uninitialized vector; Python uses the
starting point and therefore cannot reproduce that undefined behavior.
`--source-render-vector` additionally selects the source's unscaled
Bernstein-polynomial and Bézier-clipping check for render-vector selection;
it implies `--source-curve-evaluation`. This is a comparison mode, not a
safety improvement. The source's absolute clipping tolerance can miss a
parallel tangent on small-scale curves. The default uses a scale-normalized
geometric check and can choose a different vector.
`generate-redblue --source-ear-clipping` selects an experimental port of the
source's randomized ear-selection and point-in-polygon tests. It consumes
the selected random stream during cutting and retains Python's polygon
validation and Delaunay cleanup. With `--random-generator windows-c`, this
can improve comparisons with the local Windows reference; source tolerances,
traversal order, retries and floating-point differences can still diverge.
The source's absolute polygon tolerances can also cause this mode to retry or
fail on inputs where the default triangulator succeeds.
`generate-redblue --source-delaunay` additionally uses the source's ordered
floating-point Delaunay edge-flip pass and implies `--source-ear-clipping`.
The source determinant can choose a different diagonal after a large
translation, so this option is intended for source comparisons; the default
uses a filtered exact-sign test. Python still validates the resulting cut
triangles and rejects a nonconvergent source pass.
For closer comparison with the source's iteration-limit behavior, add
`--source-delaunay-allow-limit` to `generate-redblue` or `mesh-boolean`.
This implies `--source-delaunay`; when the limit is reached, Python warns and
keeps the current triangles after checking patch boundary, projected area and
face normals. These checks do not prove that the resulting surface is free of
self-intersections or that its triangulation is Delaunay. The corresponding
Python API option is `source_delaunay_allow_limit=True`, and its warning is
`SourceDelaunayLimitWarning`. Without the option, nonconvergence remains an
error.
`mesh-boolean` and the `redblue_boolean*` APIs accept the same source-ear and
source-Delaunay choices, random seed, and generator. For clockwise projected
cuts, which can occur when retaining inside surfaces, Python reflects the
projection for cleanup so the source's counter-clockwise determinant has the
expected orientation. This adjustment can produce a different triangulation
from C++.

```sh
python -m anamorph mesh-boolean red.obj blue.obj union.obj --source-delaunay --random-generator windows-c --random-seed 1
```
For a local Windows-reference comparison, these options can be combined:

```sh
python -m anamorph generate-redblue examples/branched.swc source-style.obj --force --source-delaunay --source-render-vector --source-frame --source-soma-centroid --precondition-norm-mode source-x87 --arc-length-mode source-x87 --random-generator windows-c --quad-sqrt-mode windows-x87 --render-trig-mode windows-x87 --source-retry-semantics --check-output
```
`--precondition-norm-mode source` uses the C++ operation order for edge
lengths used in preconditioning. `source-x87` additionally uses the tested
Windows reference's square-root rounding; `robust` remains the default. At
extreme scales where the unscaled squared length becomes zero or nonfinite,
source modes fall back to the scaled distance calculation. The option is also available on
analytic analysis, source-job planning and tracing, AMV export, and individual
surface generation. It has no effect when preconditioning is disabled.
`--quad-sqrt-mode windows-x87` on red-blue and individual-surface generation
optionally reproduces the tested Windows reference's extended-precision
square-root rounding when selecting canal and terminal quad diagonals. At
extreme scales it retains the Python scaled-norm fallback. With `--source-frame`,
this also selects the source-style frame normalization's square roots; other
calculations and C++ builds can still differ.
`--render-trig-mode windows-x87` separately uses high-precision sine and
cosine with extended-precision rounding for canal and terminal rendering.
This approximates the local reference's trigonometric library, rather than
reimplementing it exactly; it can still differ by an ulp. Angles outside
[-64, 64] use the host trigonometric functions.
`--source-frame` selects the source's floating-point operation order for
canal and terminal render frames. In the terminal routine the source
normalizes the direction twice for the ring frame and leaves the last axis
unnormalized; this option reproduces that behavior. At extreme scales where
unscaled squared norms cannot be represented, it falls back to Python's
scaled frame calculation. The library option is `source_frame=True`.
These options are for source comparison and do not certify mesh quality.
`--complex-edge-max-growth-factor` controls when the inner splitting
loop gives up and starts another outer attempt. By default, outer retries
continue until the safe radius bound, as in the source. Set `--outer-attempts N`
to impose a per-path limit; difficult inputs can otherwise take a long time.
`--source-retry-semantics` also reproduces the checked-in source's inner-limit
behavior, including an extra radius decrement after an exception on the final
inner attempt and a restart even when that final attempt succeeds. The default
keeps a successful final attempt. This is a source-comparison option and can
make difficult cases slower or prevent convergence.
Repairs to the red mesh are retained across
pre-cut numerical and multiple-loop retries; cut-stage failures restore the
mesh from before the path attempt. These retries remain experimental.
Its preliminary preconditioning is enabled by default; pass
`--no-precondition` to retain the adjusted input samples.
For long, thin input segments, source-style subdivision scoring tests many
candidate partitions and may take substantial time. Library callers can set
`max_splits_per_segment` on `precondition` to limit the number tested.
Add `--require-no-findings` to run the partial analytic checks before meshing
and stop if they report findings or unresolved checks. The CLI prints up to ten
findings and ten unresolved checks to standard error; use `analyze-analytic`
for the full report. Add `--check-adjacent-end-start` as well to include the supplementary check for
consecutive segments. Even when these checks report nothing, their result is
incomplete. The output JSON reports the check status. Python callers can use
`assemble_redblue_checked`, which returns the
mesh and partial report, or raises `PartialAnalysisError` with its `report`
attribute when the checks do not pass.
For legacy source comparisons, `--source-gate` follows the original program's
control flow: it runs the source-selected intersection jobs and generates only
when no job reports a finding and no job is unresolved. Minimum-distance
findings do not block this gate, matching the source's intersection-based
`clean` decision. Add `--require-no-findings` to also reject those findings.
If `--check-adjacent-end-start` is added, its supplementary findings also block
generation even though the original source skips that check.
The original can mark a pair as intersecting when its bivariate solver fails,
without returning a geometric witness. Source-selected analysis traces the
source-style bivariate clipping loop and reports a reproduced solver failure as
a `source_solver_fallback` finding with no candidates. A trace that exceeds
its work limit remains unresolved. The trace intentionally reproduces known
source subdivision defects for comparison; the separate geometric search uses
correct subdivision. Neither path certifies the full consistency analysis.
When the geometric REG, LSI or soma check has no positive candidate, the
source-selected analysis also traces the corresponding source-style
univariate check. Its `source_threshold` candidates reproduce the source's
numerical threshold decision and are not necessarily geometric witnesses; in
particular, the source's soma distance margin can report separated surfaces.
Geometric positives are retained even when the source could miss them.
For example, the checked-in source regularity solver misses the zero-speed
point of x(t) = (t - 0.5)^2, y(t) = z(t) = 0; the separate geometric check
detects it. The trace alone can therefore report a clear source REG job for
that curve.
Source-selected analysis, job boxes, per-job traces, and MorphView export use
source-order spline controls, including the Windows reference's square-root
behavior. This improves comparisons near sensitive solver thresholds; it does
not guarantee identical results on all platforms. Source-style spline
construction rejects inputs whose unscaled arithmetic is not representable.
The same analysis traces the source's global self-intersection check when the
geometric check has no positive candidate. The source's GSI exception handling
differs from its pair check: a bivariate exception alone does not remain a
positive result after its candidate evaluation.
For neurite pairs, the source trace includes boundary and corner candidates as
well as interior roots. A candidate marked `source_threshold` reflects the
source's floating-point decision; `source_root_witness` additionally passed
an exact distance comparison for the represented source Bézier controls.
In particular, the checked-in C++ four-way split constructs its right-hand
polynomial pieces from the left piece. On the included
`tests/fixtures/source_split_missed_root.json` bivariate polynomial example,
the original solver misses an exact root at (0.75, 0.75); the corrected Python clipping search
retains candidate rectangles containing it.
For example:

```sh
python -m anamorph generate-redblue examples/branched.swc source-gated.obj --source-gate --angular-segments 8 --soma-refinements 2 --check-output
```

This is an experimental reproduction of the source's *gate*, not a guarantee
of the source result or of a clean network. The source job selection omits
some soma and adjacent-segment checks, the Python solvers are numerical, and
the full classifier remains incomplete. Preparation also retains Python's
safer handling of roots exactly on the soma surface by default; use
`--source-surface-roots` to reproduce the source's insertion rule. JSON reports `source_jobs_clean` as
true, false, or null (undecided); `complete` remains false. Library callers
can select `analysis_gate="source", analysis_job_mode="source"` on
`assemble_redblue_checked` or `assemble_redblue_checked_to_obj`.
`--analysis-job-mode source` selects the original job-selection rules for that
check. It intentionally skips some soma checks, so the default `extended` mode
is preferable for finding potential problems. `--analysis-source-job-limit`
limits the number of source jobs planned; reaching the limit stops checked
generation because the analysis is unresolved.
`--flush-face-limit 100000` enables experimental partial OBJ flushing during
red-blue assembly. Completed faces outside numerical bounding boxes for the
remaining paths are written to temporary files, while shared boundary vertices
remain in memory. The output is published after assembly succeeds; the JSON
reports the number of flushes and maximum active assembled-face count. The
limit triggers a flush attempt, but geometry near future paths can keep the
active count above the limit. The box bounds and stitching are numerical and
the flushing implementation has only limited comparison coverage. Requesting
`--check-output` or post-processing reloads the complete mesh into memory
after assembly. Library callers can use `assemble_redblue_to_obj` or
`assemble_redblue_checked_to_obj`; these return counts rather than a mesh.
Add `--check-output` to check the generated mesh for boundary/nonmanifold
edges and numerical self-intersection candidates before writing it. When
`--postprocess-output` is used, both the raw and processed meshes are
checked. This floating-point diagnostic does not certify nonintersection.
`--postprocess` applies the current edge-collapse and ten-iteration HC
smoothing stages. They can substantially change surface area and volume;
selected cases have been compared with the original implementation. Use
`--edge-collapse` for the first stage alone, or `--smooth-iterations N` to set
the number of HC smoothing iterations. The command reports percentage changes
in surface area and volume when post-processing is enabled.
`--collapse-alpha`, `--collapse-area-floor`, `--collapse-area-ceiling`, and
`--collapse-depth` set the source-style edge-collapse parameters; the area
factors correspond to source lambda and mu. `--collapse-max-collapses` sets a
Python work limit. `--hc-alpha` and `--hc-beta` set the smoothing weights.
These settings work with partial flushing as well, and the JSON report records
the parameters actually applied. The standalone `postprocess` command accepts
the same collapse controls and its own HC iteration option.
Neighborhood areas are accumulated in the original edge-ordered breadth-first
traversal. As in the C++ implementation, a requested depth includes the next
ring of neighboring faces.
`--collapse-safety source` selects the original topological collapse rule;
`safe` mode also checks face orientation and duplicate faces. The
`--postprocess` preset uses `source`, while an individual `--edge-collapse`
stage defaults to `safe`. The
source rule can produce coincident or inverted triangles—for example, it can
collapse a tetrahedron into two coincident faces. Inspect the resulting mesh;
`generate-redblue --check-output` can run the available numerical checks.
Library callers can select the same rule with `greedy_edge_collapse(...,
safety="source")`. Other numerical and ordering differences may remain.
Pass `--postprocess-output processed.obj` with a post-processing option to save
the processed mesh separately while keeping the raw mesh at the main output
path, matching the source workflow's two outputs.
The Python API also exposes the source's simple umbrella smoother as
`laplacian_smooth`; like HC smoothing, it can change the shape substantially.
For an existing OBJ, `postprocess` can apply edge collapse, simple Laplacian
smoothing, and/or HC smoothing to a separate output and reports area and
volume changes. Run `python -m anamorph postprocess --help` for options.

`generate-individual` corresponds to the source's optional individual modeling
surface output: soma spheres and capped canal paths are exported together as
separate, overlapping components. It is useful for inspecting the surfaces
before stitching; it is not a watertight union of the cell.
Both analytic surface commands accept `--preserve-crease-edges` to generate
aligned canal rings, matching the corresponding source meshing option.
They default to `--arc-length-mode source`, which reproduces a last-interval
overcount in the C++ sampled arc-length calculation; for the C++ routine's
undefined short-interval case Python uses the distance from the starting point.
Use `--arc-length-mode accurate` for conventional sampled arc length instead.
The choice can change ring counts and interpolated radii.

`meshquality` reports minimum, 5th percentile, median, 95th percentile, and
maximum edge lengths and triangle areas, plus aspect-ratio statistics. Edge
lengths use the OBJ's coordinate units, and triangle areas use squared units.
Percentiles use linear interpolation. These measurements help inspect size
variation but do not certify mesh quality or nonintersection.
Mesh volume is meaningful for closed, consistently oriented components;
statistics use a separate local reference point for each component to reduce
rounding error when components are far from the origin or each other.
If a physical area or volume cannot be represented as a float, statistics
report an error instead of returning zero or infinity. The red-blue
`--check-output` topology and intersection checks do not require volume statistics.
`meshquality` also omits volume, so it can report surface sizes when the
enclosed volume is outside float range. Library callers can use
`mesh.statistics(include_volume=False)`, which returns `volume=None`.
`remesh` reports edge-length distributions before and after the operation,
along with percentage changes in area and volume.
`meshcheck` rejects degenerate triangles instead of treating them as a clean
intersection result.

The checked-in C++ analysis skips checks on consecutive end-to-start canal
segments. `analyze-analytic --check-adjacent-end-start` enables an additional
stationary-distance check for those pairs. It excludes a small parameter region
around the shared junction and uses numerical solvers; it can still miss
intersections and does not certify the network as clean.
The report also checks soma spheres from different cells against each other and
does not exempt a root segment's junction against unrelated somas. These checks
extend the source's network analysis and can produce additional findings.

`mesh-boolean` provides experimental red-blue `union`, `intersection`, and
`difference` operations; difference means the first (red) mesh minus the
second (blue) mesh. The operands must be closed, consistently oriented,
outward-facing triangular surfaces. The implementation expects one transverse
intersection loop and retries some complex-edge and affected-face cases.
Disjoint or contained surfaces, multiple loops, and unresolved numerical
edge cases raise an error. Verify the resulting surface before downstream use.
When a single intersection loop exists, untouched disconnected components are
classified against the other operand and included or removed as appropriate.
That classification uses floating-point winding numbers and can raise when
the result is ambiguous.

The same operations are available to Python callers:

```python
from anamorph import read_obj, redblue_boolean_with_retries

red = read_obj("red.obj")
blue = read_obj("blue.obj")
common = redblue_boolean_with_retries(red, blue, operation="intersection")
common.write_obj("common.obj")
```

To install the command-line tool into a Python environment, run
`python -m pip install .`, then use `anamorph inspect ...`, `anamorph analyze ...`, `anamorph generate ...`, or
`anamorph meshstat ...`. The commands emit JSON to standard output and an
attribution notice to standard error.

`analyze` uses sampled values and numerical polynomial stationary points for
near-zero tangents and local curvature, and compares nonadjacent neurite
segments using sampled centerlines. These numerical roots are not certified.
It does not
check soma intersections, adjacent edges, or global self-intersections within
one segment, and sampling can miss problems. No warnings does not mean a clean
morphology. Its pair-count limit can be changed with `--max-pairs`.

`analyze-analytic` is a separate development command using the translated
Bernstein-clipping checks on a subset of the original intersection classes.
It always reports `complete: false` and lists unresolved solver failures; it
does not yet apply the source's complete intersection classification. By
default it normalizes soma samples, adjusts neurite roots, and preconditions;
`--no-precondition` disables the last stage and `--raw` analyzes the input
samples directly.
`analyze-analytic --job-mode source` instead schedules the source's REG, LSI,
GSI, soma-neurite, adjacent and nonadjacent jobs using its subdivided and
padded bounds. It deliberately follows the source's skipped final-segment
soma checks and its root exemption against every soma, and does not add the
extended mode's soma-soma check. Solver arithmetic and some geometry ordering
still differ, so it is a comparison mode rather than exact C++ emulation.
`plan-source-jobs` lists those jobs without running the solvers; its JSON
reports whether the job limit truncated the list. A truncated plan used for
analysis is reported as unresolved and marks all segments as potentially
affected. The plan command uses the source's strict surface-root insertion by
default; `--preserve-surface-roots` selects the safer Python behavior.
`trace-source-jobs` executes the bounded source-style solver for each planned
job and reports its source-style result, candidate values and any failure.
Unlike the conservative analysis, it does not add geometric positives that the
source-style solver misses. `source_jobs_clean` is false when a job is
positive, null when no positive is known but a job or plan is unresolved, and
true when all selected jobs are clear. That value is a numerical comparison,
not a clean-mesh certificate or a guarantee of exact C++ parity. The same
interface is available to Python callers as `trace_source_analysis`.
Each job's `solver_failure` records a source-style bivariate failure, including
one the C++ global self-intersection check suppresses while continuing with
partial candidates. The separate `planned_solver_work_complete` field is
false if a planned solver failed, a job is unresolved, or the plan was
truncated. Consequently, the source-style `source_jobs_clean` can be true
while `planned_solver_work_complete` is false; neither field certifies that
the source selected every check needed for geometric consistency. The
conservative analytic report treats a suppressed global solver failure as
unresolved when that source solve is used.
The trace also reports `segment_flags` for every prepared neurite segment,
including the source-style minimum-distance and intersection flags. These
flags are based on the source-style jobs, so they can differ from the
conservative `analyze-analytic` flags when the geometric solver finds an
intersection the source misses. Each segment lists `solver_issues`; a truncated
job plan marks `source_plan_may_affect` on every segment. A false flag means
no positive was recorded, not that omitted or failed jobs proved it clear.
The library type is `SourceSegmentFlags`.
For a positive soma–neurite job, `classification` records the checked-in
source's intra-cell label, while `soma_relationship` records whether the
soma and neurite actually belong to the same cell. The source applies its
intra-cell flag even when the two belong to different cells.
An isolated neurite root has no segment to analyze and is reported in
`unresolved`; if no positive job was found, the trace reports a null decision
instead of treating an empty job list as a clean analysis.

`python -m anamorph export-amv cell.swc cell.amv` optionally writes the
little-endian MorphView data layout from the C++ source; `python -m anamorph
inspect-amv cell.amv` summarizes it. The original command's call to this writer
is disabled in the checked-in source, and the MorphView viewer is not included.
The format stores only the first soma and seven segment flags, omitting other
analysis results; it cannot certify consistency. Export refuses truncated or
unresolved source-style jobs, including a negative global self-intersection
result after a suppressed bivariate solver failure. Results remain subject to the numerical and
coverage limitations above. Python writes finite path bounds; the tested C++
writer produces infinite path bounds due to its bounding-box initialization.
The tested Windows C++ writer also opens binary data in text mode, expanding
newline bytes. Use `inspect-amv file.amv --windows-text-mode` only for files
known to have that corruption; applying it to ordinary binary can damage data.
The Python API provides `create_morphview`, `write_morphview`, and
`read_morphview`. Interoperability with the unavailable viewer and non-x86
source builds has not been verified.

`--scale-radius` on analytic analysis and red-blue generation scales all radii
after preconditioning, then repositions neurite roots on the resized somas as
in the source. The minimum scale is 0.1; a non-default scale requires preparation
and therefore cannot be combined with `--raw`.
The analytic, red-blue, individual-surface and voxel commands also accept
`--partition`, `--filter-angle` (degrees), `--radius-ratio`, and
`--parametrization` to control path construction. The checked-in C++ code uses
strict comparisons for its branch angle and radius-ratio filters.

The minimum-distance diagnostics compare squared lengths and radius thresholds
using the exact rational values represented by the input floats. This avoids
false decisions from overflow and can differ from C++ floating-point arithmetic
at a threshold. In the Python `minimum_distance_violations()` result, a length
too large for a float is reported as infinity along with its rational
`exact_length_squared` pair.

Positive analytic findings include candidate parameter values, their source,
and a measured value: squared distance for intersection candidates, squared
speed for regularity, and the local self-intersection polynomial for LSI.
When a candidate's squared distance, squared speed or local-curvature
polynomial value underflows or overflows a float, the JSON candidate has
`value: null` and `exact_value: [numerator, denominator]`.
These integers can be large, so use an arbitrary-precision JSON parser when
their exact value matters. In the curve API the corresponding fields are
`distance_squared` and `exact_distance_squared`; the soma and local-curvature
APIs without bounds return a Python `Fraction` in this case. The reported
value corresponds to the stored float coefficients and candidate parameters,
except for the interval-certified tangency described below. It is not a guarantee
that numerical clipping found every candidate or that an
approximate regularity candidate is an exact zero.
Candidates also retain their solver intervals or rectangles. Candidate
parameters and bounds are generally floating-point results, not certified exact
roots or enclosures. For the exact straight-segment minimum fallback,
`exact_parameters` contains a numerator and denominator for each parameter
when its displayed float is rounded; use those ratios to reproduce its measured
distance. Nonzero polynomials are scaled before numerical clipping to reduce sensitivity to the
units of the input; this changes numerical behavior relative to unscaled C++.
Parallel straight centerlines with monotone cubic parameterizations also use
exact projected intervals. An overlap can be reported as a rational witness;
for `source: "parallel_line_tangency"`, `parameter_bounds` instead enclose a
proven witnessing parameter that may be irrational. The displayed parameters
are interval approximations, and the reported squared distance is the proven
minimum at that parameter, not the distance evaluated at the displayed midpoint.
These exact tests apply to the represented float coefficients and the
maximum-radius bounding pipes, not necessarily to the varying-radius surfaces.
The Bézier clippers also apply a small scale-dependent roundoff allowance even
when `eps=0`; their intervals and rectangles are numerical candidates, not
certified root enclosures, and may include extra candidates near zero.
For regularity and local self-intersection, derivatives are also scaled before
forming polynomial products. Physical values outside the float range are
retained as exact rational numbers rather than zero or infinity.
Regularity also checks the exact common roots of the represented cubic
derivatives. A reported `exact_root_interval` encloses a proven zero-speed
parameter, even if normalization or clipping fails; that numerical job still
remains unresolved on failure. Its displayed parameter and squared speed are
evaluated at the interval midpoint, which may not be the exact root. Numerical
`root_interval` candidates retain their original, uncertified meaning.
If an interior or boundary solve cannot finish, the other independent solves
still run. Any overlap candidates they find are reported along with the
unresolved job. The corresponding curve-level functions raise
`IntersectionSolveError`, a `ClippingConvergenceError` subclass whose
`candidates` and `failures` attributes retain that partial information.
If clipping reaches its work limit after completing some candidate intervals
or rectangles, those numerical candidates are also retained and checked;
the unfinished search is still reported unresolved. Direct clipping callers
can inspect `ClippingConvergenceError.partial_roots`, `processed`, and
`pending`. These are candidate regions, not certified root enclosures.
`analyze-analytic --max-intervals N` sets the univariate work limit per
solve; `generate-redblue --require-no-findings --analysis-max-intervals N`
applies it to the pre-meshing checks. An unresolved check stops checked mesh
generation. The corresponding Python APIs accept `max_intervals` or
`analysis_max_intervals` respectively.
Exactly parallel affine segment pairs use a direct closest-point calculation,
and that calculation also handles an affine pair if bivariate clipping fails.
It compares exact rational representations of the input float coefficients;
exact parameters or squared distances are retained when the float values
cannot represent the witness. Unresolved edge searches remain reported even
when the direct affine calculation supplies a positive witness.
Two cubic segments that describe overlapping portions of the same curve in
the same or opposite directions, with a translation exactly perpendicular to
every nonconstant coefficient vector, also have an exact minimum distance.
This handles an otherwise non-isolated root system. The parameterizations must
be related by an exact rational affine change, including possible reversal and
speed change. These checks use the represented input floats and do not assume
that nearly matching coefficients are identical.
Parallel straight paths also use an exact minimum when one is affine and the
other is a monotone cubic parameterization of its line. The check proves the
line and monotonicity conditions on the represented coefficients; other cases
continue through numerical clipping.
This is an additional numerical path, not a port of the C++ clipping routine.
For soma-neurite, curve-pair and global self-intersection candidates, the reported root
intervals are searched for a closer witness rather than relying solely on
their midpoint. A candidate whose interval cannot be separated from the
distance threshold is marked unresolved if refinement cannot decide it.
Soma-neurite acceptance compares the witness distance with the squared sum of
the radii using exact rational representations of the supplied floats; it does
not add a fixed stationary-polynomial tolerance to that geometric threshold.
Curve-pair and global self-intersection systems are scaled before polynomial
products as well. If their coefficient ratios or reported squared distances
exceed the usable float range, the affected work is marked unresolved; other
independently verified witnesses are retained when available.
The intervals themselves come from floating-point clipping and are not
certified enclosures; this refinement does not make the overall analysis a
proof of nonintersection. Refinement uses relative coordinates to reduce
cancellation at large absolute positions; it cannot recover detail already
lost when the input coordinates were rounded.
The report also groups positive findings into source-style per-segment flags,
and records unresolved jobs on the affected segments. A false flag only means
no positive finding was reported; it does not establish that the segment is
clean. `pair_limit_reached` indicates when the pair-work limit stopped analysis,
and `first_untested_pair` identifies the first pair omitted. Per-segment
`pair_limit_may_affect` flags conservatively identify segments whose pair
checks may have been skipped; a flagged segment can have no omitted partner.
In source mode, `source_job_counts` counts the planned jobs, not necessarily
the jobs executed: a pair-work limit can stop execution early. If the source
plan itself hits its limit, `source_plan_truncated` is true and all segment
flags conservatively report that the incomplete plan may affect them.

For explicit `generate-voxel` use, the required `--spacing` is expressed in
the same units as SWC coordinates.
Use a spacing much smaller than the thinnest radius and check the resulting
geometry; resolution requirements depend on the morphology. `--max-grid-points`
limits grid size and can be raised explicitly. `--adjust-soma-roots`,
`--precondition`, `--edge-collapse` and `--smooth-iterations` are opt-in because they alter
the input geometry. Root adjustment follows the single-sphere procedure but
does not merge multi-sample somata.

The voxel mesher can produce very poor triangle aspect ratios, particularly
near grid vertices; the optional edge-collapse cleanup may not remove all of
them. Grid samples very close to the surface are shifted slightly away from
zero to reduce coincident crossings; this does not guarantee good triangles or
preservation of arbitrarily small features. Meshing raises an error if it
detects a zero-area triangle or inadequate coordinate precision. Run `meshstat` and use an appropriate downstream mesh
repair or remeshing tool when triangle quality matters.

`remesh --isotropic` targets a more uniform edge length by splitting long edges, collapsing
short edges, and flipping edges where this improves triangle shape. It first
applies a poor-triangle cleanup to help remove nearly coincident vertices. It requires
a closed, edge-manifold triangular OBJ. Its target is a goal rather than a
guarantee: topology and local orientation checks can leave edges outside the
target range. If a midpoint collapse fails local checks, other positions on
the edge may be tried with a local quality and intersection check. This check
does not cover distant faces. The algorithm alters the surface and does not
project it back onto the input or test global self-intersections. Compare area, volume, and geometry before
using its output. `generate --uniform-edge-length 0.09` applies the same pass
at the end of generation.

`meshstat` accepts triangular OBJ faces and computes absolute signed volume,
just as the C++ utility does. Volume is meaningful as enclosed volume only for
a closed, consistently oriented surface. Boundary and non-manifold edge counts
are additional diagnostics, not a complete mesh-validity test. A single-face
mesh has an undefined sample standard deviation, represented as JSON `null`.
`meshcheck` reports triangle intersection candidates. By default it excludes
face pairs sharing a vertex. `--include-adjacent` also checks for overlaps
that extend beyond a shared vertex or edge. Both modes use floating-point
predicates, so an empty result is not a proof of non-self-intersection.

This work is based on the original AnaMorph code and is distributed under its
license and additional attribution terms; see [LICENSE](../LICENSE) and
[NOTICE.md](../NOTICE.md).

Recommended citations:

- Mörschel K, Breit M, Queisser G. Generating neuron geometries for detailed
  three-dimensional simulations using AnaMorph. Neuroinformatics (2017).
- Grein S, Stepniewski M, Reiter S, Knodel MM, Queisser G. 1D-3D hybrid
  modelling – from multi-compartment models to full resolution models in space
  and time. Frontiers in Neuroinformatics 8, 68 (2014).
- Breit M, Stepniewski M, Grein S, Gottmann P, Reinhardt L, Queisser G.
  Anatomically detailed and large-scale simulations studying synapse loss and
  synchrony using NeuroBox. Frontiers in Neuroanatomy 10 (2016).
