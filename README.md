# AnaMorph Python

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).

This project is a Python version derived from the existing
[AnaMorph](https://github.com/NeuroBox3D/AnaMorph) software, originally written
in C++.

AnaMorph Python generates triangular neuron surface meshes from SWC morphology
files. It uses analytic canal surfaces and red-blue surface stitching by default,
and can save both the original mesh and a postprocessed mesh with a quality
report. This is an experimental Python implementation; it does not reproduce
every feature or guarantee of the original AnaMorph software.

## Install

Python 3.10 or later is required. The package has no runtime dependencies.
From the repository directory:

```sh
python -m pip install .
```

Alternatively, run `python -m anamorph` from the repository directory without
installing the package. The examples below work with either approach.

## Generate a mesh

```sh
python -m anamorph workflow examples/branched.swc --output-dir cell_results
```

The workflow saves `input.swc`, `raw.obj`, `remeshed.obj`, `report.md`, and
`report.json`. It will not overwrite existing result files. The report compares
mesh topology, edge lengths, surface area, volume, and numerical intersection
checks; the JSON file includes the settings and checksums.

By default, the workflow generates the analytic raw mesh, applies isotropic
remeshing, runs ten Taubin smoothing iterations, and then applies the experimental
local `mortar` fill to soma–neurite junctions. If no target edge length is
specified, the median input neurite radius is used as a starting heuristic.
Postprocessing can substantially change the surface: compare the raw and
processed meshes and inspect area, volume, and intersection checks in the report.

The soma-junction blend adds a small outward fillet around each primary neurite
attachment to make the soma-to-neurite transition less abrupt. The blend is an
approximate geometric modification, not a reconstruction of missing anatomy.
Its width defaults to three local neurite radii and is capped at half the soma
radius. Use `--soma-blend-width` to adjust the factor, or set it to `0` to
disable this stage. The report records how many junctions were adjusted; the
raw OBJ is saved before blending. Soma-free components have no soma junction to
blend. Inspect the resulting mesh, particularly near closely spaced roots.

The experimental `mortar` method fills and relaxes the attachment region after
remeshing and smoothing, allowing local volume to increase:

```sh
python -m anamorph workflow path/to/cell.swc --output-dir mortar_results --target-edge-length 0.4
```

This is an approximate geometric adjustment. It does not guarantee that every
cross-section becomes wider or prevent intersections; compare it with the raw
mesh and inspect the report. The earlier `blend` method remains available via
`--soma-blend-method blend`; it runs before the selected smoothing stage.

To use source-style edge collapse, local blending and HC smoothing instead:

```sh
python -m anamorph workflow path/to/cell.swc --output-dir standard_results --standard-postprocess
```

Taubin smoothing alternates a positive and a negative Laplacian pass in each
iteration. Its defaults are 10 iterations, lambda 0.5 and mu -0.53. Use
`--taubin-iterations`, `--taubin-lambda` and `--taubin-mu` to change them.
The report records the applied parameters. It changes vertex positions and
does not enforce volume preservation or prevent self-intersections. Use
`--smoothing none` to omit the global smoothing pass or `--soma-blend-width 0`
to omit the local fill.

To choose a target edge length explicitly:

```sh
python -m anamorph workflow examples/branched.swc --output-dir uniform_results --target-edge-length 0.14
```

The target is in SWC coordinate units. If omitted, the median input neurite
radius is used as a starting heuristic. `--iterations` controls
isotropic passes. Isotropic remeshing changes the surface and does not guarantee
uniform triangles.

Soma-free neuron components are supported when their root is a neurite (SWC type
2, 3, or 4) with at least one outgoing segment:

```sh
python -m anamorph workflow examples/no_soma.swc --output-dir no_soma_results
```

An unbranched neurite root gets a flat cap; an immediately branching root gets
a rounded junction using its recorded radius. These do not reconstruct a missing
soma. Forests may combine soma-containing and soma-free components.

The workflow checks the output meshes numerically, but its checks do not certify
non-self-intersection or suitability for simulation. Use `--analyze` for the
optional, incomplete pre-meshing analysis, or `--skip-intersection-check` to omit
the potentially expensive intersection checks. Inspect meshes before use.

## Other commands

```sh
python -m anamorph generate examples/branched.swc raw.obj
python -m anamorph remesh raw.obj processed.obj
python -m anamorph remesh raw.obj taubin_postprocessed.obj --smoothing taubin
python -m anamorph postprocess raw.obj taubin_only.obj --taubin-iterations 10
python -m anamorph remesh raw.obj uniform.obj --isotropic --target-edge-length 0.14
python -m anamorph meshstat processed.obj
python -m anamorph meshquality processed.obj
python -m anamorph meshcheck processed.obj --include-adjacent
python -m anamorph --help
```

`generate` uses the analytic mesher and `generate-redblue` is an alias. A
voxel-based approximation is available only when explicitly requested with
`generate-voxel`; there is no automatic fallback to it. The standalone `remesh`
command retains its input and uses edge collapse and HC smoothing by default.
Soma-junction blending requires morphology information and is applied by
`workflow`, not by the OBJ-only `remesh` or `postprocess` commands.
`postprocess --taubin-iterations 10` applies Taubin smoothing alone,
without edge collapse. For Python callers, use `anamorph.taubin_smooth` to
smooth an existing `Mesh` or `anamorph.workflow.run_workflow` with
the default isotropic, Taubin, and mortar pipeline. Pass
`postprocess_method="standard"` to select source-style postprocessing.

See the [technical reference](docs/technical-reference.md) for additional
commands and implementation details. Command-specific options are available via
`python -m anamorph <command> --help`.

## Tests

```sh
python -m unittest discover -s tests
```

Some historical comparison tests require optional Windows C++ reference
executables. They skip when those local tools are unavailable; the Python tests
do not require them.

## License and attribution

This project is derived from AnaMorph, copyright 2013–2017 G-CSC, Goethe
University Frankfurt – Queisser group. It is distributed under LGPL version 3
with the original additional attribution terms. See [LICENSE](LICENSE),
[COPYING](COPYING), and [NOTICE.md](NOTICE.md) for the license and notices.

Recommended citations:

- Mörschel K, Breit M, Queisser G. Generating neuron geometries for detailed three-dimensional simulations using AnaMorph. Neuroinformatics (2017).
- Grein S, Stepniewski M, Reiter S, Knodel MM, Queisser G. 1D-3D hybrid modelling – from multi-compartment models to full resolution models in space and time. Frontiers in Neuroinformatics 8, 68 (2014).
- Breit M, Stepniewski M, Grein S, Gottmann P, Reinhardt L, Queisser G. Anatomically detailed and large-scale simulations studying synapse loss and synchrony using NeuroBox. Frontiers in Neuroanatomy 10 (2016).
