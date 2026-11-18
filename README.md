# AnaMorph Python

Based on AnaMorph (https://github.com/NeuroBox3D/AnaMorph).

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

By default, postprocessing applies source-style greedy edge collapse (alpha 1.5,
lambda 0.125, mu 0.5, depth 5), followed by ten HC smoothing iterations (alpha
0.4, beta 0.7). These use the original default parameters and topological rule;
the Python implementation has a 100,000-collapse work limit and may differ
numerically. Postprocessing can substantially change the surface: compare the
raw and processed meshes and check the area and volume in the report.

To use isotropic remeshing instead, select it explicitly:

```sh
python -m anamorph workflow examples/branched.swc --output-dir uniform_results --isotropic --target-edge-length 0.14
```

The target is in SWC coordinate units. If omitted for `--isotropic`, the median
input neurite radius is used as a starting heuristic. `--iterations` controls
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
python -m anamorph remesh raw.obj uniform.obj --isotropic --target-edge-length 0.14
python -m anamorph meshstat processed.obj
python -m anamorph meshquality processed.obj
python -m anamorph meshcheck processed.obj --include-adjacent
python -m anamorph --help
```

`generate` uses the analytic mesher and `generate-redblue` is an alias. A
voxel-based approximation is available only when explicitly requested with
`generate-voxel`; there is no automatic fallback to it. The standalone `remesh`
command retains its input and uses the same default postprocessing as the
workflow. For Python callers, use `anamorph.workflow.run_workflow`; pass
`postprocess_method="isotropic"` to select the optional isotropic method.

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
