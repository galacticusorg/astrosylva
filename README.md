# astrosylva

> *Carry merger trees from the forest to Galacticus.*

A Python library and CLI that converts halo merger-tree catalogues into the
[Galacticus](https://github.com/galacticusorg/galacticus) HDF5 input format.

Supports:

- **Consistent-Trees** (Rockstar pipeline; `forests.list` + `locations.dat` + `tree_*.dat`)
- **LHaloTree** (Millennium-style binary)
- **SubLink** (IllustrisTNG-style HDF5)
- **AHF** (Amiga Halo Finder MergerTree output)

This package is a Python port of the original C utility `rockstar2galacticus`.
The legacy C sources are preserved on the `rockstar2galacticus` branch of
this repository for reference; new development happens on `main` in
`src/astrosylva/`.

## Install

```bash
pip install astrosylva
```

Or, from a checkout:

```bash
pip install -e ".[dev]"
```

## Quickstart

```bash
astrosylva convert config.yaml
```

A minimal config:

```yaml
reader:
  name: consistent_trees
  source:
    input_path: ./Illustris-3/tree/trees/
    forests_path: ./Illustris-3/tree/forests.list
    locations_path: ./Illustris-3/tree/locations.dat
  options:
    host_source: pid           # or "upid"
    scale_radius_source: rs    # or "rs_klypin"

writer:
  output_path: galacticus_illustris3.hdf5

metadata:
  cosmology:
    HubbleParam: 0.704
    Omega0: 0.2726
    OmegaLambda: 0.7274
    OmegaBaryon: 0.0456
    sigma_8: 0.809
  simulation:
    boxSize: 75000.0
    simulationCode: AREPO
    startRedshift: 127.0
  groupFinder:
    groupFinderCode: rockstar
    minimumParticleNumber: 10
  haloTrees:
    haloMassesIncludeSubhalos: 1
    forestsAreSelfContained: 1
    treesHaveSubhalos: 1
    velocitiesIncludeHubbleFlow: 0
```

Values that the reader can introspect from the input data (e.g. cosmological
parameters from a Consistent-Trees header) override the `metadata` block;
conflicts emit a `MetadataConflictWarning` showing both values.

## CLI

| Command                               | What it does                                  |
|---------------------------------------|-----------------------------------------------|
| `astrosylva convert <config.yaml>`    | Run the full conversion.                      |
| `astrosylva readers`                  | List discovered readers (entry-point plugins).|
| `astrosylva validate <config.yaml>`   | Parse and validate the config without writing.|

## Implementation status

| Reader            | Status        | Notes                                          |
|-------------------|---------------|------------------------------------------------|
| Consistent-Trees  | Implemented   | Full port; column lookup is name-based.        |
| AHF               | Experimental  | `.AHF_halos` + `.AHF_mtree`; union-find forests. |
| SubLink           | Experimental  | HDF5; multi-chunk, FOF host resolution.        |
| LHaloTree         | Experimental  | Millennium binary; multi-chunk; some proxies.  |

The Galacticus writer emits `formatVersion = 2`, matching the legacy C tool's
output layout byte-for-byte (group structure, dataset names, attribute names).

## License

MIT — see [`LICENSE`](LICENSE).

## Contributing

Bug reports, feature requests, and pull requests welcome. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the dev-loop guide and
[`CONTRIBUTORS.md`](CONTRIBUTORS.md) for credits.
