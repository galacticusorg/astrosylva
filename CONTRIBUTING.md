# Contributing to astrosylva

Thanks for taking a look. This project converts halo merger-tree
catalogues into the [Galacticus](https://github.com/galacticusorg/galacticus)
HDF5 input format. Contributions of all sizes are welcome — bug
reports, format-coverage additions, performance work, documentation.

## Development setup

```bash
git clone https://github.com/galacticusorg/rockstar2galacticus.git
cd rockstar2galacticus
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The dev extra installs `pytest`, `ruff`, and `mypy`.

## The dev loop

Run before pushing:

```bash
ruff check . && ruff format --check .
mypy src
pytest
```

All three are enforced in CI. The unit test suite runs in under a
second; ytree integration tests skip silently unless
`ASTROSYLVA_YTREE_DATA` is set (see [`docs/integration.rst`](docs/integration.rst)).

## Repository layout

| Path | What lives there |
|---|---|
| `src/astrosylva/readers/` | One module per merger-tree format. Each subclasses `TreeReader`. |
| `src/astrosylva/writers/` | Only one for now: `GalacticusWriter`. |
| `src/astrosylva/schema.py` | The canonical `HALO_DTYPE`, `Forest`, `Metadata` definitions every reader and the writer must speak. |
| `src/astrosylva/cli.py` | The `astrosylva` entry-point (`convert`, `readers`, `validate`). |
| `tests/` | Per-reader tests plus `test_cross_reader.py` (four-way equivalence) and `test_streaming.py` (memory contracts). |
| `docs/` | Sphinx skeleton + the user-facing reference. |

The original C tool that astrosylva replaces is preserved on the
[`rockstar2galacticus`](https://github.com/galacticusorg/rockstar2galacticus/tree/rockstar2galacticus)
branch.

## Adding a new reader

1. Create `src/astrosylva/readers/myformat.py` with a class that
   subclasses `TreeReader` and implements `metadata`, `__len__`,
   and `__iter__`. See `consistent_trees.py` for the canonical
   shape.
2. Register the entry point in `pyproject.toml` under
   `[project.entry-points."astrosylva.readers"]`.
3. Add a unit test file `tests/test_myformat.py` with a tiny
   hand-built fixture (see how the other readers do it in
   `tests/conftest.py`).
4. Add `myformat` to `_ALL_READER_BUILDERS` in
   `tests/test_cross_reader.py` so it participates in the
   four-way equivalence check.
5. Document any format-specific options in
   [`docs/readers.rst`](docs/readers.rst).

Readers must convert to the canonical units (Mpc/h, km/s,
M☉/h, dimensionless `a`) and follow the Galacticus
`hostIndex == nodeIndex` convention for halos with no host. If
your format has format-specific conventions for `/forestHalos`
attributes, return them from `defaults()`; reader-introspected
values (e.g. cosmology from a file header) belong in
`metadata()`.

## Pull-request style

- Small, focused commits over giant ones. The Git history reads
  top-down as a design document — please keep it readable.
- Commit messages explain *why*. The diff already shows *what*.
- One feature or bug fix per PR where practical.

## License

By contributing, you agree your contributions are licensed
under the [MIT License](LICENSE), the same as the rest of the
project.
