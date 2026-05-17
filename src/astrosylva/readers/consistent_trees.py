"""Consistent-Trees reader.

Inputs (the Rockstar / Consistent-Trees pipeline):

- ``input_path``     : directory containing the ``tree_*.dat`` files
- ``forests_path``   : ``forests.list`` (tree_root_id  forest_id  [weight])
- ``locations_path`` : ``locations.dat`` (tree_root_id  file_id  offset  filename)

Column lookup is by *name* (parsed from the ``#scale(0) id(1) ...`` header
line) rather than by hardcoded index — this fixes a latent fragility in the
legacy C tool, which silently broke if a column was added or reordered.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, TextIO

import numpy as np

from astrosylva.exceptions import ReaderError
from astrosylva.readers.base import ReaderSource, TreeReader
from astrosylva.schema import DEFAULT_UNITS, HALO_DTYPE, Forest, Metadata


@dataclass
class _ForestEntry:
    forest_id: int
    weight: float


@dataclass
class _TreeRef:
    tree_root_id: int
    forest_id: int
    forest_weight: float
    file_id: int
    offset: int
    file_path: Path


@dataclass
class _ForestRef:
    forest_id: int
    weight: float
    trees: list[_TreeRef] = field(default_factory=list)


# Column name -> dtype field name. Mass / scale-radius source keys are
# resolved at runtime based on reader options.
_FIXED_COLUMN_MAP: dict[str, str] = {
    "scale": "expansionFactor",
    "id": "nodeIndex",
    "desc_id": "descendantIndex",
    "mvir": "nodeMass",
    "x": "position.0",
    "y": "position.1",
    "z": "position.2",
    "vx": "velocity.0",
    "vy": "velocity.1",
    "vz": "velocity.2",
    "Jx": "angularMomentum.0",
    "Jy": "angularMomentum.1",
    "Jz": "angularMomentum.2",
    "Spin": "spin",
}

# Consistent-Trees emits scale radius in *kpc/h* even though all other length
# quantities are Mpc/h. Galacticus expects Mpc/h throughout.
_RS_KPC_TO_MPC = 1.0 / 1000.0

_HEADER_COL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(\d+\)$")


def _parse_header_columns(line: str) -> dict[str, int]:
    """Parse the ``#scale(0) id(1) desc_scale(2) ...`` header line."""
    tokens = line.lstrip("#").split()
    cols: dict[str, int] = {}
    for tok in tokens:
        m = _HEADER_COL_RE.match(tok)
        if m is None:
            continue
        cols[m.group(1)] = int(tok[tok.index("(") + 1 : tok.index(")")])
    return cols


def _parse_cosmology_header(lines: list[str]) -> dict[str, float]:
    """Extract cosmological parameters from CT header comments.

    Recognised tokens (case-insensitive variants accepted): ``Omega_M``,
    ``Omega_L``, ``Omega_b`` / ``Omega_B``, ``h0`` / ``h``, ``sigma_8``
    / ``sigma8``. Mapped to the Galacticus attribute names ``Omega0``,
    ``OmegaLambda``, ``OmegaBaryon``, ``HubbleParam``, ``sigma_8``.
    Anything we don't recognise is silently ignored; users can still
    fill those gaps via ``metadata.cosmology``.
    """
    out: dict[str, float] = {}
    pattern = re.compile(r"\b(Omega_M|Omega_L|Omega_[bB]|h0|h|sigma_?8)\s*=\s*([0-9.eE+-]+)")
    key_map = {
        "Omega_M": "Omega0",
        "Omega_L": "OmegaLambda",
        "Omega_b": "OmegaBaryon",
        "Omega_B": "OmegaBaryon",
        "h0": "HubbleParam",
        "h": "HubbleParam",
        "sigma_8": "sigma_8",
        "sigma8": "sigma_8",
    }
    for line in lines:
        for key, value in pattern.findall(line):
            out[key_map[key]] = float(value)
    return out


def _parse_box_size(lines: list[str]) -> float | None:
    pattern = re.compile(r"Full box size\s*=\s*([0-9.eE+-]+)\s*Mpc")
    for line in lines:
        m = pattern.search(line)
        if m:
            return float(m.group(1))
    return None


class ConsistentTreesReader(TreeReader):
    """Reader for the Consistent-Trees output of the Rockstar pipeline."""

    name: ClassVar[str] = "consistent_trees"
    aliases: ClassVar[tuple[str, ...]] = ("consistent-trees", "ctrees")

    def __init__(self, source: ReaderSource, options: dict[str, Any] | None = None) -> None:
        super().__init__(source, options)
        self._input_path = Path(source.require("input_path"))
        self._forests_path = Path(source.require("forests_path"))
        self._locations_path = Path(source.require("locations_path"))
        self._host_source: str = self.options.get("host_source", "pid")
        self._scale_radius_source: str = self.options.get("scale_radius_source", "rs")
        if self._host_source not in ("pid", "upid"):
            raise ReaderError(f"host_source must be 'pid' or 'upid', got {self._host_source!r}")
        if self._scale_radius_source not in ("rs", "rs_klypin"):
            raise ReaderError(
                "scale_radius_source must be 'rs' or 'rs_klypin', got "
                f"{self._scale_radius_source!r}"
            )
        self._tree_index: list[_TreeRef] | None = None
        self._forest_index: list[_ForestRef] | None = None
        self._header_cache: dict[Path, dict[str, int]] = {}
        self._cosmo_cache: dict[str, float] | None = None
        self._box_size_cache: float | None = None

    # ------------------------------------------------------------ public API

    def metadata(self) -> Metadata:
        self._ensure_indexed()
        cosmo = dict(self._cosmo_cache or {})
        sim: dict[str, Any] = {}
        if self._box_size_cache is not None:
            sim["boxSize"] = self._box_size_cache * 1000.0  # Mpc/h -> kpc/h
        return Metadata(
            cosmology=cosmo,
            units=dict(DEFAULT_UNITS),
            simulation=sim,
        )

    def __len__(self) -> int:
        self._ensure_indexed()
        assert self._forest_index is not None
        return len(self._forest_index)

    def __iter__(self) -> Iterator[Forest]:
        self._ensure_indexed()
        assert self._forest_index is not None
        for forest_ref in self._forest_index:
            yield self._load_forest(forest_ref)

    # ----------------------------------------------------------- indexing

    def _ensure_indexed(self) -> None:
        if self._forest_index is not None:
            return
        tree_to_forest = self._read_forests_list()
        trees = self._read_locations(tree_to_forest)
        self._tree_index = trees
        self._forest_index = self._group_trees_into_forests(trees)
        self._cache_header_metadata(trees)

    def _read_forests_list(self) -> dict[int, _ForestEntry]:
        out: dict[int, _ForestEntry] = {}
        with self._forests_path.open() as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                try:
                    tree_root_id = int(parts[0])
                except ValueError:
                    continue  # header line
                forest_id = int(parts[1])
                weight = float(parts[2]) if len(parts) >= 3 else 1.0
                out[tree_root_id] = _ForestEntry(forest_id, weight)
        return out

    def _read_locations(self, tree_to_forest: dict[int, _ForestEntry]) -> list[_TreeRef]:
        trees: list[_TreeRef] = []
        with self._locations_path.open() as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    tree_root_id = int(parts[0])
                except ValueError:
                    continue  # header line
                entry = tree_to_forest.get(tree_root_id)
                if entry is None:
                    raise ReaderError(
                        f"tree_root_id {tree_root_id} in locations.dat has no "
                        "matching entry in forests.list"
                    )
                trees.append(
                    _TreeRef(
                        tree_root_id=tree_root_id,
                        forest_id=entry.forest_id,
                        forest_weight=entry.weight,
                        file_id=int(parts[1]),
                        offset=int(parts[2]),
                        file_path=self._input_path / parts[3],
                    )
                )
        return trees

    @staticmethod
    def _group_trees_into_forests(trees: list[_TreeRef]) -> list[_ForestRef]:
        forests: dict[int, _ForestRef] = {}
        for t in trees:
            fr = forests.get(t.forest_id)
            if fr is None:
                fr = _ForestRef(forest_id=t.forest_id, weight=t.forest_weight, trees=[])
                forests[t.forest_id] = fr
            fr.trees.append(t)
        return list(forests.values())

    def _cache_header_metadata(self, trees: list[_TreeRef]) -> None:
        if not trees:
            return
        first_file = trees[0].file_path
        header_lines: list[str] = []
        with first_file.open() as fh:
            for _ in range(80):
                line = fh.readline()
                if not line or not line.startswith("#"):
                    break
                header_lines.append(line)
        self._cosmo_cache = _parse_cosmology_header(header_lines)
        self._box_size_cache = _parse_box_size(header_lines)

    def _column_map(self, file_path: Path) -> dict[str, int]:
        if file_path in self._header_cache:
            return self._header_cache[file_path]
        with file_path.open() as fh:
            header_line: str | None = None
            for line in fh:
                if line.startswith("#") and "(0)" in line and "id(" in line:
                    header_line = line
                    break
        if header_line is None:
            raise ReaderError(f"Could not find column header in {file_path}")
        cols = _parse_header_columns(header_line)
        self._header_cache[file_path] = cols
        return cols

    # ----------------------------------------------------------- loading

    def _load_forest(self, forest_ref: _ForestRef) -> Forest:
        """Materialise one forest, opening each tree file at most once.

        Trees in a forest commonly live in the same ``tree_*.dat``
        file; iterating per-tree would reopen it N times. We group
        trees by file, open each file once, read its trees in offset
        order (sequential I/O), then restore the original
        ``forests.list`` order in the concatenated output.
        """
        if not forest_ref.trees:
            return Forest(
                forest_id=forest_ref.forest_id,
                halos=np.empty(0, dtype=HALO_DTYPE),
                weight=forest_ref.weight,
            )

        trees_by_file: dict[Path, list[tuple[int, _TreeRef]]] = {}
        for i, tree in enumerate(forest_ref.trees):
            trees_by_file.setdefault(tree.file_path, []).append((i, tree))

        indexed_halos: list[tuple[int, np.ndarray]] = []
        for file_path, indexed_trees in trees_by_file.items():
            # Process in offset order to keep file reads sequential.
            indexed_trees.sort(key=lambda it: it[1].offset)
            with file_path.open() as fh:
                for i, tree in indexed_trees:
                    indexed_halos.append((i, self._load_tree(fh, tree)))

        # Restore original tree order before concatenating.
        indexed_halos.sort(key=lambda ih: ih[0])
        halos_per_tree = [halos for _, halos in indexed_halos]
        halos = np.concatenate(halos_per_tree)
        return Forest(
            forest_id=forest_ref.forest_id,
            halos=halos,
            weight=forest_ref.weight,
        )

    def _load_tree(self, fh: TextIO, tree: _TreeRef) -> np.ndarray:
        """Parse one tree's halos from an already-open file handle.

        Caller is responsible for opening / closing ``fh``; this method
        seeks to ``tree.offset`` and reads forward until the next ``#``
        line or EOF.
        """
        cols = self._column_map(tree.file_path)
        try:
            i_scale = cols["scale"]
            i_id = cols["id"]
            i_desc_id = cols["desc_id"]
            i_mvir = cols["mvir"]
            i_x, i_y, i_z = cols["x"], cols["y"], cols["z"]
            i_vx, i_vy, i_vz = cols["vx"], cols["vy"], cols["vz"]
            i_jx, i_jy, i_jz = cols["Jx"], cols["Jy"], cols["Jz"]
            i_spin = cols["Spin"]
        except KeyError as exc:
            raise ReaderError(
                f"Tree file {tree.file_path} is missing required column {exc}"
            ) from exc
        i_host = cols[self._host_source]
        rs_key = "rs_klypin" if self._scale_radius_source == "rs_klypin" else "rs"
        if rs_key not in cols:
            raise ReaderError(
                f"Tree file {tree.file_path} has no {rs_key!r} column; "
                "set options.scale_radius_source accordingly."
            )
        i_rs = cols[rs_key]

        # Read raw rows starting at the tree's byte offset, stopping at the
        # next ``#`` line or EOF.
        rows: list[list[str]] = []
        fh.seek(tree.offset)
        for line in fh:
            if not line.strip():
                continue
            if line.startswith("#"):
                break
            rows.append(line.split())
        if not rows:
            return np.empty(0, dtype=HALO_DTYPE)

        n = len(rows)
        halos = np.empty(n, dtype=HALO_DTYPE)
        for k, row in enumerate(rows):
            halos["nodeIndex"][k] = int(row[i_id])
            halos["descendantIndex"][k] = int(row[i_desc_id])
            host = int(row[i_host])
            halos["hostIndex"][k] = host
            halos["expansionFactor"][k] = float(row[i_scale])
            halos["nodeMass"][k] = float(row[i_mvir])
            halos["scaleRadius"][k] = float(row[i_rs]) * _RS_KPC_TO_MPC
            halos["halfMassRadius"][k] = np.nan
            halos["position"][k, 0] = float(row[i_x])
            halos["position"][k, 1] = float(row[i_y])
            halos["position"][k, 2] = float(row[i_z])
            halos["velocity"][k, 0] = float(row[i_vx])
            halos["velocity"][k, 1] = float(row[i_vy])
            halos["velocity"][k, 2] = float(row[i_vz])
            halos["angularMomentum"][k, 0] = float(row[i_jx])
            halos["angularMomentum"][k, 1] = float(row[i_jy])
            halos["angularMomentum"][k, 2] = float(row[i_jz])
            halos["spin"][k] = float(row[i_spin])

        # Galacticus convention: nodes with no parent have hostIndex==nodeIndex,
        # not -1.
        no_host = halos["hostIndex"] == -1
        halos["hostIndex"][no_host] = halos["nodeIndex"][no_host]
        return halos
