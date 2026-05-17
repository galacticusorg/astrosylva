"""LHaloTree (Millennium-style binary) reader.

Source keys
-----------

- ``source.tree_files`` : list of binary chunk paths (or a single path).
- ``source.tree_file``  : single chunk path (back-compat alias).

File layout
-----------

Each chunk is one binary file::

    int32          Ntrees
    int32          TotNHalos
    int32[Ntrees]  NhalosPerTree
    halo_struct[TotNHalos]

``halo_struct`` is the 104-byte Millennium / L-Galaxies layout — see
:data:`LHALO_HALO_DTYPE` below. Within each tree the parent / progenitor
/ FOF pointers are *local* (zero-based index into that tree's halos);
the reader assigns a global ``nodeIndex`` per halo (a running counter
across trees and chunks) and rewrites the pointers accordingly. ``-1``
sentinels propagate (no descendant ⇒ ``descendantIndex == -1``).

A "tree" in LHaloTree is already a self-contained set of halos
connected by descendant and FOF edges, so each tree maps to one
Galacticus :class:`Forest`. ``forest_id`` is the global ``nodeIndex``
of the tree's first halo (i.e. the running counter at the start of
the tree).

Scale factor lookup
-------------------

LHaloTree stores ``SnapNum`` but not the scale factor. Supply a
``SnapNum → a`` mapping the same way as the SubLink reader:

1. ``source.snapshot_table``  — whitespace-delimited file
   (``options.snapshot_table_quantity`` selects ``"scale_factor"`` or
   ``"redshift"``).
2. ``options.scale_factors``  — inline ``{snap_num: a}`` mapping.
3. ``options.redshifts``      — inline ``{snap_num: z}``, converted to
   ``a = 1/(1+z)``; mutually exclusive with ``scale_factors``.

Missing-table behaviour is governed by ``options.strict_scale_factors``
(default ``True``).

Field mapping
-------------

The output schema (:data:`astrosylva.schema.HALO_DTYPE`) is filled in as:

==================  =======================================================
schema field        LHaloTree source
==================  =======================================================
nodeIndex           running global counter
descendantIndex     ``Descendant``       (local → global, or -1)
hostIndex           ``FirstHaloInFOFgroup`` (local → global, defaults to self)
expansionFactor     scale-factor table[``SnapNum``]
nodeMass            ``Mvir`` * 1e10        (10^10 M_sun/h → M_sun/h)
scaleRadius         ``SubHalfMass``       (half-mass radius, Mpc/h)
position            ``Pos``               (Mpc/h)
velocity            ``Vel``               (km/s; Hubble flow not removed)
angularMomentum     ``Spin``              (Millennium specific-J vector)
spin                0.0                   (dimensionless spin not stored)
==================  =======================================================

Status: experimental — proxy fields are documented above. Multi-file
runs are supported via ``tree_files``; halo indices are kept globally
unique across chunks.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from astrosylva.exceptions import ReaderError
from astrosylva.readers._snapshot_table import apply_scale_factors, load_snap_table
from astrosylva.readers.base import ReaderSource, TreeReader
from astrosylva.schema import DEFAULT_UNITS, HALO_DTYPE, Forest, Metadata

LHALO_HALO_DTYPE = np.dtype(
    [
        ("Descendant", "<i4"),
        ("FirstProgenitor", "<i4"),
        ("NextProgenitor", "<i4"),
        ("FirstHaloInFOFgroup", "<i4"),
        ("NextHaloInFOFgroup", "<i4"),
        ("Len", "<i4"),
        ("M_Mean200", "<f4"),
        ("Mvir", "<f4"),
        ("M_TopHat", "<f4"),
        ("Pos", "<f4", (3,)),
        ("Vel", "<f4", (3,)),
        ("VelDisp", "<f4"),
        ("Vmax", "<f4"),
        ("Spin", "<f4", (3,)),
        ("MostBoundID", "<i8"),
        ("SnapNum", "<i4"),
        ("FileNr", "<i4"),
        ("SubhaloIndex", "<i4"),
        ("SubHalfMass", "<f4"),
    ]
)
assert LHALO_HALO_DTYPE.itemsize == 104, "LHaloTree halo struct must be 104 bytes"

_HEADER_STRUCT = struct.Struct("<ii")  # Ntrees, TotNHalos


class LHaloTreeReader(TreeReader):
    """Reader for the Millennium-style LHaloTree binary format."""

    name: ClassVar[str] = "lhalotree"
    aliases: ClassVar[tuple[str, ...]] = ("lhalo", "millennium")

    def __init__(self, source: ReaderSource, options: dict[str, Any] | None = None) -> None:
        super().__init__(source, options)
        self._strict_scale_factors = bool(self.options.get("strict_scale_factors", True))
        self._snap_to_a = load_snap_table(source, self.options)
        self._forests: list[np.ndarray] | None = None
        self._forest_ids: list[int] | None = None

    # ------------------------------------------------------------ public

    def metadata(self) -> Metadata:
        return Metadata(units=dict(DEFAULT_UNITS))

    def __len__(self) -> int:
        self._ensure_loaded()
        assert self._forests is not None
        return len(self._forests)

    def __iter__(self) -> Iterator[Forest]:
        self._ensure_loaded()
        assert self._forests is not None
        assert self._forest_ids is not None
        for fid, halos in zip(self._forest_ids, self._forests, strict=False):
            yield Forest(forest_id=fid, halos=halos)

    # ----------------------------------------------------------- helpers

    def _resolve_chunk_paths(self) -> list[Path]:
        files = self.source.get("tree_files")
        if files is not None:
            if isinstance(files, (str, Path)):
                paths = [Path(files)]
            elif isinstance(files, (list, tuple)):
                paths = [Path(p) for p in files]
            else:
                raise ReaderError(
                    "source.tree_files must be a list of paths or a single path; "
                    f"got {type(files).__name__}"
                )
        elif "tree_file" in self.source.paths:
            paths = [Path(self.source.require("tree_file"))]
        else:
            raise ReaderError(
                "LHaloTree reader requires source.tree_file (single chunk) or "
                "source.tree_files (list of chunks)."
            )
        if not paths:
            raise ReaderError("LHaloTree reader received an empty chunk list.")
        return paths

    def _ensure_loaded(self) -> None:
        if self._forests is not None:
            return
        forests: list[np.ndarray] = []
        forest_ids: list[int] = []
        offset = 0
        for path in self._resolve_chunk_paths():
            for raw_tree in self._load_chunk(path):
                halos, forest_id = self._convert_tree(raw_tree, offset)
                forests.append(halos)
                forest_ids.append(forest_id)
                offset += raw_tree.shape[0]
        self._forests = forests
        self._forest_ids = forest_ids

    @staticmethod
    def _load_chunk(path: Path) -> list[np.ndarray]:
        if not path.is_file():
            raise ReaderError(f"LHaloTree file not found: {path}")
        with path.open("rb") as fh:
            header = fh.read(_HEADER_STRUCT.size)
            if len(header) < _HEADER_STRUCT.size:
                raise ReaderError(f"Truncated LHaloTree header in {path}")
            n_trees, tot_n_halos = _HEADER_STRUCT.unpack(header)
            if n_trees < 0 or tot_n_halos < 0:
                raise ReaderError(f"Negative tree/halo count in {path}")
            per_tree = np.fromfile(fh, dtype="<i4", count=n_trees)
            if per_tree.size != n_trees:
                raise ReaderError(f"Truncated NhalosPerTree in {path}")
            if int(per_tree.sum()) != tot_n_halos:
                raise ReaderError(
                    f"NhalosPerTree sum ({int(per_tree.sum())}) "
                    f"does not match TotNHalos ({tot_n_halos}) in {path}"
                )
            all_halos = np.fromfile(fh, dtype=LHALO_HALO_DTYPE, count=tot_n_halos)
            if all_halos.size != tot_n_halos:
                raise ReaderError(f"Truncated halo payload in {path}")
        offsets = np.concatenate([[0], np.cumsum(per_tree)])
        return [all_halos[offsets[i] : offsets[i + 1]] for i in range(n_trees)]

    def _convert_tree(self, raw: np.ndarray, global_offset: int) -> tuple[np.ndarray, int]:
        n = raw.shape[0]
        if n == 0:
            return np.empty(0, dtype=HALO_DTYPE), int(global_offset)

        node_indices = np.arange(global_offset, global_offset + n, dtype=np.int64)
        halos = np.empty(n, dtype=HALO_DTYPE)
        halos["nodeIndex"] = node_indices
        halos["descendantIndex"] = _remap_local_to_global(
            raw["Descendant"], node_indices, n, "Descendant"
        )
        halos["hostIndex"] = _remap_host(raw["FirstHaloInFOFgroup"], node_indices, n)

        snap_nums = raw["SnapNum"].astype(np.int64)
        halos["expansionFactor"] = apply_scale_factors(
            self._snap_to_a,
            snap_nums,
            strict=self._strict_scale_factors,
            reader_name="LHaloTree reader",
        )
        halos["nodeMass"] = raw["Mvir"].astype(np.float64) * 1e10
        halos["scaleRadius"] = raw["SubHalfMass"].astype(np.float64)
        halos["position"] = raw["Pos"].astype(np.float64)
        halos["velocity"] = raw["Vel"].astype(np.float64)
        halos["angularMomentum"] = raw["Spin"].astype(np.float64)
        halos["spin"] = 0.0
        return halos, int(global_offset)


def _remap_local_to_global(
    local: np.ndarray, node_indices: np.ndarray, n: int, field_name: str
) -> np.ndarray:
    """Translate a vector of local indices to global ``nodeIndex`` values.

    ``-1`` stays as ``-1``. Out-of-bounds local indices are rejected.
    """
    out = np.full(local.shape, -1, dtype=np.int64)
    mask = local >= 0
    if mask.any():
        valid_local = local[mask]
        if int(valid_local.max()) >= n or int(valid_local.min()) < 0:
            raise ReaderError(f"LHaloTree {field_name} pointer out of bounds for tree of size {n}")
        out[mask] = node_indices[valid_local]
    return out


def _remap_host(local_fof: np.ndarray, node_indices: np.ndarray, n: int) -> np.ndarray:
    """Translate ``FirstHaloInFOFgroup`` to global ``hostIndex``.

    Halos with no FOF central (sentinel ``-1``) default to self, matching
    Galacticus's "no host" convention.
    """
    out = np.array(node_indices, copy=True)
    mask = local_fof >= 0
    if mask.any():
        valid_local = local_fof[mask]
        if int(valid_local.max()) >= n or int(valid_local.min()) < 0:
            raise ReaderError(
                f"LHaloTree FirstHaloInFOFgroup pointer out of bounds for tree of size {n}"
            )
        out[mask] = node_indices[valid_local]
    return out
