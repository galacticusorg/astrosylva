"""SubLink (IllustrisTNG-style) HDF5 reader.

Source keys
-----------

- ``source.tree_files`` : list of HDF5 chunk paths (or a single path).
- ``source.tree_file``  : single HDF5 chunk path (back-compat alias).

A SubLink run is typically sharded into many ``tree_extended.<N>.hdf5``
files. ``tree_files`` accepts the full list; the reader loads every
chunk and runs forest grouping over the union. Host and descendant
pointers that cross chunk boundaries resolve correctly as long as both
ends are included in the file list.

SubLink emits one HDF5 file per "chunk" with a flat layout::

    /SubhaloID                int64[N]
    /DescendantID             int64[N]
    /FirstProgenitorID        int64[N]
    /NextProgenitorID         int64[N]
    /SubhaloMass              float32[N]
    /SubhaloPos               float32[N, 3]
    /SubhaloVel               float32[N, 3]
    /SubhaloSpin              float32[N, 3]
    /SubhaloHalfmassRad       float32[N]
    /SnapNum                  int32[N]
    /TreeID                   int64[N]
    /RootDescendantID         int64[N]
    ...

Forest grouping
---------------

A Galacticus forest must be self-contained: every gravitational interaction
that affects a halo's evolution should be inside the same forest. The
``RootDescendantID`` SubLink uses is not enough on its own — a satellite
that is disrupted before merging with its host has a different
``RootDescendantID`` from the host, even though they shared a FOF group
for most of cosmic history.

Forests are therefore the connected components of the union of two relations:

1. *descendant edges*  — ``DescendantID(i) == nodeIndex(j)``
2. *FOF edges*         — ``host(i) == nodeIndex(j)`` (host resolved per the
   ``host_resolution`` option, below)

Computed via union-find. Each component's ``forest_id`` is the minimum
``RootDescendantID`` of its members. Switch back to the legacy
``RootDescendantID``-only grouping with
``options.forest_grouping = "root_descendant"``.

Scale factor lookup
-------------------

The SubLink HDF5 file does not record the per-snapshot scale factor, so
:attr:`HALO_DTYPE["expansionFactor"]` must be filled in from an external
table. Supply one of the following, in priority order:

1. ``source.snapshot_table`` — path to a whitespace-delimited file with two
   columns: ``SnapNum`` and either scale factor or redshift. Set
   ``options.snapshot_table_quantity`` to ``"scale_factor"`` (default) or
   ``"redshift"``.
2. ``options.scale_factors`` — inline ``{snap_num: a}`` mapping.
3. ``options.redshifts`` — inline ``{snap_num: z}`` mapping (converted to
   ``a = 1/(1+z)``). Mutually exclusive with ``scale_factors``.

If no table is supplied or required snaps are missing, the reader raises
:class:`ReaderError` by default. Pass ``options.strict_scale_factors: false``
to downgrade these to warnings (missing values become ``NaN``).

Host-pointer resolution
-----------------------

Galacticus stores a ``hostIndex`` that points each subhalo to the central
of its FOF group. The reader resolves it via ``options.host_resolution``:

- ``"auto"`` (default): use ``/FirstSubhaloInFOFGroupID`` if present,
  else compute hosts from ``/SubhaloGrNr`` + ``/SubfindID`` (the
  subhalo with ``SubfindID == 0`` in each ``(SnapNum, SubhaloGrNr)``
  bucket is the central), else fall back to self-host with a warning.
- ``"field"``: require ``/FirstSubhaloInFOFGroupID``; raise otherwise.
- ``"fof_compute"``: require ``/SubhaloGrNr`` + ``/SubfindID``; raise
  otherwise.
- ``"self"``: every subhalo is its own host.

Host pointers that reference a subhalo outside the current forest
(common across SubLink chunks) are silently remapped to self, matching
Galacticus's "no host" convention.

Streaming
---------

Indexing loads only the small per-halo bookkeeping fields (SubhaloID,
DescendantID, RootDescendantID, SnapNum, raw host pointers) across all
chunks; this is what union-find needs and what ``__len__`` returns.
Payload arrays (mass, half-mass radius, position, velocity, spin) are
read lazily from the HDF5 files per-forest, so peak memory scales with
the largest single forest rather than the whole run.

Status: experimental.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np

from astrosylva.exceptions import ReaderError
from astrosylva.readers._forests import (
    clamp_hosts_to_forest as _clamp_hosts_to_forest,
)
from astrosylva.readers._forests import (
    group_by_root_descendant as _group_by_root_descendant,
)
from astrosylva.readers._forests import (
    group_by_union_find as _group_by_union_find,
)
from astrosylva.readers._snapshot_table import apply_scale_factors, load_snap_table
from astrosylva.readers.base import ReaderSource, TreeReader
from astrosylva.schema import DEFAULT_UNITS, HALO_DTYPE, Forest, Metadata


@dataclass
class _ChunkRef:
    path: Path
    n_halos: int


class SubLinkReader(TreeReader):
    """Reader for SubLink HDF5 merger trees."""

    name: ClassVar[str] = "sublink"
    aliases: ClassVar[tuple[str, ...]] = ("illustris-sublink", "tng-sublink")

    def __init__(self, source: ReaderSource, options: dict[str, Any] | None = None) -> None:
        super().__init__(source, options)
        self._strict_scale_factors = bool(self.options.get("strict_scale_factors", True))
        self._host_resolution: str = self.options.get("host_resolution", "auto")
        if self._host_resolution not in ("auto", "field", "fof_compute", "self"):
            raise ReaderError(
                "host_resolution must be one of 'auto', 'field', 'fof_compute', 'self'; "
                f"got {self._host_resolution!r}"
            )
        self._forest_grouping: str = self.options.get("forest_grouping", "union_find")
        if self._forest_grouping not in ("union_find", "root_descendant"):
            raise ReaderError(
                "forest_grouping must be 'union_find' or 'root_descendant'; "
                f"got {self._forest_grouping!r}"
            )
        self._snap_to_a = load_snap_table(source, self.options)

        # Index-time arrays populated by _ensure_indexed (cheap, kept).
        # Halo payload (mass, radius, position, velocity, angular
        # momentum) is streamed per-forest from the HDF5 files in
        # __iter__ rather than held in memory.
        self._forest_index: dict[int, np.ndarray] | None = None
        self._chunk_refs: list[_ChunkRef] | None = None
        self._chunk_offsets: np.ndarray | None = None
        self._node_ids: np.ndarray | None = None
        self._descendants: np.ndarray | None = None
        self._snap_nums: np.ndarray | None = None
        self._raw_hosts: np.ndarray | None = None

    # -------------------------------------------------------- public API

    def metadata(self) -> Metadata:
        return Metadata(units=dict(DEFAULT_UNITS))

    def __len__(self) -> int:
        self._ensure_indexed()
        assert self._forest_index is not None
        return len(self._forest_index)

    def __iter__(self) -> Iterator[Forest]:
        """Stream forests with payload loaded lazily per-forest.

        Index-time arrays (IDs, descendants, hosts, snap nums) are kept in
        memory; payload arrays (mass, radius, position, velocity, angular
        momentum) are read on demand from each chunk, just for the halos
        in the current forest. All chunk files are held open for the
        duration of the iteration to avoid reopen overhead per forest.
        """
        self._ensure_indexed()
        assert self._forest_index is not None
        assert self._chunk_refs is not None
        files = [h5py.File(c.path, "r") for c in self._chunk_refs]
        try:
            for forest_id, indices in self._forest_index.items():
                yield Forest(forest_id=forest_id, halos=self._build_halos(indices, files))
        finally:
            for f in files:
                f.close()

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
                "SubLink reader requires source.tree_file (single chunk) or "
                "source.tree_files (list of chunks)."
            )
        if not paths:
            raise ReaderError("SubLink reader received an empty chunk list.")
        return paths

    def _load_chunk_index(self, path: Path) -> dict[str, np.ndarray]:
        """Load only the small index-time fields from a chunk.

        Payload arrays (mass / radius / position / velocity / spin)
        are read lazily in :meth:`_build_halos`, so they don't all
        have to sit in memory at once.
        """
        if not path.is_file():
            raise ReaderError(f"SubLink tree file not found: {path}")
        with h5py.File(path, "r") as f:
            if "RootDescendantID" not in f:
                raise ReaderError(
                    f"{path} does not look like a SubLink tree file (missing /RootDescendantID)."
                )
            node_ids = np.asarray(f["SubhaloID"][:], dtype=np.int64)
            snap_nums = np.asarray(f["SnapNum"][:], dtype=np.int64)
            return {
                "node_ids": node_ids,
                "descendants": np.asarray(f["DescendantID"][:], dtype=np.int64),
                "root_desc": np.asarray(f["RootDescendantID"][:], dtype=np.int64),
                "snap_nums": snap_nums,
                "raw_hosts": self._compute_raw_hosts(f, node_ids, snap_nums),
            }

    def _ensure_indexed(self) -> None:
        if self._forest_index is not None:
            return
        paths = self._resolve_chunk_paths()
        chunks = [self._load_chunk_index(p) for p in paths]
        per_chunk_n_halos = [int(c["node_ids"].shape[0]) for c in chunks]
        self._chunk_refs = [
            _ChunkRef(path=p, n_halos=n) for p, n in zip(paths, per_chunk_n_halos, strict=True)
        ]
        self._chunk_offsets = np.concatenate([[0], np.cumsum(per_chunk_n_halos)])
        self._node_ids = np.concatenate([c["node_ids"] for c in chunks])
        self._descendants = np.concatenate([c["descendants"] for c in chunks])
        root_desc = np.concatenate([c["root_desc"] for c in chunks])
        self._snap_nums = np.concatenate([c["snap_nums"] for c in chunks])
        self._raw_hosts = np.concatenate([c["raw_hosts"] for c in chunks])

        if self._forest_grouping == "root_descendant":
            self._forest_index = _group_by_root_descendant(root_desc)
        else:
            self._forest_index = _group_by_union_find(
                self._node_ids, root_desc, self._descendants, self._raw_hosts
            )

    def _scale_factors_for(self, snap_nums: np.ndarray) -> np.ndarray:
        return apply_scale_factors(
            self._snap_to_a,
            snap_nums,
            strict=self._strict_scale_factors,
            reader_name="SubLink reader",
        )

    def _build_halos(self, indices: np.ndarray, files: list[h5py.File]) -> np.ndarray:
        """Materialise one forest's halos. Payload is read lazily from
        the open chunk handles, only for the rows in ``indices``."""
        assert self._node_ids is not None
        assert self._descendants is not None
        assert self._snap_nums is not None
        assert self._raw_hosts is not None
        assert self._chunk_offsets is not None

        n = len(indices)
        halos = np.empty(n, dtype=HALO_DTYPE)
        node_ids = self._node_ids[indices]
        halos["nodeIndex"] = node_ids
        halos["descendantIndex"] = self._descendants[indices]
        snap_nums = self._snap_nums[indices]
        raw_hosts_forest = self._raw_hosts[indices]
        halos["hostIndex"] = _clamp_hosts_to_forest(raw_hosts_forest, node_ids)
        halos["expansionFactor"] = self._scale_factors_for(snap_nums)
        # SubLink stores a half-mass radius (SubhaloHalfmassRad) but no NFW
        # scale radius, so route it to halfMassRadius and leave scaleRadius
        # as NaN.
        halos["scaleRadius"] = np.nan
        halos["spin"] = 0.0

        # Locate each halo's source chunk; read payload rows in one shot
        # per chunk (h5py requires sorted indices for fancy indexing, so
        # we sort, read, and unsort).
        chunk_ids = np.searchsorted(self._chunk_offsets[1:], indices, side="right")
        local_indices = indices - self._chunk_offsets[chunk_ids]

        mass = np.empty(n, dtype=np.float64)
        radius = np.empty(n, dtype=np.float64)
        position = np.empty((n, 3), dtype=np.float64)
        velocity = np.empty((n, 3), dtype=np.float64)
        ang_mom = np.empty((n, 3), dtype=np.float64)
        for chunk_id in np.unique(chunk_ids):
            mask = chunk_ids == chunk_id
            local_subset = local_indices[mask]
            sort_order = np.argsort(local_subset)
            sorted_local = local_subset[sort_order].tolist()
            unsort_order = np.argsort(sort_order)
            f = files[int(chunk_id)]
            mass_slice = np.asarray(f["SubhaloMass"][sorted_local], dtype=np.float64) * 1e10
            radius_slice = (
                np.asarray(f["SubhaloHalfmassRad"][sorted_local], dtype=np.float64) / 1000.0
            )
            pos_slice = np.asarray(f["SubhaloPos"][sorted_local], dtype=np.float64) / 1000.0
            vel_slice = np.asarray(f["SubhaloVel"][sorted_local], dtype=np.float64)
            spin_slice = np.asarray(f["SubhaloSpin"][sorted_local], dtype=np.float64)
            mass[mask] = mass_slice[unsort_order]
            radius[mask] = radius_slice[unsort_order]
            position[mask] = pos_slice[unsort_order]
            velocity[mask] = vel_slice[unsort_order]
            ang_mom[mask] = spin_slice[unsort_order]
        halos["nodeMass"] = mass
        halos["halfMassRadius"] = radius
        halos["position"] = position
        halos["velocity"] = velocity
        halos["angularMomentum"] = ang_mom
        return halos

    def _compute_raw_hosts(
        self, f: h5py.File, node_ids: np.ndarray, snap_nums: np.ndarray
    ) -> np.ndarray:
        """Return raw host pointers for every subhalo in the chunk.

        These are used both to build forests (union-find) and to populate
        ``hostIndex`` per-halo (after forest-level clamping).
        """
        mode = self._host_resolution
        if mode == "self":
            return np.array(node_ids, copy=True)

        if mode in ("auto", "field") and "FirstSubhaloInFOFGroupID" in f:
            return np.asarray(f["FirstSubhaloInFOFGroupID"][:], dtype=np.int64)
        if mode == "field":
            raise ReaderError(
                "host_resolution='field' requires /FirstSubhaloInFOFGroupID in the SubLink file."
            )

        if mode in ("auto", "fof_compute"):
            has_grnr = "SubhaloGrNr" in f
            has_subfind = "SubfindID" in f
            if has_grnr and has_subfind:
                grnr = np.asarray(f["SubhaloGrNr"][:], dtype=np.int64)
                subfind = np.asarray(f["SubfindID"][:], dtype=np.int64)
                return _hosts_from_fof(node_ids, snap_nums, grnr, subfind)
            if mode == "fof_compute":
                raise ReaderError(
                    "host_resolution='fof_compute' requires both /SubhaloGrNr and "
                    "/SubfindID in the SubLink file."
                )

        warnings.warn(
            "SubLink file has no /FirstSubhaloInFOFGroupID and no "
            "/SubhaloGrNr+/SubfindID; falling back to self-host. "
            "Set options.host_resolution='self' to silence this warning.",
            stacklevel=2,
        )
        return np.array(node_ids, copy=True)


def _hosts_from_fof(
    node_ids: np.ndarray,
    snap_nums: np.ndarray,
    grnr: np.ndarray,
    subfind: np.ndarray,
) -> np.ndarray:
    """Resolve hostIndex from (SnapNum, SubhaloGrNr) FOF groups.

    The subhalo with ``SubfindID == 0`` is the central of its FOF group;
    every other subhalo in the same ``(snap, grnr)`` bucket points to it.
    Subhalos whose central is not in this forest's slice fall back to self.
    """
    out = np.array(node_ids, copy=True)
    central_mask = subfind == 0
    central_map: dict[tuple[int, int], int] = {
        (int(s), int(g)): int(nid)
        for s, g, nid in zip(
            snap_nums[central_mask],
            grnr[central_mask],
            node_ids[central_mask],
            strict=False,
        )
    }
    for i in range(node_ids.shape[0]):
        cid = central_map.get((int(snap_nums[i]), int(grnr[i])))
        if cid is not None:
            out[i] = cid
    return out
