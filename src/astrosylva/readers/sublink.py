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

Status: experimental — reads a single chunk.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np

from astrosylva.exceptions import ReaderError
from astrosylva.readers._snapshot_table import apply_scale_factors, load_snap_table
from astrosylva.readers.base import ReaderSource, TreeReader
from astrosylva.schema import DEFAULT_UNITS, HALO_DTYPE, Forest, Metadata


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

        # Populated by _ensure_indexed.
        self._forest_index: dict[int, np.ndarray] | None = None
        self._node_ids: np.ndarray | None = None
        self._descendants: np.ndarray | None = None
        self._snap_nums: np.ndarray | None = None
        self._mass: np.ndarray | None = None
        self._radius: np.ndarray | None = None
        self._position: np.ndarray | None = None
        self._velocity: np.ndarray | None = None
        self._ang_mom: np.ndarray | None = None
        self._raw_hosts: np.ndarray | None = None

    # -------------------------------------------------------- public API

    def metadata(self) -> Metadata:
        return Metadata(units=dict(DEFAULT_UNITS))

    def __len__(self) -> int:
        self._ensure_indexed()
        assert self._forest_index is not None
        return len(self._forest_index)

    def __iter__(self) -> Iterator[Forest]:
        self._ensure_indexed()
        assert self._forest_index is not None
        for forest_id, indices in self._forest_index.items():
            yield Forest(forest_id=forest_id, halos=self._build_halos(indices))

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

    def _load_chunk(self, path: Path) -> dict[str, np.ndarray]:
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
                "mass": np.asarray(f["SubhaloMass"][:], dtype=np.float64) * 1e10,
                "radius": np.asarray(f["SubhaloHalfmassRad"][:], dtype=np.float64) / 1000.0,
                "position": np.asarray(f["SubhaloPos"][:], dtype=np.float64) / 1000.0,
                "velocity": np.asarray(f["SubhaloVel"][:], dtype=np.float64),
                "ang_mom": np.asarray(f["SubhaloSpin"][:], dtype=np.float64),
                "raw_hosts": self._compute_raw_hosts(f, node_ids, snap_nums),
            }

    def _ensure_indexed(self) -> None:
        if self._forest_index is not None:
            return
        paths = self._resolve_chunk_paths()
        chunks = [self._load_chunk(p) for p in paths]
        self._node_ids = np.concatenate([c["node_ids"] for c in chunks])
        self._descendants = np.concatenate([c["descendants"] for c in chunks])
        root_desc = np.concatenate([c["root_desc"] for c in chunks])
        self._snap_nums = np.concatenate([c["snap_nums"] for c in chunks])
        self._mass = np.concatenate([c["mass"] for c in chunks])
        self._radius = np.concatenate([c["radius"] for c in chunks])
        self._position = np.concatenate([c["position"] for c in chunks])
        self._velocity = np.concatenate([c["velocity"] for c in chunks])
        self._ang_mom = np.concatenate([c["ang_mom"] for c in chunks])
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

    def _build_halos(self, indices: np.ndarray) -> np.ndarray:
        assert self._node_ids is not None
        assert self._descendants is not None
        assert self._snap_nums is not None
        assert self._mass is not None
        assert self._radius is not None
        assert self._position is not None
        assert self._velocity is not None
        assert self._ang_mom is not None
        assert self._raw_hosts is not None

        n = len(indices)
        halos = np.empty(n, dtype=HALO_DTYPE)
        node_ids = self._node_ids[indices]
        halos["nodeIndex"] = node_ids
        halos["descendantIndex"] = self._descendants[indices]
        snap_nums = self._snap_nums[indices]
        raw_hosts_forest = self._raw_hosts[indices]
        halos["hostIndex"] = _clamp_hosts_to_forest(raw_hosts_forest, node_ids)
        halos["expansionFactor"] = self._scale_factors_for(snap_nums)
        halos["nodeMass"] = self._mass[indices]
        # SubLink stores a half-mass radius (SubhaloHalfmassRad) but no NFW
        # scale radius, so route it to halfMassRadius and leave scaleRadius
        # as NaN.
        halos["scaleRadius"] = np.nan
        halos["halfMassRadius"] = self._radius[indices]
        halos["position"] = self._position[indices]
        halos["velocity"] = self._velocity[indices]
        halos["angularMomentum"] = self._ang_mom[indices]
        halos["spin"] = 0.0
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


def _clamp_hosts_to_forest(hosts: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    """Remap host pointers that fall outside the current forest's nodes to self."""
    in_forest = np.isin(hosts, node_ids)
    out = np.array(hosts, copy=True)
    out[~in_forest] = node_ids[~in_forest]
    return out


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


def _group_by_root_descendant(root_desc: np.ndarray) -> dict[int, np.ndarray]:
    """Legacy grouping: each ``RootDescendantID`` is its own forest."""
    out: dict[int, list[int]] = {}
    for i, rd in enumerate(root_desc):
        out.setdefault(int(rd), []).append(i)
    return {fid: np.array(idxs, dtype=np.int64) for fid, idxs in sorted(out.items())}


def _group_by_union_find(
    node_ids: np.ndarray,
    root_desc: np.ndarray,
    descendants: np.ndarray,
    hosts: np.ndarray,
) -> dict[int, np.ndarray]:
    """Union-find on the union of descendant edges and host edges.

    Forest ID for each connected component is the minimum
    ``RootDescendantID`` of its members.
    """
    n = node_ids.shape[0]
    id_to_idx: dict[int, int] = {int(nid): i for i, nid in enumerate(node_ids)}
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        d_idx = id_to_idx.get(int(descendants[i]))
        if d_idx is not None:
            union(i, d_idx)
        h_idx = id_to_idx.get(int(hosts[i]))
        if h_idx is not None:
            union(i, h_idx)

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    labeled: dict[int, list[int]] = {}
    for indices in components.values():
        forest_id = min(int(root_desc[i]) for i in indices)
        # If two components produced the same forest_id (RootDescendantID
        # collision across components), merge them — losing halos here
        # would be silent corruption.
        labeled.setdefault(forest_id, []).extend(indices)
    return {fid: np.array(sorted(idxs), dtype=np.int64) for fid, idxs in sorted(labeled.items())}
