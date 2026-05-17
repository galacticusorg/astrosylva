"""SubLink (IllustrisTNG-style) HDF5 reader.

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

Subhalos are grouped into forests by ``RootDescendantID`` (Galacticus's
"forest" semantics — all subhalos sharing an ultimate descendant land in the
same forest).

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
from astrosylva.readers.base import ReaderSource, TreeReader
from astrosylva.schema import DEFAULT_UNITS, HALO_DTYPE, Forest, Metadata


def _read_snap_table_file(path: Path, quantity: str) -> dict[int, float]:
    if quantity not in ("scale_factor", "redshift"):
        raise ReaderError(
            f"snapshot_table_quantity must be 'scale_factor' or 'redshift', got {quantity!r}"
        )
    if not path.is_file():
        raise ReaderError(f"snapshot table file not found: {path}")
    out: dict[int, float] = {}
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                snap = int(parts[0])
            except ValueError:
                continue  # header line
            value = float(parts[1])
            out[snap] = value if quantity == "scale_factor" else 1.0 / (1.0 + value)
    if not out:
        raise ReaderError(f"snapshot table {path} contained no usable rows")
    return out


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
        self._snap_to_a = self._load_snap_table()
        self._forest_ids: np.ndarray | None = None

    # -------------------------------------------------------- public API

    def metadata(self) -> Metadata:
        return Metadata(units=dict(DEFAULT_UNITS))

    def __len__(self) -> int:
        self._ensure_indexed()
        assert self._forest_ids is not None
        return len(self._forest_ids)

    def __iter__(self) -> Iterator[Forest]:
        self._ensure_indexed()
        assert self._forest_ids is not None
        path = self._tree_path()
        with h5py.File(path, "r") as f:
            root_desc = f["RootDescendantID"][:]
            for forest_id in self._forest_ids:
                mask = root_desc == forest_id
                halos = self._build_halos(f, mask)
                yield Forest(forest_id=int(forest_id), halos=halos)

    # ----------------------------------------------------------- helpers

    def _tree_path(self) -> Path:
        return Path(self.source.require("tree_file"))

    def _ensure_indexed(self) -> None:
        if self._forest_ids is not None:
            return
        path = self._tree_path()
        if not path.is_file():
            raise ReaderError(f"SubLink tree file not found: {path}")
        with h5py.File(path, "r") as f:
            if "RootDescendantID" not in f:
                raise ReaderError(
                    f"{path} does not look like a SubLink tree file (missing /RootDescendantID)."
                )
            self._forest_ids = np.unique(f["RootDescendantID"][:])

    def _load_snap_table(self) -> dict[int, float]:
        table_path = self.source.get("snapshot_table")
        has_scales = "scale_factors" in self.options
        has_redshifts = "redshifts" in self.options
        if has_scales and has_redshifts:
            raise ReaderError("Specify at most one of options.scale_factors or options.redshifts.")
        if table_path is not None:
            qty = self.options.get("snapshot_table_quantity", "scale_factor")
            return _read_snap_table_file(Path(table_path), qty)
        if has_scales:
            return {int(k): float(v) for k, v in self.options["scale_factors"].items()}
        if has_redshifts:
            return {int(k): 1.0 / (1.0 + float(v)) for k, v in self.options["redshifts"].items()}
        return {}

    def _scale_factors_for(self, snap_nums: np.ndarray) -> np.ndarray:
        if not self._snap_to_a:
            msg = (
                "SubLink reader has no snapshot scale-factor table; supply one of "
                "source.snapshot_table, options.scale_factors, or options.redshifts."
            )
            if self._strict_scale_factors:
                raise ReaderError(msg)
            warnings.warn(msg, stacklevel=2)
            return np.full(snap_nums.shape, np.nan, dtype=np.float64)

        unique = {int(s) for s in np.unique(snap_nums)}
        missing = sorted(unique - self._snap_to_a.keys())
        if missing:
            msg = f"snapshot scale-factor table is missing entries for SnapNum: {missing}"
            if self._strict_scale_factors:
                raise ReaderError(msg)
            warnings.warn(msg, stacklevel=2)

        out = np.full(snap_nums.shape, np.nan, dtype=np.float64)
        for snap, a in self._snap_to_a.items():
            out[snap_nums == snap] = a
        return out

    def _build_halos(self, f: h5py.File, mask: np.ndarray) -> np.ndarray:
        n = int(mask.sum())
        halos = np.empty(n, dtype=HALO_DTYPE)
        node_ids = f["SubhaloID"][mask].astype(np.int64)
        halos["nodeIndex"] = node_ids
        halos["descendantIndex"] = f["DescendantID"][mask]
        snap_nums = f["SnapNum"][mask].astype(np.int64)
        halos["hostIndex"] = self._resolve_hosts(f, mask, node_ids, snap_nums)
        halos["expansionFactor"] = self._scale_factors_for(snap_nums)
        halos["nodeMass"] = f["SubhaloMass"][mask].astype(np.float64) * 1e10  # 10^10 Msun/h
        halos["scaleRadius"] = f["SubhaloHalfmassRad"][mask].astype(np.float64) / 1000.0
        halos["position"] = f["SubhaloPos"][mask].astype(np.float64) / 1000.0
        halos["velocity"] = f["SubhaloVel"][mask].astype(np.float64)
        halos["angularMomentum"] = f["SubhaloSpin"][mask].astype(np.float64)
        halos["spin"] = 0.0
        return halos

    def _resolve_hosts(
        self,
        f: h5py.File,
        mask: np.ndarray,
        node_ids: np.ndarray,
        snap_nums: np.ndarray,
    ) -> np.ndarray:
        mode = self._host_resolution
        if mode == "self":
            return np.array(node_ids, copy=True)

        if mode in ("auto", "field") and "FirstSubhaloInFOFGroupID" in f:
            hosts = np.asarray(f["FirstSubhaloInFOFGroupID"][mask], dtype=np.int64)
            return _clamp_hosts_to_forest(hosts, node_ids)
        if mode == "field":
            raise ReaderError(
                "host_resolution='field' requires /FirstSubhaloInFOFGroupID in the SubLink file."
            )

        if mode in ("auto", "fof_compute"):
            has_grnr = "SubhaloGrNr" in f
            has_subfind = "SubfindID" in f
            if has_grnr and has_subfind:
                grnr = np.asarray(f["SubhaloGrNr"][mask], dtype=np.int64)
                subfind = np.asarray(f["SubfindID"][mask], dtype=np.int64)
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
