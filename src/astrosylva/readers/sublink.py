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

Status: experimental — reads a single chunk; host-pointer resolution is
left to future work.
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
        halos["nodeIndex"] = f["SubhaloID"][mask]
        halos["descendantIndex"] = f["DescendantID"][mask]
        # SubLink doesn't have a direct host pointer at this level; default
        # to self until a host-resolution step is wired in.
        halos["hostIndex"] = halos["nodeIndex"]
        snap_nums = f["SnapNum"][mask].astype(np.int64)
        halos["expansionFactor"] = self._scale_factors_for(snap_nums)
        halos["nodeMass"] = f["SubhaloMass"][mask].astype(np.float64) * 1e10  # 10^10 Msun/h
        halos["scaleRadius"] = f["SubhaloHalfmassRad"][mask].astype(np.float64) / 1000.0
        halos["position"] = f["SubhaloPos"][mask].astype(np.float64) / 1000.0
        halos["velocity"] = f["SubhaloVel"][mask].astype(np.float64)
        halos["angularMomentum"] = f["SubhaloSpin"][mask].astype(np.float64)
        halos["spin"] = 0.0
        return halos
