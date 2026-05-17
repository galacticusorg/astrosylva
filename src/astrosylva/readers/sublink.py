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
    ...

Subhalos are grouped into trees by ``TreeID``; multiple trees may share a
``RootDescendantID`` which we treat as the forest id (Galacticus's
"forest" semantics). Status: experimental — reads a single chunk, no
scale-factor lookup yet.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np

from astrosylva.exceptions import ReaderError
from astrosylva.readers.base import TreeReader
from astrosylva.schema import DEFAULT_UNITS, HALO_DTYPE, Forest, Metadata


class SubLinkReader(TreeReader):
    """Reader for SubLink HDF5 merger trees."""

    name: ClassVar[str] = "sublink"
    aliases: ClassVar[tuple[str, ...]] = ("illustris-sublink", "tng-sublink")

    def metadata(self) -> Metadata:
        return Metadata(units=dict(DEFAULT_UNITS))

    def __len__(self) -> int:
        self._ensure_indexed()
        return len(self._forest_ids)

    def __iter__(self) -> Iterator[Forest]:
        self._ensure_indexed()
        path = self._tree_path()
        with h5py.File(path, "r") as f:
            root_desc = f["RootDescendantID"][:]
            for forest_id in self._forest_ids:
                mask = root_desc == forest_id
                halos = self._build_halos(f, mask)
                yield Forest(forest_id=int(forest_id), halos=halos)

    def _tree_path(self) -> Path:
        return Path(self.source.require("tree_file"))

    def _ensure_indexed(self) -> None:
        if getattr(self, "_forest_ids", None) is not None:
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

    @staticmethod
    def _build_halos(f, mask) -> np.ndarray:  # type: ignore[no-untyped-def]
        n = int(mask.sum())
        halos = np.empty(n, dtype=HALO_DTYPE)
        halos["nodeIndex"] = f["SubhaloID"][mask]
        halos["descendantIndex"] = f["DescendantID"][mask]
        # SubLink doesn't have a direct host pointer at this level; default
        # to self until a host-resolution step is wired in.
        halos["hostIndex"] = halos["nodeIndex"]
        # Scale factor must be looked up from SnapNum via a user-supplied
        # snapshot table; for now leave 0 and document the gap.
        halos["expansionFactor"] = 0.0
        halos["nodeMass"] = f["SubhaloMass"][mask].astype(np.float64) * 1e10  # 10^10 Msun/h
        halos["scaleRadius"] = f["SubhaloHalfmassRad"][mask].astype(np.float64) / 1000.0
        halos["position"] = f["SubhaloPos"][mask].astype(np.float64) / 1000.0
        halos["velocity"] = f["SubhaloVel"][mask].astype(np.float64)
        halos["angularMomentum"] = f["SubhaloSpin"][mask].astype(np.float64)
        halos["spin"] = 0.0
        return halos
