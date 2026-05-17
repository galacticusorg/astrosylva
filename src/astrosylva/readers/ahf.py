"""AHF (Amiga Halo Finder) reader.

AHF emits one ``.AHF_halos`` catalogue per snapshot plus ``.AHF_mtree`` files
linking halos across snapshots. To produce merger trees in Galacticus format
we walk pairs of snapshots and stitch descendant pointers.

Source keys:

- ``snapshots`` : list of ``{halos: path, mtree: path | null, a: float}``
  ordered from earliest to latest. The last snapshot has ``mtree: null``.

Status: experimental — handles the common AHF column layout (ID, hostHalo,
Mvir, Rvir, Xc, Yc, Zc, VXc, VYc, VZc, Lx, Ly, Lz, lambda). Adjust
``_AHF_COLUMNS`` if your build emits a different schema.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from astrosylva.exceptions import ReaderError
from astrosylva.readers.base import TreeReader
from astrosylva.schema import DEFAULT_UNITS, HALO_DTYPE, Forest, Metadata

# 1-indexed positions of the columns we use in standard AHF .AHF_halos output.
# (AHF column order is configurable per build; users may need to override.)
_AHF_COLUMNS = {
    "ID": 0,
    "hostHalo": 1,
    "Mvir": 3,
    "Rvir": 11,
    "Xc": 5,
    "Yc": 6,
    "Zc": 7,
    "VXc": 8,
    "VYc": 9,
    "VZc": 10,
    "Lx": 21,
    "Ly": 22,
    "Lz": 23,
    "lambda": 20,
}


class AHFReader(TreeReader):
    """Reader for AHF halo catalogues + merger-tree files."""

    name: ClassVar[str] = "ahf"
    aliases: ClassVar[tuple[str, ...]] = ("amiga",)

    def metadata(self) -> Metadata:
        return Metadata(units=dict(DEFAULT_UNITS))

    def __len__(self) -> int:
        self._ensure_loaded()
        assert self._forests is not None
        return len(self._forests)

    def __iter__(self) -> Iterator[Forest]:
        self._ensure_loaded()
        assert self._forests is not None
        yield from self._forests

    def _ensure_loaded(self) -> None:
        if getattr(self, "_forests", None) is not None:
            return
        snapshots: list[dict[str, Any]] = self.source.require("snapshots")
        if not snapshots:
            raise ReaderError("AHF reader requires at least one snapshot")
        # Load halos snapshot-by-snapshot, oldest first.
        per_snap: list[np.ndarray] = []
        for snap in snapshots:
            per_snap.append(self._load_halo_catalogue(Path(snap["halos"]), float(snap["a"])))
        # Build descendant links via .AHF_mtree (current -> next snapshot).
        for i, snap in enumerate(snapshots[:-1]):
            mtree_path = snap.get("mtree")
            if mtree_path is None:
                continue
            self._apply_mtree(Path(mtree_path), per_snap[i], per_snap[i + 1])
        # All halos in one giant forest for now; partitioning by connected
        # components requires a union-find walk that is left for v2.
        halos = np.concatenate(per_snap) if per_snap else np.empty(0, dtype=HALO_DTYPE)
        self._forests = [Forest(forest_id=0, halos=halos, weight=1.0)]

    @staticmethod
    def _load_halo_catalogue(path: Path, a: float) -> np.ndarray:
        if not path.is_file():
            raise ReaderError(f"AHF halos file not found: {path}")
        rows: list[list[str]] = []
        with path.open() as fh:
            for raw in fh:
                stripped = raw.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                rows.append(stripped.split())
        n = len(rows)
        halos = np.empty(n, dtype=HALO_DTYPE)
        c = _AHF_COLUMNS
        for k, row in enumerate(rows):
            halos["nodeIndex"][k] = int(row[c["ID"]])
            halos["descendantIndex"][k] = -1
            halos["hostIndex"][k] = int(row[c["hostHalo"]])
            halos["expansionFactor"][k] = a
            halos["nodeMass"][k] = float(row[c["Mvir"]])
            halos["scaleRadius"][k] = float(row[c["Rvir"]]) / 1000.0  # kpc/h -> Mpc/h
            halos["position"][k, 0] = float(row[c["Xc"]]) / 1000.0
            halos["position"][k, 1] = float(row[c["Yc"]]) / 1000.0
            halos["position"][k, 2] = float(row[c["Zc"]]) / 1000.0
            halos["velocity"][k, 0] = float(row[c["VXc"]])
            halos["velocity"][k, 1] = float(row[c["VYc"]])
            halos["velocity"][k, 2] = float(row[c["VZc"]])
            halos["angularMomentum"][k, 0] = float(row[c["Lx"]])
            halos["angularMomentum"][k, 1] = float(row[c["Ly"]])
            halos["angularMomentum"][k, 2] = float(row[c["Lz"]])
            halos["spin"][k] = float(row[c["lambda"]])
        # Galacticus convention: no-host -> self.
        no_host = halos["hostIndex"] == 0
        halos["hostIndex"][no_host] = halos["nodeIndex"][no_host]
        return halos

    @staticmethod
    def _apply_mtree(
        path: Path,
        current: np.ndarray,
        descendants: np.ndarray,  # noqa: ARG004  reserved for future cross-validation
    ) -> None:
        if not path.is_file():
            raise ReaderError(f"AHF mtree file not found: {path}")
        # AHF .AHF_mtree format: blocks of
        #   HaloID(at this snap)   #shared_particles   DescendantID(at next snap)   ...
        # followed by per-progenitor lines we don't need here.
        id_to_idx = {int(h): i for i, h in enumerate(current["nodeIndex"])}
        with path.open() as fh:
            for raw in fh:
                stripped = raw.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 3:
                    continue
                halo_id = int(parts[0])
                desc_id = int(parts[2])
                idx = id_to_idx.get(halo_id)
                if idx is None:
                    continue
                current["descendantIndex"][idx] = desc_id
