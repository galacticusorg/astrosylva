"""Streaming HDF5 writer that produces Galacticus-format merger trees.

Output layout (``formatVersion = 2``)::

    /                            (root attrs: formatVersion)
    /forestHalos/                (attrs: forest-level flags)
        descendantIndex          int64[N]
        nodeIndex                int64[N]
        hostIndex                int64[N]
        expansionFactor          float64[N]
        nodeMass                 float64[N]
        scaleRadius              float64[N]
        position                 float64[N, 3]
        velocity                 float64[N, 3]
        angularMomentum          float64[N, 3]
        spin                     float64[N]
    /forestIndex/
        firstNode                int64[F]
        numberOfNodes            int64[F]
        forestIndex              int64[F]
        forestWeight             float64[F]
    /cosmology/                  (attrs)
    /units/                      (attrs)
    /groupFinder/                (attrs)
    /simulation/                 (attrs)

Forests are streamed one at a time: per-forest each halo dataset is resized
with ``maxshape=(None,)`` and the new slab is written in place.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

import h5py
import numpy as np

from astrosylva.exceptions import WriterError
from astrosylva.schema import Forest, Metadata

_SCALAR_DATASETS: tuple[tuple[str, str], ...] = (
    ("descendantIndex", "<i8"),
    ("nodeIndex", "<i8"),
    ("hostIndex", "<i8"),
    ("expansionFactor", "<f8"),
    ("nodeMass", "<f8"),
    ("scaleRadius", "<f8"),
    ("halfMassRadius", "<f8"),
    ("spin", "<f8"),
)
_VECTOR_DATASETS: tuple[str, ...] = ("position", "velocity", "angularMomentum")

_FOREST_INDEX_DATASETS: tuple[tuple[str, str], ...] = (
    ("firstNode", "<i8"),
    ("numberOfNodes", "<i8"),
    ("forestIndex", "<i8"),
    ("forestWeight", "<f8"),
)


class GalacticusWriter:
    """Write Galacticus-format HDF5, streaming forest-by-forest."""

    def __init__(
        self,
        path: str | Path,
        metadata: Metadata,
        *,
        chunk_size: int = 4096,
    ) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self.chunk_size = chunk_size
        self._file: h5py.File | None = None
        self._halo_offset = 0
        self._forest_count = 0

    # -------------------------------------------------------- lifecycle

    def __enter__(self) -> GalacticusWriter:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def open(self) -> None:
        if self._file is not None:
            raise WriterError("Writer is already open")
        self._file = h5py.File(self.path, "w")
        f = self._file
        f.attrs["formatVersion"] = np.int32(self.metadata.format_version)

        halos = f.create_group("forestHalos")
        for name, dtype in _SCALAR_DATASETS:
            halos.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(self.chunk_size,),
            )
        for name in _VECTOR_DATASETS:
            halos.create_dataset(
                name,
                shape=(0, 3),
                maxshape=(None, 3),
                dtype="<f8",
                chunks=(self.chunk_size, 3),
            )

        forest = f.create_group("forestIndex")
        for name, dtype in _FOREST_INDEX_DATASETS:
            forest.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(self.chunk_size,),
            )

    def close(self) -> None:
        if self._file is None:
            return
        try:
            self._write_metadata_attrs()
        finally:
            self._file.close()
            self._file = None

    # ------------------------------------------------------------ writes

    def write_forest(self, forest: Forest) -> None:
        if self._file is None:
            raise WriterError("Writer is not open")
        n = forest.n_halos
        halos_group = self._file["forestHalos"]
        new_total = self._halo_offset + n

        for name, _ in _SCALAR_DATASETS:
            ds = halos_group[name]
            ds.resize((new_total,))
            if n:
                ds[self._halo_offset : new_total] = forest.halos[name]
        for name in _VECTOR_DATASETS:
            ds = halos_group[name]
            ds.resize((new_total, 3))
            if n:
                ds[self._halo_offset : new_total, :] = forest.halos[name]

        forest_group = self._file["forestIndex"]
        new_forest_total = self._forest_count + 1
        for name, _ in _FOREST_INDEX_DATASETS:
            forest_group[name].resize((new_forest_total,))
        forest_group["firstNode"][self._forest_count] = self._halo_offset
        forest_group["numberOfNodes"][self._forest_count] = n
        forest_group["forestIndex"][self._forest_count] = forest.forest_id
        forest_group["forestWeight"][self._forest_count] = forest.weight

        self._halo_offset = new_total
        self._forest_count = new_forest_total

    # ------------------------------------------------------------ attrs

    def _write_metadata_attrs(self) -> None:
        assert self._file is not None
        for group_path, attrs in self.metadata.groups().items():
            grp = self._file.require_group(group_path)
            for key, value in attrs.items():
                grp.attrs[key] = _coerce_attr(value)


def _coerce_attr(value: Any) -> Any:
    """Coerce Python types to h5py-friendly representations."""
    if isinstance(value, bool):
        return np.int32(1 if value else 0)
    if isinstance(value, int):
        return np.int32(value) if -(2**31) <= value < 2**31 else np.int64(value)
    if isinstance(value, float):
        return np.float64(value)
    return value
