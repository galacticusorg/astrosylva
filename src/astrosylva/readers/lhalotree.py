"""LHaloTree (Millennium-style binary) reader.

The format is a sequence of files each laid out as::

    int32  Ntrees
    int32  TotNHalos
    int32[Ntrees]  NhalosPerTree
    halo_struct[TotNHalos]

where ``halo_struct`` is the fixed-size struct defined by the Millennium
post-processing pipeline (Descendant, FirstProgenitor, NextProgenitor,
FirstHaloInFOFgroup, NextHaloInFOFgroup, Len, M_Mean200, Mvir, M_TopHat,
Pos[3], Vel[3], VelDisp, Vmax, Spin[3], MostBoundID, SnapNum, FileNr,
SubhaloIndex, SubHalfMass).

Status: skeleton — header parsing only. Payload parsing requires the
simulation's per-snapshot ``a(z)`` table to recover ``expansionFactor``,
which the binary file does not contain. Wire that in via reader options
before unblocking the payload path.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from astrosylva.exceptions import ReaderError
from astrosylva.readers.base import TreeReader
from astrosylva.schema import Forest, Metadata

_HEADER_STRUCT = struct.Struct("<ii")  # Ntrees, TotNHalos


class LHaloTreeReader(TreeReader):
    """Reader for the Millennium-style LHaloTree binary format."""

    name: ClassVar[str] = "lhalotree"
    aliases: ClassVar[tuple[str, ...]] = ("lhalo", "millennium")

    def metadata(self) -> Metadata:
        return Metadata()

    def __len__(self) -> int:
        self._ensure_indexed()
        assert self._n_forests is not None
        return self._n_forests

    def __iter__(self) -> Iterator[Forest]:
        raise NotImplementedError(
            "LHaloTreeReader payload parsing is not yet implemented. "
            "Open an issue with a fixture file to prioritise this format."
        )

    def _ensure_indexed(self) -> None:
        if getattr(self, "_n_forests", None) is not None:
            return
        input_path = Path(self.source.require("input_path"))
        files = sorted(input_path.glob("trees_*.*")) if input_path.is_dir() else [input_path]
        if not files:
            raise ReaderError(f"No LHaloTree files found at {input_path}")
        total = 0
        for f in files:
            with f.open("rb") as fh:
                header = fh.read(_HEADER_STRUCT.size)
                if len(header) < _HEADER_STRUCT.size:
                    raise ReaderError(f"Truncated LHaloTree header in {f}")
                n_trees, _ = _HEADER_STRUCT.unpack(header)
                total += n_trees
        self._n_forests = total
