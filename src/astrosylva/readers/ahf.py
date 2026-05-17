"""AHF (Amiga Halo Finder) reader.

AHF emits one ``.AHF_halos`` catalogue per snapshot plus ``.AHF_mtree``
(or ``.AHF_mtree_idx``) files linking halos across snapshots. To produce
merger trees in Galacticus format we walk pairs of snapshots, stitch
descendant pointers, and partition the resulting halo set into forests
via union-find on descendant + host edges.

Source keys
-----------

- ``snapshots`` : list of ``{halos: path, mtree: path | null, a: float}``
  ordered from earliest to latest. The last snapshot has ``mtree: null``.

mtree formats
-------------

Two AHF mtree variants are recognised:

- ``idx``   — two columns per line: ``ProgID DescID``  (the
              ``.AHF_mtree_idx`` format).
- ``block`` — blocks of ``DescID HaloPart NumProgenitors`` followed by
              ``NumProgenitors`` lines of ``SharedPart ProgID HaloPart``
              (the standard ``.AHF_mtree`` format).

``options.mtree_format`` selects ``"auto"`` (default), ``"idx"``, or
``"block"``. Auto-detect picks ``idx`` when every record is two
tokens wide and ``block`` when every record is at least three; mixed
widths raise.

Status: experimental — column layout is auto-detected from the
``.AHF_halos`` header (``#name(1) name(2) ...``) and falls back to the
defaults below when the file has no recognisable header. Override
individual columns via ``options.columns``. Halo IDs are expected to be
globally unique across snapshots (standard AHF behaviour).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from astrosylva.exceptions import ReaderError
from astrosylva.readers._forests import clamp_hosts_to_forest, group_by_union_find
from astrosylva.readers.base import TreeReader
from astrosylva.schema import DEFAULT_UNITS, HALO_DTYPE, Forest, Metadata

# 0-based positions of the columns we use in a common AHF .AHF_halos layout.
# Real AHF builds vary, so this is only a fallback — auto-detected header
# values and ``options.columns`` overrides take precedence.
_AHF_DEFAULT_COLUMNS: dict[str, int] = {
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
_AHF_REQUIRED_COLUMNS = frozenset(_AHF_DEFAULT_COLUMNS)

_AHF_HEADER_TOKEN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\((\d+)\)")


def _parse_ahf_header(line: str) -> dict[str, int]:
    """Parse ``#name(1) other(2) ...`` into a ``{name: 0-based-index}`` map.

    Tokens that don't match the ``name(N)`` shape (e.g. annotated with
    units like ``Mvir(4)[Msun/h]``) are silently skipped — the user can
    fill those gaps with ``options.columns``.
    """
    out: dict[int, str] = {}
    tokens = line.lstrip("#").split()
    for tok in tokens:
        m = _AHF_HEADER_TOKEN.match(tok)
        if m is None:
            continue
        pos = int(m.group(2)) - 1  # AHF headers are 1-based
        out[pos] = m.group(1)
    return {name: pos for pos, name in out.items()}


_VALID_MTREE_FORMATS = ("auto", "idx", "block")


class AHFReader(TreeReader):
    """Reader for AHF halo catalogues + merger-tree files."""

    name: ClassVar[str] = "ahf"
    aliases: ClassVar[tuple[str, ...]] = ("amiga",)

    def __init__(self, source: Any, options: dict[str, Any] | None = None) -> None:
        super().__init__(source, options)
        self._mtree_format: str = self.options.get("mtree_format", "auto")
        if self._mtree_format not in _VALID_MTREE_FORMATS:
            raise ReaderError(
                f"mtree_format must be one of 'auto', 'idx', 'block'; got {self._mtree_format!r}"
            )
        overrides = self.options.get("columns", {}) or {}
        if not isinstance(overrides, dict):
            raise ReaderError(
                "options.columns must be a mapping of column-name to 0-based index; "
                f"got {type(overrides).__name__}"
            )
        try:
            self._column_overrides: dict[str, int] = {str(k): int(v) for k, v in overrides.items()}
        except (TypeError, ValueError) as exc:
            raise ReaderError("options.columns values must be integer column indices") from exc

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
            self._apply_mtree(Path(mtree_path), per_snap[i])
        if not per_snap:
            self._forests = []
            return
        halos = np.concatenate(per_snap)
        # Partition into self-contained forests via union-find on
        # descendant + host edges. ``root_desc`` is just nodeIndex
        # because AHF has no separate root-descendant concept.
        forest_index = group_by_union_find(
            halos["nodeIndex"],
            halos["nodeIndex"],
            halos["descendantIndex"],
            halos["hostIndex"],
        )
        forests: list[Forest] = []
        for forest_id, indices in forest_index.items():
            forest_halos = halos[indices].copy()
            forest_halos["hostIndex"] = clamp_hosts_to_forest(
                forest_halos["hostIndex"], forest_halos["nodeIndex"]
            )
            forests.append(Forest(forest_id=forest_id, halos=forest_halos))
        self._forests = forests

    def _load_halo_catalogue(self, path: Path, a: float) -> np.ndarray:
        if not path.is_file():
            raise ReaderError(f"AHF halos file not found: {path}")
        header_columns, rows = _read_ahf_halos_file(path)
        row_width = len(rows[0]) if rows else 0
        column_map = self._resolve_columns(header_columns, path, row_width)
        max_index = max(column_map.values()) if column_map else -1

        n = len(rows)
        halos = np.empty(n, dtype=HALO_DTYPE)
        c = column_map
        for k, row in enumerate(rows):
            if len(row) <= max_index:
                raise ReaderError(
                    f"AHF row {k} in {path} has {len(row)} fields but column "
                    f"layout needs at least {max_index + 1}."
                )
            halos["nodeIndex"][k] = int(row[c["ID"]])
            halos["descendantIndex"][k] = -1
            halos["hostIndex"][k] = int(row[c["hostHalo"]])
            halos["expansionFactor"][k] = a
            halos["nodeMass"][k] = float(row[c["Mvir"]])
            halos["scaleRadius"][k] = float(row[c["Rvir"]]) / 1000.0  # kpc/h -> Mpc/h
            halos["halfMassRadius"][k] = np.nan  # AHF default output lacks half-mass radius
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

    def _resolve_columns(
        self,
        header_columns: dict[str, int],
        path: Path,
        row_width: int,
    ) -> dict[str, int]:
        """Merge defaults, header-detected, and user-supplied column maps.

        Precedence (lowest → highest): defaults → header → user overrides.
        Any entry pointing past the file's actual row width is treated as
        absent (the default may be a phantom that the user's build doesn't
        actually emit). Raises if any required column is still missing.
        """
        column_map: dict[str, int] = dict(_AHF_DEFAULT_COLUMNS)
        column_map.update(header_columns)
        column_map.update(self._column_overrides)
        if row_width > 0:
            column_map = {name: idx for name, idx in column_map.items() if idx < row_width}
        missing = sorted(_AHF_REQUIRED_COLUMNS - column_map.keys())
        if missing:
            raise ReaderError(
                f"AHF file {path} is missing required columns {missing}; "
                "supply them via options.columns."
            )
        return column_map

    def _apply_mtree(self, path: Path, current: np.ndarray) -> None:
        if not path.is_file():
            raise ReaderError(f"AHF mtree file not found: {path}")
        records = _read_mtree_records(path)
        if not records:
            return
        fmt = self._resolve_mtree_format(records, path)
        id_to_idx = {int(h): i for i, h in enumerate(current["nodeIndex"])}
        if fmt == "idx":
            _apply_mtree_idx(records, id_to_idx, current, path)
        else:
            _apply_mtree_block(records, id_to_idx, current, path)

    def _resolve_mtree_format(self, records: list[list[str]], path: Path) -> str:
        if self._mtree_format != "auto":
            return self._mtree_format
        widths = {len(r) for r in records}
        if widths == {2}:
            return "idx"
        if min(widths) >= 3:
            return "block"
        raise ReaderError(
            f"AHF mtree at {path} has inconsistent record widths {sorted(widths)}; "
            "set options.mtree_format explicitly ('idx' or 'block')."
        )


def _read_ahf_halos_file(path: Path) -> tuple[dict[str, int], list[list[str]]]:
    """Parse the column header (from any ``#name(N) ...`` line) and rows.

    Returns ``({name: 0-based-index}, rows)``. Header tokens accumulate
    across multiple comment lines — useful for AHF outputs that wrap the
    column list. An empty header dict means "no usable header found".
    """
    header: dict[str, int] = {}
    rows: list[list[str]] = []
    with path.open() as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header.update(_parse_ahf_header(stripped))
                continue
            rows.append(stripped.split())
    return header, rows


def _read_mtree_records(path: Path) -> list[list[str]]:
    out: list[list[str]] = []
    with path.open() as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            out.append(stripped.split())
    return out


def _apply_mtree_idx(
    records: list[list[str]],
    id_to_idx: dict[int, int],
    current: np.ndarray,
    path: Path,
) -> None:
    """Apply ``ProgID DescID`` pairs from an .AHF_mtree_idx-style file."""
    for parts in records:
        if len(parts) != 2:
            raise ReaderError(
                f"AHF idx-format mtree expected 2 fields per line, got {parts!r} in {path}"
            )
        try:
            prog_id, desc_id = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ReaderError(f"Non-integer field in {path}: {parts!r}") from exc
        idx = id_to_idx.get(prog_id)
        if idx is not None:
            current["descendantIndex"][idx] = desc_id


def _apply_mtree_block(
    records: list[list[str]],
    id_to_idx: dict[int, int],
    current: np.ndarray,
    path: Path,
) -> None:
    """Apply descendant pointers from a block-format ``.AHF_mtree``.

    Each block is a 3-field header ``DescID HaloPart NumProgenitors``
    followed by ``NumProgenitors`` 3-field progenitor lines
    ``SharedPart ProgID HaloPart``.
    """
    i = 0
    while i < len(records):
        header = records[i]
        if len(header) < 3:
            raise ReaderError(
                f"AHF block-format mtree header expects >=3 fields: {header!r} in {path}"
            )
        try:
            desc_id = int(header[0])
            n_prog = int(header[2])
        except ValueError as exc:
            raise ReaderError(
                f"AHF block-format mtree header not parseable: {header!r} in {path}"
            ) from exc
        if n_prog < 0:
            raise ReaderError(
                f"AHF block-format mtree NumProgenitors negative: {header!r} in {path}"
            )
        i += 1
        for _ in range(n_prog):
            if i >= len(records):
                raise ReaderError(
                    f"Truncated AHF block-format mtree near descendant {desc_id} in {path}"
                )
            prog_line = records[i]
            if len(prog_line) < 2:
                raise ReaderError(
                    f"AHF block-format progenitor line expects >=2 fields: {prog_line!r} in {path}"
                )
            try:
                prog_id = int(prog_line[1])
            except ValueError as exc:
                raise ReaderError(f"Non-integer progenitor ID in {path}: {prog_line!r}") from exc
            idx = id_to_idx.get(prog_id)
            if idx is not None:
                current["descendantIndex"][idx] = desc_id
            i += 1
