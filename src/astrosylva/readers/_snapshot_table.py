"""Shared snapshot scale-factor table loading.

SubLink and LHaloTree both deliver halos tagged with a ``SnapNum`` but no
scale factor; the user supplies a ``snap -> a`` mapping. This module
centralises the loading / lookup logic so both readers share the exact
same option names, precedence, and error/warning surface.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from astrosylva.exceptions import ReaderError
from astrosylva.readers.base import ReaderSource


def _read_snap_table_file(path: Path, quantity: str) -> dict[int, float]:
    """Parse a snapshot-table text file in one of two auto-detected shapes:

    - **2-column** (``snap_num value``) — each row is a ``snap`` and a
      scale factor or redshift.
    - **1-column** (``value`` per line) — the snap number is the
      0-based line index. This is the Millennium / L-Galaxies
      ``a_list`` convention.

    Auto-detection picks 1-column when every record has a single
    token, 2-column when every record has at least two; mixed widths
    raise.

    ``quantity`` selects whether the right-hand value is a scale
    factor (passed through) or a redshift (converted to ``a =
    1/(1+z)``).
    """
    if quantity not in ("scale_factor", "redshift"):
        raise ReaderError(
            f"snapshot_table_quantity must be 'scale_factor' or 'redshift', got {quantity!r}"
        )
    if not path.is_file():
        raise ReaderError(f"snapshot table file not found: {path}")
    records = _load_snap_table_records(path)
    widths = {len(r) for r in records}
    if widths == {1}:
        out = _parse_snap_table_alist(records, path, quantity)
    elif min(widths) >= 2:
        out = _parse_snap_table_snap_keyed(records, quantity)
    else:
        raise ReaderError(
            f"snapshot table {path} has inconsistent row widths {sorted(widths)}; "
            "expected either 1 column (snap inferred from line index) or >=2."
        )
    if not out:
        raise ReaderError(f"snapshot table {path} contained no usable rows")
    return out


def _load_snap_table_records(path: Path) -> list[list[str]]:
    records: list[list[str]] = []
    with path.open() as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            records.append(stripped.split())
    if not records:
        raise ReaderError(f"snapshot table {path} contained no usable rows")
    return records


def _parse_snap_table_alist(
    records: list[list[str]], path: Path, quantity: str
) -> dict[int, float]:
    out: dict[int, float] = {}
    for snap, parts in enumerate(records):
        try:
            value = float(parts[0])
        except ValueError as exc:
            raise ReaderError(
                f"snapshot table {path} (a-list format) has non-numeric "
                f"value at line {snap}: {parts[0]!r}"
            ) from exc
        out[snap] = value if quantity == "scale_factor" else 1.0 / (1.0 + value)
    return out


def _parse_snap_table_snap_keyed(records: list[list[str]], quantity: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for parts in records:
        try:
            snap = int(parts[0])
        except ValueError:
            continue  # header line
        value = float(parts[1])
        out[snap] = value if quantity == "scale_factor" else 1.0 / (1.0 + value)
    return out


def load_snap_table(
    source: ReaderSource,
    options: Mapping[str, Any],
) -> dict[int, float]:
    """Build a ``snap_num -> scale_factor`` mapping from source/options.

    Source precedence: ``source.snapshot_table`` then ``options.scale_factors``
    then ``options.redshifts``. Returns an empty dict when no source is
    configured (callers can choose strict-vs-lenient handling).
    """
    table_path = source.get("snapshot_table")
    has_scales = "scale_factors" in options
    has_redshifts = "redshifts" in options
    if has_scales and has_redshifts:
        raise ReaderError("Specify at most one of options.scale_factors or options.redshifts.")
    if table_path is not None:
        qty = options.get("snapshot_table_quantity", "scale_factor")
        return _read_snap_table_file(Path(table_path), qty)
    if has_scales:
        return {int(k): float(v) for k, v in options["scale_factors"].items()}
    if has_redshifts:
        return {int(k): 1.0 / (1.0 + float(v)) for k, v in options["redshifts"].items()}
    return {}


def apply_scale_factors(
    snap_to_a: Mapping[int, float],
    snap_nums: np.ndarray,
    *,
    strict: bool,
    reader_name: str,
) -> np.ndarray:
    """Look up scale factors for an array of ``SnapNum`` values.

    Missing table (or missing entries) raises :class:`ReaderError` when
    ``strict`` is true; otherwise warns and fills the gaps with ``NaN``.
    ``reader_name`` is used in the error/warning text so users see which
    reader is complaining.
    """
    if not snap_to_a:
        msg = (
            f"{reader_name} has no snapshot scale-factor table; supply one of "
            "source.snapshot_table, options.scale_factors, or options.redshifts."
        )
        if strict:
            raise ReaderError(msg)
        warnings.warn(msg, stacklevel=3)
        return np.full(snap_nums.shape, np.nan, dtype=np.float64)

    unique = {int(s) for s in np.unique(snap_nums)}
    missing = sorted(unique - snap_to_a.keys())
    if missing:
        msg = f"snapshot scale-factor table is missing entries for SnapNum: {missing}"
        if strict:
            raise ReaderError(msg)
        warnings.warn(msg, stacklevel=3)

    out = np.full(snap_nums.shape, np.nan, dtype=np.float64)
    for snap, a in snap_to_a.items():
        out[snap_nums == snap] = a
    return out
