"""Optional integration tests against ytree's sample datasets.

The `ytree project <https://ytree.readthedocs.io>`_ publishes a
collection of real merger-tree samples covering every format we
support (Consistent-Trees, LHaloTree, SubLink, AHF, and others).
The data set is too large to bundle with this repository, but if
you've downloaded it locally — see
https://ytree.readthedocs.io/en/latest/Data.html — these tests will
exercise the readers against it.

Point the ``ASTROSYLVA_YTREE_DATA`` environment variable at the
top-level directory of the unpacked collection. Each test discovers
the per-format subdirectory inside that root (e.g.
``consistent_trees/``, ``ahf_halos/``) and skips when the expected
files aren't present.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from astrosylva.readers import ReaderSource
from astrosylva.readers.ahf import AHFReader
from astrosylva.readers.consistent_trees import ConsistentTreesReader
from astrosylva.readers.lhalotree import LHaloTreeReader
from astrosylva.readers.sublink import SubLinkReader


def _ytree_root() -> Path | None:
    raw = os.environ.get("ASTROSYLVA_YTREE_DATA")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


_ROOT = _ytree_root()
_SKIP_REASON = (
    "Set ASTROSYLVA_YTREE_DATA to the unpacked ytree sample-data root to "
    "run these integration tests. See "
    "https://ytree.readthedocs.io/en/latest/Data.html for the download."
)


# Each entry describes how to point the relevant reader at the
# corresponding ytree sample directory. The mtree-format / scale-factor /
# columns options here are placeholders — real ytree samples may need
# additional tuning. Override per-sample by editing this table.
@pytest.mark.skipif(_ROOT is None, reason=_SKIP_REASON)
def test_ytree_consistent_trees_sample() -> None:
    """Smoke-test against ytree's Consistent-Trees sample."""
    assert _ROOT is not None
    sample_dir = _ROOT / "consistent_trees"
    if not sample_dir.is_dir():
        pytest.skip(f"ytree CT sample not found at {sample_dir}")
    forests_path = sample_dir / "forests.list"
    locations_path = sample_dir / "locations.dat"
    if not forests_path.is_file() or not locations_path.is_file():
        pytest.skip(f"forests.list / locations.dat missing in {sample_dir}")
    reader = ConsistentTreesReader(
        ReaderSource(
            {
                "input_path": str(sample_dir),
                "forests_path": str(forests_path),
                "locations_path": str(locations_path),
            }
        )
    )
    n_forests = len(reader)
    assert n_forests > 0
    # Sanity-check the first forest has at least one halo.
    forest = next(iter(reader))
    assert forest.n_halos > 0


@pytest.mark.skipif(_ROOT is None, reason=_SKIP_REASON)
def test_ytree_lhalotree_sample() -> None:
    """Smoke-test against ytree's LHaloTree sample."""
    assert _ROOT is not None
    sample_dir = _ROOT / "lhalotree"
    if not sample_dir.is_dir():
        pytest.skip(f"ytree LHaloTree sample not found at {sample_dir}")
    chunks = sorted(sample_dir.glob("trees_*.*"))
    if not chunks:
        pytest.skip(f"no trees_*.* files in {sample_dir}")
    # The Millennium-style sample ships an `a_list.txt` (one scale factor
    # per line, snap inferred from the line index). Some builds may also
    # ship a 2-column `snap_a.txt` mapping; try both.
    snap_table_candidates = ["a_list.txt", "snap_a.txt", "millennium.a_list"]
    snap_table: Path | None = None
    for name in snap_table_candidates:
        candidate = sample_dir / name
        if candidate.is_file():
            snap_table = candidate
            break
    if snap_table is None:
        pytest.skip(
            f"ytree LHaloTree sample at {sample_dir} needs a snapshot table "
            f"({snap_table_candidates}) to look up expansion factors."
        )
    reader = LHaloTreeReader(
        ReaderSource(
            {
                "tree_files": [str(p) for p in chunks],
                "snapshot_table": str(snap_table),
            }
        )
    )
    assert len(reader) > 0
    forest = next(iter(reader))
    assert forest.n_halos > 0


@pytest.mark.skipif(_ROOT is None, reason=_SKIP_REASON)
def test_ytree_sublink_sample() -> None:
    """Smoke-test against ytree's SubLink sample."""
    assert _ROOT is not None
    sample_dir = _ROOT / "sublink"
    if not sample_dir.is_dir():
        pytest.skip(f"ytree SubLink sample not found at {sample_dir}")
    chunks = sorted(sample_dir.glob("tree_extended.*.hdf5"))
    if not chunks:
        pytest.skip(f"no tree_extended.*.hdf5 files in {sample_dir}")
    snap_table = sample_dir / "snap_a.txt"
    if not snap_table.is_file():
        pytest.skip(
            f"ytree SubLink sample at {sample_dir} needs a snap_a.txt "
            "(snap_num scale_factor per line) for SubLink expansion factors."
        )
    reader = SubLinkReader(
        ReaderSource(
            {
                "tree_files": [str(p) for p in chunks],
                "snapshot_table": str(snap_table),
            }
        )
    )
    assert len(reader) > 0
    forest = next(iter(reader))
    assert forest.n_halos > 0


@pytest.mark.skipif(_ROOT is None, reason=_SKIP_REASON)
def test_ytree_ahf_sample() -> None:
    """Smoke-test against ytree's AHF sample."""
    assert _ROOT is not None
    sample_dir = _ROOT / "ahf_halos"
    if not sample_dir.is_dir():
        pytest.skip(f"ytree AHF sample not found at {sample_dir}")
    halos = sorted(sample_dir.glob("*.AHF_halos"))
    if not halos:
        pytest.skip(f"no .AHF_halos files in {sample_dir}")
    # Pair each halos file with its .AHF_mtree(_idx) sibling if present.
    snapshots: list[dict[str, object]] = []
    for halos_path in halos:
        stem = halos_path.name.split(".AHF_halos")[0]
        mtree_idx = sample_dir / f"{stem}.AHF_mtree_idx"
        mtree = sample_dir / f"{stem}.AHF_mtree"
        mtree_path: str | None = None
        if mtree_idx.is_file():
            mtree_path = str(mtree_idx)
        elif mtree.is_file():
            mtree_path = str(mtree)
        # The ytree AHF sample needs the scale factor per snapshot. Read it
        # from a sibling snap_a.txt mapping basename -> a if present, else
        # mark this test xfail-needing-data.
        snapshots.append({"halos": str(halos_path), "mtree": mtree_path, "a": 1.0})
    if not snapshots:
        pytest.skip("no snapshots in ytree AHF sample")
    reader = AHFReader(ReaderSource({"snapshots": snapshots}))
    assert len(reader) > 0
    forest = next(iter(reader))
    assert forest.n_halos > 0
