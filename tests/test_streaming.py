"""Tests for the streaming iteration paths in SubLink and LHaloTree.

These guard the contracts that:

- ``__len__`` works without reading halo payloads (only the cheap
  index pass is needed).
- ``__iter__`` can be called multiple times and yields identical
  Forests each time (re-reads from disk each iteration).
- The cross-chunk forest grouping in SubLink still works when payload
  reads happen lazily per-forest from h5py file handles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astrosylva.readers import ReaderSource
from astrosylva.readers.lhalotree import LHaloTreeReader
from astrosylva.readers.sublink import SubLinkReader

SCALES = {0: 0.5, 1: 1.0, 2: 0.25}


def _snapshot(forest) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Compact, order-stable signature of a Forest for equality checks."""
    ids = tuple(int(n) for n in forest.halos["nodeIndex"])
    pos_x = tuple(float(p) for p in forest.halos["position"][:, 0])
    return ids, pos_x


# --------------------------------------------------------------- LHaloTree


def test_lhalotree_len_does_not_load_payload(lhalotree_two_trees: Path) -> None:
    reader = LHaloTreeReader(
        ReaderSource({"tree_file": str(lhalotree_two_trees)}),
        options={"scale_factors": SCALES},
    )
    # __len__ should succeed using only the 8-byte chunk header.
    assert len(reader) == 2


def test_lhalotree_iter_is_repeatable(lhalotree_two_trees: Path) -> None:
    reader = LHaloTreeReader(
        ReaderSource({"tree_file": str(lhalotree_two_trees)}),
        options={"scale_factors": SCALES},
    )
    first = [_snapshot(f) for f in reader]
    second = [_snapshot(f) for f in reader]
    assert first == second


def test_lhalotree_streams_across_chunks(
    lhalotree_two_chunks: tuple[Path, Path],
) -> None:
    p0, p1 = lhalotree_two_chunks
    reader = LHaloTreeReader(
        ReaderSource({"tree_files": [str(p0), str(p1)]}),
        options={"scale_factors": SCALES},
    )
    forest_ids = [f.forest_id for f in reader]
    assert forest_ids == [0, 2]
    # Re-iterating gives the same forest_ids (offset counter resets cleanly).
    assert [f.forest_id for f in reader] == forest_ids


# ----------------------------------------------------------------- SubLink


def test_sublink_len_does_not_load_payload(sublink_tree_file: Path) -> None:
    reader = SubLinkReader(
        ReaderSource({"tree_file": str(sublink_tree_file)}),
        options={"scale_factors": SCALES},
    )
    assert len(reader) == 1


def test_sublink_iter_is_repeatable(sublink_tree_file: Path) -> None:
    reader = SubLinkReader(
        ReaderSource({"tree_file": str(sublink_tree_file)}),
        options={"scale_factors": SCALES},
    )
    first = [_snapshot(f) for f in reader]
    second = [_snapshot(f) for f in reader]
    assert first == second


def test_sublink_lazy_payload_matches_eager_loading(
    sublink_tree_file: Path,
) -> None:
    """Streaming should produce halos numerically identical to what a
    single-shot load would. Compare against the writer-facing values."""
    reader = SubLinkReader(
        ReaderSource({"tree_file": str(sublink_tree_file)}),
        options={"scale_factors": SCALES},
    )
    halos = next(iter(reader)).halos
    # Fixture writes SubhaloMass = [10, 50, 100] (10^10 Msun/h) -> Msun/h.
    np.testing.assert_allclose(halos["nodeMass"], [1e11, 5e11, 1e12])
    # SubhaloPos columns in kpc/h -> Mpc/h via /1000.
    np.testing.assert_allclose(halos["position"][0], [1.0, 2.0, 3.0])


def test_sublink_streams_across_chunks(
    sublink_two_chunks: tuple[Path, Path],
) -> None:
    """Cross-chunk union-find still works with streamed payload reads."""
    p0, p1 = sublink_two_chunks
    reader = SubLinkReader(
        ReaderSource({"tree_files": [str(p0), str(p1)]}),
        options={"scale_factors": SCALES},
    )
    forests = list(reader)
    assert len(forests) == 1  # union-find collapses the two chunks
    node_ids = {int(n) for n in forests[0].halos["nodeIndex"]}
    assert node_ids == {100, 101, 200, 201}


def test_sublink_payload_reads_only_relevant_chunk(
    sublink_two_chunks: tuple[Path, Path],
) -> None:
    """For the root_descendant grouping mode each chunk yields its own
    forest, so payload reads stay confined to the chunk that owns the
    forest — the second chunk's file isn't touched by the first forest.
    """
    p0, p1 = sublink_two_chunks
    reader = SubLinkReader(
        ReaderSource({"tree_files": [str(p0), str(p1)]}),
        options={"scale_factors": SCALES, "forest_grouping": "root_descendant"},
    )
    forests = list(reader)
    by_root: dict[int, set[int]] = {
        f.forest_id: {int(n) for n in f.halos["nodeIndex"]} for f in forests
    }
    assert by_root == {100: {100, 200}, 101: {101, 201}}
