"""Tests for the LHaloTree binary reader."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from astrosylva.exceptions import ReaderError
from astrosylva.readers import ReaderSource
from astrosylva.readers.lhalotree import LHALO_HALO_DTYPE, LHaloTreeReader

SCALES = {0: 0.5, 1: 1.0}


def _source(path: Path, **extra: object) -> ReaderSource:
    return ReaderSource({"tree_file": str(path), **extra})


def test_single_tree_yields_one_forest(lhalotree_single_tree: Path) -> None:
    reader = LHaloTreeReader(_source(lhalotree_single_tree), options={"scale_factors": SCALES})
    forests = list(reader)
    assert len(forests) == 1
    forest = forests[0]
    assert forest.forest_id == 0  # global counter starts at 0
    assert forest.n_halos == 4


def test_single_tree_assigns_contiguous_node_indices(lhalotree_single_tree: Path) -> None:
    reader = LHaloTreeReader(_source(lhalotree_single_tree), options={"scale_factors": SCALES})
    forest = next(iter(reader))
    np.testing.assert_array_equal(forest.halos["nodeIndex"], [0, 1, 2, 3])


def test_single_tree_remaps_descendants(lhalotree_single_tree: Path) -> None:
    """Local Descendant -> global nodeIndex, with -1 preserved."""
    reader = LHaloTreeReader(_source(lhalotree_single_tree), options={"scale_factors": SCALES})
    forest = next(iter(reader))
    # Halos 0 and 1 are roots (desc=-1); halos 2 and 3 descend into 0 and 1.
    np.testing.assert_array_equal(forest.halos["descendantIndex"], [-1, -1, 0, 1])


def test_single_tree_remaps_hosts_for_satellite(lhalotree_single_tree: Path) -> None:
    """FirstHaloInFOFgroup -> global hostIndex; satellites point at centrals."""
    reader = LHaloTreeReader(_source(lhalotree_single_tree), options={"scale_factors": SCALES})
    forest = next(iter(reader))
    # Halos 0, 2 are centrals (host=self). Halos 1, 3 are satellites pointing
    # at the central in their snap.
    np.testing.assert_array_equal(forest.halos["hostIndex"], [0, 0, 2, 2])


def test_single_tree_expansion_factor_from_snap_table(
    lhalotree_single_tree: Path,
) -> None:
    reader = LHaloTreeReader(_source(lhalotree_single_tree), options={"scale_factors": SCALES})
    forest = next(iter(reader))
    # SnapNum = [1, 1, 0, 0] -> a = [1.0, 1.0, 0.5, 0.5]
    np.testing.assert_array_equal(forest.halos["expansionFactor"], [1.0, 1.0, 0.5, 0.5])


def test_single_tree_mass_scaled_by_1e10(lhalotree_single_tree: Path) -> None:
    """Mvir is stored in 10^10 Msun/h units; reader rescales to Msun/h."""
    reader = LHaloTreeReader(_source(lhalotree_single_tree), options={"scale_factors": SCALES})
    forest = next(iter(reader))
    np.testing.assert_allclose(
        forest.halos["nodeMass"], [10.0 * 1e10, 1.0 * 1e10, 8.0 * 1e10, 0.8 * 1e10]
    )


def test_single_tree_vectors_pass_through(lhalotree_single_tree: Path) -> None:
    reader = LHaloTreeReader(_source(lhalotree_single_tree), options={"scale_factors": SCALES})
    forest = next(iter(reader))
    np.testing.assert_allclose(
        forest.halos["position"],
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    )
    np.testing.assert_allclose(
        forest.halos["angularMomentum"],
        [[1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]],
    )
    # Dimensionless spin is not stored in LHaloTree; reader leaves zero.
    np.testing.assert_array_equal(forest.halos["spin"], [0.0, 0.0, 0.0, 0.0])


def test_two_trees_in_one_file(lhalotree_two_trees: Path) -> None:
    reader = LHaloTreeReader(_source(lhalotree_two_trees), options={"scale_factors": SCALES})
    forests = list(reader)
    assert len(forests) == 2
    # nodeIndex counter is contiguous across trees.
    assert forests[0].forest_id == 0
    assert forests[1].forest_id == 2
    np.testing.assert_array_equal(forests[0].halos["nodeIndex"], [0, 1])
    np.testing.assert_array_equal(forests[1].halos["nodeIndex"], [2, 3])


def test_two_chunks_globally_unique_indices(
    lhalotree_two_chunks: tuple[Path, Path],
) -> None:
    p0, p1 = lhalotree_two_chunks
    reader = LHaloTreeReader(
        ReaderSource({"tree_files": [str(p0), str(p1)]}),
        options={"scale_factors": SCALES},
    )
    forests = list(reader)
    # Chunk 0: one tree of 2 halos (indices 0, 1).
    # Chunk 1: one tree of 1 halo  (index 2).
    assert [f.forest_id for f in forests] == [0, 2]
    np.testing.assert_array_equal(forests[0].halos["nodeIndex"], [0, 1])
    np.testing.assert_array_equal(forests[1].halos["nodeIndex"], [2])


def test_missing_scale_factor_table_raises(lhalotree_single_tree: Path) -> None:
    reader = LHaloTreeReader(_source(lhalotree_single_tree))
    with pytest.raises(ReaderError, match="scale-factor table"):
        list(reader)


def test_strict_false_fills_with_nan(lhalotree_single_tree: Path) -> None:
    reader = LHaloTreeReader(
        _source(lhalotree_single_tree),
        options={"strict_scale_factors": False},
    )
    with pytest.warns(UserWarning, match="scale-factor table"):
        forest = next(iter(reader))
    assert np.all(np.isnan(forest.halos["expansionFactor"]))


def test_neither_source_key_raises(tmp_path: Path) -> None:
    reader = LHaloTreeReader(ReaderSource({}), options={"scale_factors": SCALES})
    with pytest.raises(ReaderError, match="tree_file"):
        list(reader)


def test_truncated_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.bin"
    path.write_bytes(b"\x00")  # less than 8 bytes
    reader = LHaloTreeReader(_source(path), options={"scale_factors": SCALES})
    with pytest.raises(ReaderError, match="Truncated LHaloTree header"):
        list(reader)


def test_per_tree_sum_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.bin"
    with path.open("wb") as fh:
        fh.write(struct.pack("<ii", 1, 3))  # claims 3 halos
        np.array([2], dtype="<i4").tofile(fh)  # but only 2 in NhalosPerTree
        np.zeros(3, dtype=LHALO_HALO_DTYPE).tofile(fh)
    reader = LHaloTreeReader(_source(path), options={"scale_factors": SCALES})
    with pytest.raises(ReaderError, match="NhalosPerTree sum"):
        list(reader)


def test_out_of_bounds_descendant_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad_desc.bin"
    halos = np.zeros(2, dtype=LHALO_HALO_DTYPE)
    halos["Descendant"] = [5, -1]  # 5 is out of range for a 2-halo tree
    halos["FirstHaloInFOFgroup"] = [0, 1]
    halos["SnapNum"] = [0, 1]
    with path.open("wb") as fh:
        fh.write(struct.pack("<ii", 1, 2))
        np.array([2], dtype="<i4").tofile(fh)
        halos.tofile(fh)
    reader = LHaloTreeReader(_source(path), options={"scale_factors": SCALES})
    with pytest.raises(ReaderError, match="Descendant pointer out of bounds"):
        list(reader)
