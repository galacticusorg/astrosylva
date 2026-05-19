"""Tests for the SubLink reader's scale-factor lookup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astrosylva.exceptions import ReaderError
from astrosylva.readers import ReaderSource
from astrosylva.readers.sublink import SubLinkReader

# Fixture from conftest provides three subhalos at SnapNum 2, 1, 0
# in one forest (RootDescendantID = 100).
SCALE_FACTORS = {0: 0.05, 1: 0.5, 2: 1.0}
REDSHIFTS = {0: 19.0, 1: 1.0, 2: 0.0}  # implies a = 1/(1+z) = 0.05, 0.5, 1.0


def _source(path: Path, **extra: object) -> ReaderSource:
    return ReaderSource({"tree_file": str(path), **extra})


def _expansion_for_snaps(forest, snap_to_a: dict[int, float]) -> np.ndarray:
    # Halos in the fixture are written in SnapNum order 2, 1, 0.
    return np.array([snap_to_a[s] for s in (2, 1, 0)])


def test_sublink_scale_factors_inline_dict(sublink_tree_file: Path) -> None:
    reader = SubLinkReader(_source(sublink_tree_file), options={"scale_factors": SCALE_FACTORS})
    forests = list(reader)
    assert len(forests) == 1
    assert forests[0].forest_id == 100
    np.testing.assert_allclose(
        forests[0].halos["expansionFactor"],
        _expansion_for_snaps(forests[0], SCALE_FACTORS),
    )


def test_sublink_routes_halfmassrad_to_half_mass_radius(sublink_tree_file: Path) -> None:
    """SubhaloHalfmassRad is a half-mass radius, not an NFW Rs."""
    reader = SubLinkReader(_source(sublink_tree_file), options={"scale_factors": SCALE_FACTORS})
    forest = next(iter(reader))
    # Fixture writes SubhaloHalfmassRad = [5.0, 10.0, 20.0] kpc/h.
    np.testing.assert_allclose(forest.halos["halfMassRadius"], [0.005, 0.010, 0.020])
    # scaleRadius is unknown for SubLink.
    assert np.all(np.isnan(forest.halos["scaleRadius"]))


def test_sublink_redshifts_inline_dict(sublink_tree_file: Path) -> None:
    reader = SubLinkReader(_source(sublink_tree_file), options={"redshifts": REDSHIFTS})
    forest = next(iter(reader))
    np.testing.assert_allclose(
        forest.halos["expansionFactor"],
        np.array([1.0 / (1.0 + REDSHIFTS[s]) for s in (2, 1, 0)]),
    )


def test_sublink_scale_factors_from_file(sublink_tree_file: Path, tmp_path: Path) -> None:
    table = tmp_path / "snaps_a.txt"
    table.write_text("# SnapNum scale_factor\n0 0.05\n1 0.5\n2 1.0\n")
    reader = SubLinkReader(_source(sublink_tree_file, snapshot_table=str(table)))
    forest = next(iter(reader))
    np.testing.assert_allclose(forest.halos["expansionFactor"], [1.0, 0.5, 0.05])


def test_sublink_scale_factors_from_a_list_file(sublink_tree_file: Path, tmp_path: Path) -> None:
    """The Millennium-style 1-column ``a_list`` format is auto-detected:
    snap number is the 0-based line index."""
    table = tmp_path / "a_list.txt"
    table.write_text("0.05\n0.5\n1.0\n")  # snap 0 -> 0.05, snap 1 -> 0.5, snap 2 -> 1.0
    reader = SubLinkReader(_source(sublink_tree_file, snapshot_table=str(table)))
    forest = next(iter(reader))
    # Halos appear in snap order [2, 1, 0] in the fixture, so a = [1.0, 0.5, 0.05].
    np.testing.assert_allclose(forest.halos["expansionFactor"], [1.0, 0.5, 0.05])


def test_sublink_snapshot_table_rejects_mixed_widths(
    sublink_tree_file: Path, tmp_path: Path
) -> None:
    """A file with inconsistent column widths raises a clear error."""
    table = tmp_path / "bad.txt"
    table.write_text("0.05\n1 0.5\n")  # one 1-token, one 2-token row
    with pytest.raises(ReaderError, match="inconsistent row widths"):
        SubLinkReader(_source(sublink_tree_file, snapshot_table=str(table)))


def test_sublink_redshift_table_quantity(sublink_tree_file: Path, tmp_path: Path) -> None:
    table = tmp_path / "snaps_z.txt"
    table.write_text("0 19.0\n1 1.0\n2 0.0\n")
    reader = SubLinkReader(
        _source(sublink_tree_file, snapshot_table=str(table)),
        options={"snapshot_table_quantity": "redshift"},
    )
    forest = next(iter(reader))
    np.testing.assert_allclose(forest.halos["expansionFactor"], [1.0, 0.5, 0.05])


def test_sublink_missing_table_raises_by_default(sublink_tree_file: Path) -> None:
    reader = SubLinkReader(_source(sublink_tree_file))
    with pytest.raises(ReaderError, match="scale-factor table"):
        list(reader)


def test_sublink_missing_table_warns_when_non_strict(sublink_tree_file: Path) -> None:
    reader = SubLinkReader(_source(sublink_tree_file), options={"strict_scale_factors": False})
    with pytest.warns(UserWarning, match="scale-factor table"):
        forest = next(iter(reader))
    assert np.all(np.isnan(forest.halos["expansionFactor"]))


def test_sublink_missing_snap_raises(sublink_tree_file: Path) -> None:
    reader = SubLinkReader(
        _source(sublink_tree_file), options={"scale_factors": {0: 0.05, 1: 0.5}}
    )  # snap 2 missing
    with pytest.raises(ReaderError, match=r"missing entries for SnapNum: \[2\]"):
        list(reader)


def test_sublink_missing_snap_warns_when_non_strict(sublink_tree_file: Path) -> None:
    reader = SubLinkReader(
        _source(sublink_tree_file),
        options={"scale_factors": {0: 0.05, 1: 0.5}, "strict_scale_factors": False},
    )
    with pytest.warns(UserWarning, match=r"missing entries for SnapNum: \[2\]"):
        forest = next(iter(reader))
    # Snap-2 halo should be NaN; snap-0 and snap-1 should be filled.
    a = forest.halos["expansionFactor"]
    assert np.isnan(a[0])  # snap 2
    assert a[1] == 0.5  # snap 1
    assert a[2] == 0.05  # snap 0


def test_sublink_rejects_both_scales_and_redshifts(sublink_tree_file: Path) -> None:
    with pytest.raises(ReaderError, match="at most one"):
        SubLinkReader(
            _source(sublink_tree_file),
            options={"scale_factors": {0: 0.1}, "redshifts": {0: 9.0}},
        )


def test_sublink_rejects_bad_table_quantity(sublink_tree_file: Path, tmp_path: Path) -> None:
    table = tmp_path / "t.txt"
    table.write_text("0 1.0\n")
    with pytest.raises(ReaderError, match="snapshot_table_quantity"):
        SubLinkReader(
            _source(sublink_tree_file, snapshot_table=str(table)),
            options={"snapshot_table_quantity": "wrong"},
        )


def test_sublink_rejects_missing_table_file(sublink_tree_file: Path, tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(ReaderError, match="snapshot table file not found"):
        SubLinkReader(_source(sublink_tree_file, snapshot_table=str(missing)))
