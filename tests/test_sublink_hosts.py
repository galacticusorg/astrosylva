"""Tests for SubLink host-pointer resolution."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from astrosylva.exceptions import ReaderError
from astrosylva.readers import ReaderSource
from astrosylva.readers.sublink import SubLinkReader

# All host-resolution fixtures use the same SnapNums.
SCALE_FACTORS = {0: 0.5, 1: 1.0}


def _source(path: Path) -> ReaderSource:
    return ReaderSource({"tree_file": str(path)})


def _opts(**extra: object) -> dict[str, object]:
    return {"scale_factors": SCALE_FACTORS, **extra}


def _hosts_by_id(forest) -> dict[int, int]:
    return {int(h["nodeIndex"]): int(h["hostIndex"]) for h in forest.halos}


def test_hosts_from_field_by_default(sublink_hosts_file: Path) -> None:
    """With /FirstSubhaloInFOFGroupID present, auto mode uses it."""
    reader = SubLinkReader(_source(sublink_hosts_file), options=_opts())
    forest = next(iter(reader))
    by_id = _hosts_by_id(forest)
    # Centrals point to themselves; satellites point to their centrals.
    assert by_id == {100: 100, 101: 100, 200: 200, 201: 200}


def test_hosts_fof_compute_when_field_missing(
    sublink_hosts_fof_only_file: Path,
) -> None:
    """Auto cascades to FOF computation when the field is absent."""
    reader = SubLinkReader(_source(sublink_hosts_fof_only_file), options=_opts())
    forest = next(iter(reader))
    assert _hosts_by_id(forest) == {100: 100, 101: 100, 200: 200, 201: 200}


def test_hosts_explicit_fof_compute(sublink_hosts_file: Path) -> None:
    """`fof_compute` ignores the field even when present."""
    reader = SubLinkReader(
        _source(sublink_hosts_file), options=_opts(host_resolution="fof_compute")
    )
    forest = next(iter(reader))
    assert _hosts_by_id(forest) == {100: 100, 101: 100, 200: 200, 201: 200}


def test_hosts_self_mode(sublink_hosts_file: Path) -> None:
    reader = SubLinkReader(_source(sublink_hosts_file), options=_opts(host_resolution="self"))
    forest = next(iter(reader))
    by_id = _hosts_by_id(forest)
    # Every subhalo is its own host.
    assert all(nid == hid for nid, hid in by_id.items())


def test_hosts_field_mode_requires_field(sublink_hosts_fof_only_file: Path) -> None:
    reader = SubLinkReader(
        _source(sublink_hosts_fof_only_file), options=_opts(host_resolution="field")
    )
    with pytest.raises(ReaderError, match="FirstSubhaloInFOFGroupID"):
        list(reader)


def test_hosts_fof_compute_requires_fof_columns(
    sublink_hosts_no_info_file: Path,
) -> None:
    reader = SubLinkReader(
        _source(sublink_hosts_no_info_file),
        options=_opts(host_resolution="fof_compute"),
    )
    with pytest.raises(ReaderError, match="SubhaloGrNr"):
        list(reader)


def test_hosts_auto_falls_back_to_self_with_warning(
    sublink_hosts_no_info_file: Path,
) -> None:
    reader = SubLinkReader(_source(sublink_hosts_no_info_file), options=_opts())
    with pytest.warns(UserWarning, match="falling back to self-host"):
        forest = next(iter(reader))
    by_id = _hosts_by_id(forest)
    assert all(nid == hid for nid, hid in by_id.items())


def test_hosts_cross_chunk_pointer_remapped_to_self(tmp_path: Path) -> None:
    """A FirstSubhaloInFOFGroupID pointing outside the forest -> self."""
    path = tmp_path / "tree_cross_chunk.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("SubhaloID", data=np.array([100, 101], dtype=np.int64))
        f.create_dataset("DescendantID", data=np.array([-1, -1], dtype=np.int64))
        f.create_dataset("FirstProgenitorID", data=np.array([-1, -1], dtype=np.int64))
        f.create_dataset("NextProgenitorID", data=np.array([-1, -1], dtype=np.int64))
        f.create_dataset("RootDescendantID", data=np.array([100, 101], dtype=np.int64))
        f.create_dataset("SnapNum", data=np.array([1, 1], dtype=np.int32))
        # 100 is its own host; 101 points to 999 (a subhalo in another chunk).
        f.create_dataset("FirstSubhaloInFOFGroupID", data=np.array([100, 999], dtype=np.int64))
        f.create_dataset("SubhaloMass", data=np.array([10.0, 1.0], dtype=np.float32))
        f.create_dataset("SubhaloHalfmassRad", data=np.array([5.0, 1.0], dtype=np.float32))
        f.create_dataset("SubhaloPos", data=np.zeros((2, 3), dtype=np.float32))
        f.create_dataset("SubhaloVel", data=np.zeros((2, 3), dtype=np.float32))
        f.create_dataset("SubhaloSpin", data=np.zeros((2, 3), dtype=np.float32))

    reader = SubLinkReader(
        ReaderSource({"tree_file": str(path)}),
        options={"scale_factors": {1: 1.0}},
    )
    by_root: dict[int, dict[int, int]] = {}
    for forest in reader:
        by_root[forest.forest_id] = _hosts_by_id(forest)
    # Forest 100: 100 -> 100 (self).
    assert by_root[100] == {100: 100}
    # Forest 101: cross-chunk host (999) -> remapped to self.
    assert by_root[101] == {101: 101}


def test_hosts_rejects_bad_resolution_mode(sublink_hosts_file: Path) -> None:
    with pytest.raises(ReaderError, match="host_resolution"):
        SubLinkReader(_source(sublink_hosts_file), options=_opts(host_resolution="wrong"))
