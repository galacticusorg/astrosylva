"""Tests for the AHF reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astrosylva.exceptions import ReaderError
from astrosylva.readers import ReaderSource
from astrosylva.readers.ahf import AHFReader


def _forests_by_id(reader: AHFReader) -> dict[int, set[int]]:
    return {f.forest_id: {int(h["nodeIndex"]) for h in f.halos} for f in reader}


def _hosts_by_id(reader: AHFReader) -> dict[int, int]:
    out: dict[int, int] = {}
    for forest in reader:
        for h in forest.halos:
            out[int(h["nodeIndex"])] = int(h["hostIndex"])
    return out


def test_union_find_splits_physically_independent_forests(
    ahf_two_independent_forests: list[dict[str, object]],
) -> None:
    """Two FOF-disconnected halos with independent progenitors must land in
    different forests, not be lumped together as the old reader did."""
    reader = AHFReader(ReaderSource({"snapshots": ahf_two_independent_forests}))
    forests = _forests_by_id(reader)
    assert forests == {
        100: {100, 101, 1100, 1101},
        200: {200, 1200},
    }


def test_satellite_host_resolves_to_central(
    ahf_two_independent_forests: list[dict[str, object]],
) -> None:
    """Satellites should point at their FOF central within the same forest;
    centrals point at themselves (Galacticus's no-host convention)."""
    reader = AHFReader(ReaderSource({"snapshots": ahf_two_independent_forests}))
    hosts = _hosts_by_id(reader)
    # Forest 100: 100 and 1100 are centrals; 101/1101 point at them.
    assert hosts[100] == 100
    assert hosts[1100] == 1100
    assert hosts[101] == 100
    assert hosts[1101] == 1100
    # Forest 200: independent central + its progenitor.
    assert hosts[200] == 200
    assert hosts[1200] == 1200


def test_descendant_links_applied_from_mtree(
    ahf_two_independent_forests: list[dict[str, object]],
) -> None:
    reader = AHFReader(ReaderSource({"snapshots": ahf_two_independent_forests}))
    descendants: dict[int, int] = {}
    for forest in reader:
        for h in forest.halos:
            descendants[int(h["nodeIndex"])] = int(h["descendantIndex"])
    # Snap-0 halos point at their snap-1 descendants.
    assert descendants[1100] == 100
    assert descendants[1101] == 101
    assert descendants[1200] == 200
    # Snap-1 halos have no descendant.
    assert descendants[100] == -1
    assert descendants[200] == -1


def test_expansion_factor_per_snapshot(
    ahf_two_independent_forests: list[dict[str, object]],
) -> None:
    reader = AHFReader(ReaderSource({"snapshots": ahf_two_independent_forests}))
    by_id: dict[int, float] = {}
    for forest in reader:
        for h in forest.halos:
            by_id[int(h["nodeIndex"])] = float(h["expansionFactor"])
    # Snap 0 a=0.5, snap 1 a=1.0.
    assert by_id[1100] == 0.5
    assert by_id[100] == 1.0
    assert by_id[1200] == 0.5
    assert by_id[200] == 1.0


def test_single_snapshot_yields_one_forest_per_independent_halo(
    ahf_single_snapshot: list[dict[str, object]],
) -> None:
    reader = AHFReader(ReaderSource({"snapshots": ahf_single_snapshot}))
    forests = _forests_by_id(reader)
    assert forests == {100: {100}, 200: {200}, 300: {300}}


def test_empty_snapshots_raises() -> None:
    reader = AHFReader(ReaderSource({"snapshots": []}))
    with pytest.raises(ReaderError, match="at least one snapshot"):
        list(reader)


def test_missing_halos_file_raises(tmp_path: Path) -> None:
    reader = AHFReader(
        ReaderSource(
            {"snapshots": [{"halos": str(tmp_path / "nope.AHF_halos"), "mtree": None, "a": 1.0}]}
        )
    )
    with pytest.raises(ReaderError, match="AHF halos file not found"):
        list(reader)


def test_mass_passes_through(
    ahf_two_independent_forests: list[dict[str, object]],
) -> None:
    reader = AHFReader(ReaderSource({"snapshots": ahf_two_independent_forests}))
    by_id_mass: dict[int, float] = {}
    for forest in reader:
        for h in forest.halos:
            by_id_mass[int(h["nodeIndex"])] = float(h["nodeMass"])
    assert by_id_mass[100] == 2e12
    assert by_id_mass[1100] == 1e12
    assert by_id_mass[200] == 1.5e12


def test_half_mass_radius_is_nan_for_ahf(
    ahf_two_independent_forests: list[dict[str, object]],
) -> None:
    """AHF's default output has neither half-mass nor NFW scale radius;
    halfMassRadius should be NaN throughout."""
    reader = AHFReader(ReaderSource({"snapshots": ahf_two_independent_forests}))
    for forest in reader:
        assert np.all(np.isnan(forest.halos["halfMassRadius"]))
