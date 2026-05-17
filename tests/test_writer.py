"""Tests for the Galacticus writer."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from astrosylva.schema import HALO_DTYPE, Forest, Metadata
from astrosylva.writers import GalacticusWriter


def _make_forest(forest_id: int, n: int) -> Forest:
    halos = np.zeros(n, dtype=HALO_DTYPE)
    halos["nodeIndex"] = np.arange(n, dtype=np.int64) + forest_id * 100
    halos["descendantIndex"] = -1
    halos["hostIndex"] = halos["nodeIndex"]
    halos["expansionFactor"] = np.linspace(0.1, 1.0, n)
    halos["nodeMass"] = np.linspace(1e10, 1e13, n)
    halos["scaleRadius"] = np.linspace(0.001, 0.05, n)
    halos["position"] = np.tile(np.arange(n, dtype=np.float64), (3, 1)).T
    halos["velocity"] = halos["position"] * 10
    halos["angularMomentum"] = halos["position"] * 1e10
    halos["spin"] = 0.05
    return Forest(forest_id=forest_id, halos=halos, weight=float(forest_id))


def test_writer_streams_multiple_forests(tmp_path: Path) -> None:
    out = tmp_path / "out.hdf5"
    metadata = Metadata(
        cosmology={"HubbleParam": 0.7, "Omega0": 0.3},
        simulation={"boxSize": 100.0},
    )
    forests = [_make_forest(1, 3), _make_forest(2, 5), _make_forest(3, 1)]

    with GalacticusWriter(out, metadata) as w:
        for forest in forests:
            w.write_forest(forest)

    with h5py.File(out, "r") as f:
        assert f.attrs["formatVersion"] == 2

        halos = f["forestHalos"]
        assert halos["nodeIndex"].shape == (9,)
        assert halos["position"].shape == (9, 3)
        assert halos["halfMassRadius"].shape == (9,)
        np.testing.assert_array_equal(
            halos["nodeIndex"][:],
            np.concatenate([forest.halos["nodeIndex"] for forest in forests]),
        )

        idx = f["forestIndex"]
        np.testing.assert_array_equal(idx["firstNode"][:], [0, 3, 8])
        np.testing.assert_array_equal(idx["numberOfNodes"][:], [3, 5, 1])
        np.testing.assert_array_equal(idx["forestIndex"][:], [1, 2, 3])
        np.testing.assert_array_equal(idx["forestWeight"][:], [1.0, 2.0, 3.0])

        assert f["cosmology"].attrs["HubbleParam"] == 0.7
        assert f["simulation"].attrs["boxSize"] == 100.0


def test_writer_handles_empty_forest(tmp_path: Path) -> None:
    out = tmp_path / "empty.hdf5"
    empty = Forest(forest_id=42, halos=np.empty(0, dtype=HALO_DTYPE))

    with GalacticusWriter(out, Metadata()) as w:
        w.write_forest(empty)

    with h5py.File(out, "r") as f:
        assert f["forestHalos/nodeIndex"].shape == (0,)
        assert f["forestIndex/firstNode"].shape == (1,)
        assert f["forestIndex/numberOfNodes"][0] == 0
