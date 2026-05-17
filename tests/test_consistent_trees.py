"""Tests for the Consistent-Trees reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astrosylva.exceptions import ReaderError
from astrosylva.readers import ReaderSource
from astrosylva.readers.consistent_trees import (
    ConsistentTreesReader,
    _parse_header_columns,
)


def _make_source(ct_data_dir: Path) -> ReaderSource:
    return ReaderSource(
        {
            "input_path": ct_data_dir,
            "forests_path": ct_data_dir / "forests.list",
            "locations_path": ct_data_dir / "locations.dat",
        }
    )


def test_parse_header_columns() -> None:
    header = "#scale(0) id(1) desc_scale(2) desc_id(3) mvir(10) rs(12)\n"
    cols = _parse_header_columns(header)
    assert cols == {
        "scale": 0,
        "id": 1,
        "desc_scale": 2,
        "desc_id": 3,
        "mvir": 10,
        "rs": 12,
    }


def test_reader_reports_one_forest(ct_data_dir: Path) -> None:
    reader = ConsistentTreesReader(_make_source(ct_data_dir))
    assert len(reader) == 1


def test_reader_yields_all_halos(ct_data_dir: Path) -> None:
    reader = ConsistentTreesReader(_make_source(ct_data_dir))
    forests = list(reader)
    assert len(forests) == 1
    forest = forests[0]
    assert forest.forest_id == 100
    assert forest.n_halos == 4
    assert forest.weight == 1.0


def test_reader_converts_units_and_remaps_host(ct_data_dir: Path) -> None:
    reader = ConsistentTreesReader(_make_source(ct_data_dir))
    forest = next(iter(reader))
    halos = forest.halos

    # scaleRadius converted from kpc/h to Mpc/h.
    np.testing.assert_allclose(halos["scaleRadius"], np.array([0.010, 0.008, 0.015, 0.012]))
    # Position untouched (already Mpc/h).
    np.testing.assert_allclose(halos["position"][0], [5.0, 5.0, 5.0])
    # Mass passed through (Msun/h).
    np.testing.assert_allclose(halos["nodeMass"][0], 1e12)

    # No-host sentinel (-1) is remapped to nodeIndex (Galacticus convention).
    # Halo 1 had pid=-1 -> hostIndex should equal nodeIndex (=1).
    # Halo 4 had pid=3  -> hostIndex stays 3.
    by_id = {int(h["nodeIndex"]): h for h in halos}
    assert int(by_id[1]["hostIndex"]) == 1
    assert int(by_id[3]["hostIndex"]) == 3
    assert int(by_id[4]["hostIndex"]) == 3


def test_reader_metadata_introspects_cosmology(ct_data_dir: Path) -> None:
    reader = ConsistentTreesReader(_make_source(ct_data_dir))
    meta = reader.metadata()
    assert meta.cosmology["Omega0"] == pytest.approx(0.27)
    assert meta.cosmology["OmegaLambda"] == pytest.approx(0.73)
    assert meta.cosmology["HubbleParam"] == pytest.approx(0.7)


def test_reader_rejects_bad_options(ct_data_dir: Path) -> None:
    with pytest.raises(ReaderError, match="host_source"):
        ConsistentTreesReader(_make_source(ct_data_dir), {"host_source": "wrong"})


def test_reader_host_source_upid(ct_data_dir: Path) -> None:
    reader = ConsistentTreesReader(_make_source(ct_data_dir), {"host_source": "upid"})
    forest = next(iter(reader))
    by_id = {int(h["nodeIndex"]): h for h in forest.halos}
    # Halo 4 had upid=3 in the fixture (same as pid here).
    assert int(by_id[4]["hostIndex"]) == 3
