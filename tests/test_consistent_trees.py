"""Tests for the Consistent-Trees reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astrosylva.exceptions import ReaderError
from astrosylva.readers import ReaderSource
from astrosylva.readers.consistent_trees import (
    ConsistentTreesReader,
    _find_col,
    _parse_cosmology_header,
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


# An actual CT header copied verbatim from a ytree sample build, used to
# pin down case-variation and trailing-unindexed-column handling.
_YTREE_CT_HEADER = (
    "#scale(0) id(1) desc_scale(2) desc_id(3) num_prog(4) pid(5) upid(6) "
    "desc_pid(7) phantom(8) sam_Mvir(9) Mvir(10) Rvir(11) rs(12) vrms(13) "
    "mmp?(14) scale_of_last_MM(15) vmax(16) x(17) y(18) z(19) vx(20) vy(21) "
    "vz(22) Jx(23) Jy(24) Jz(25) Spin(26) Breadth_first_ID(27) "
    "Depth_first_ID(28) Tree_root_ID(29) Orig_halo_ID(30) Snap_num(31) "
    "Next_coprogenitor_depthfirst_ID(32) Last_progenitor_depthfirst_ID(33) "
    "Last_mainleaf_depthfirst_ID(34) Tidal_Force(35) Tidal_ID(36) "
    "Rs_Klypin Mvir_all M200b M200c M500c M2500c Xoff Voff Spin_Bullock "
    "b_to_a c_to_a A[x] A[y] A[z] b_to_a(500c) c_to_a(500c) A[x](500c) "
    "A[y](500c) A[z](500c) T/|U| M_pe_Behroozi M_pe_Diemer\n"
)


def test_ytree_header_indexed_columns_preserve_case() -> None:
    """The real ytree-sample header capitalises Mvir / Rvir — the parser
    keeps the original casing as the dictionary key."""
    cols = _parse_header_columns(_YTREE_CT_HEADER)
    assert cols["Mvir"] == 10
    assert cols["Rvir"] == 11
    assert cols["Spin"] == 26
    # mmp?(14) has a special char in the name; regex skips it but the
    # position counter advances so following indices stay aligned.
    assert "mmp?" not in cols


def test_ytree_header_trailing_unindexed_columns_get_sequential_indices() -> None:
    """Tokens after Tidal_ID(36) have no (N) — they inherit positions 37+."""
    cols = _parse_header_columns(_YTREE_CT_HEADER)
    assert cols["Rs_Klypin"] == 37
    assert cols["Mvir_all"] == 38
    assert cols["M200b"] == 39
    assert cols["Spin_Bullock"] == 45
    assert cols["M_pe_Behroozi"] == 57
    assert cols["M_pe_Diemer"] == 58
    # Tokens with parens or special chars in the trailing region still
    # occupy a position but aren't stored under a name.
    assert "b_to_a(500c)" not in cols
    assert "T/|U|" in cols  # no parens -> kept


def test_find_col_is_case_insensitive() -> None:
    cols = {"Mvir": 10, "Rs_Klypin": 37}
    assert _find_col(cols, "mvir") == 10
    assert _find_col(cols, "MVIR") == 10
    assert _find_col(cols, "rs_klypin") == 37
    with pytest.raises(KeyError):
        _find_col(cols, "nonexistent")


def test_reader_loads_ytree_style_capitalised_header(tmp_path: Path) -> None:
    """End-to-end smoke: reader succeeds on a CT file whose header uses
    capitalised Mvir / Rvir / Rs_Klypin (unindexed)."""
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    tree_path = ct_dir / "tree_0_0_0.dat"
    contents = _YTREE_CT_HEADER
    contents += "#Tree 0 0 0\n1\n"
    tree_offset = len(contents.encode()) + len(b"#tree 300\n")
    # Build a single 60-column row matching the header. Indices 0..36
    # are the explicitly-numbered columns; 37..59 are the trailing
    # unindexed ones (Rs_Klypin at 37, Mvir_all at 38, ...).
    row = (
        ["1.0", "300", "-1", "-1", "0", "-1", "-1", "-1", "0", "2e12"]  # 0-9
        + ["2e12", "200", "20", "100", "1", "1.0", "250"]  # 10-16
        + ["5.0", "5.0", "5.0"]  # 17-19
        + ["100", "100", "100"]  # 20-22
        + ["0", "0", "0", "0.05"]  # 23-26
        + ["0"] * 10  # 27-36
        + ["10.0"]  # 37 Rs_Klypin
        + ["0"] * 22  # 38-59
    )
    assert len(row) == 60
    contents += "#tree 300\n" + " ".join(row) + "\n"
    tree_path.write_bytes(contents.encode())
    (ct_dir / "locations.dat").write_text(
        f"TreeRootID FileID Offset Filename\n300 0 {tree_offset} tree_0_0_0.dat\n"
    )
    (ct_dir / "forests.list").write_text("TreeRootID ForestID Weight\n300 300 1.0\n")

    reader = ConsistentTreesReader(
        ReaderSource(
            {
                "input_path": str(ct_dir),
                "forests_path": str(ct_dir / "forests.list"),
                "locations_path": str(ct_dir / "locations.dat"),
            }
        )
    )
    forest = next(iter(reader))
    assert forest.n_halos == 1
    assert forest.halos["nodeMass"][0] == 2e12

    # Setting scale_radius_source='rs_klypin' picks up the unindexed
    # column at position 37.
    reader_klypin = ConsistentTreesReader(
        ReaderSource(
            {
                "input_path": str(ct_dir),
                "forests_path": str(ct_dir / "forests.list"),
                "locations_path": str(ct_dir / "locations.dat"),
            }
        ),
        {"scale_radius_source": "rs_klypin"},
    )
    f2 = next(iter(reader_klypin))
    # Rs_Klypin = 10.0 kpc/h -> 0.010 Mpc/h.
    assert f2.halos["scaleRadius"][0] == pytest.approx(0.010)


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


def test_reader_parses_extended_cosmology_header() -> None:
    """``Omega_b`` and ``sigma_8`` in the CT header propagate into metadata."""
    cosmo = _parse_cosmology_header(
        [
            "#Omega_M = 0.27; Omega_L = 0.73; h = 0.7\n",
            "#Omega_b = 0.046; sigma_8 = 0.81\n",
        ]
    )
    assert cosmo == {
        "Omega0": 0.27,
        "OmegaLambda": 0.73,
        "HubbleParam": 0.7,
        "OmegaBaryon": 0.046,
        "sigma_8": 0.81,
    }


def test_reader_cosmology_header_accepts_sigma8_variant_and_uppercase_b() -> None:
    cosmo = _parse_cosmology_header(
        [
            "#sigma8 = 0.82\n",
            "#Omega_B = 0.05\n",
        ]
    )
    assert cosmo == {"sigma_8": 0.82, "OmegaBaryon": 0.05}


def test_reader_rejects_bad_options(ct_data_dir: Path) -> None:
    with pytest.raises(ReaderError, match="host_source"):
        ConsistentTreesReader(_make_source(ct_data_dir), {"host_source": "wrong"})


def test_reader_host_source_upid(ct_data_dir: Path) -> None:
    reader = ConsistentTreesReader(_make_source(ct_data_dir), {"host_source": "upid"})
    forest = next(iter(reader))
    by_id = {int(h["nodeIndex"]): h for h in forest.halos}
    # Halo 4 had upid=3 in the fixture (same as pid here).
    assert int(by_id[4]["hostIndex"]) == 3
