"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

CT_HEADER = (
    "#scale(0) id(1) desc_scale(2) desc_id(3) num_prog(4) pid(5) upid(6) "
    "desc_pid(7) phantom(8) sam_mvir(9) mvir(10) rvir(11) rs(12) vrms(13) "
    "mmp(14) scale_of_last_MM(15) vmax(16) x(17) y(18) z(19) vx(20) vy(21) "
    "vz(22) Jx(23) Jy(24) Jz(25) Spin(26)\n"
    "#Omega_M = 0.27; Omega_L = 0.73; h0 = 0.7\n"
    "#Full box size = 100.000000 Mpc/h\n"
    "#Tree 0 0 0\n"
    "2\n"
)

# Two trees, each with two halos. rs is in kpc/h (CT convention).
TREE_1001 = (
    "#tree 1001\n"
    "1.0 1 -1 -1 1 -1 -1 -1 0 1e12 1e12 100 10 50 1 0.9 200 5.0 5.0 5.0 "
    "100 100 100 1e10 1e10 1e10 0.05\n"
    "0.5 2 1.0 1 0 -1 -1 -1 0 5e11 5e11 80 8 40 1 0.9 150 5.1 5.1 5.1 "
    "100 100 100 5e9 5e9 5e9 0.04\n"
)
TREE_2001 = (
    "#tree 2001\n"
    "1.0 3 -1 -1 1 -1 -1 -1 0 2e12 2e12 120 15 60 1 0.9 250 10.0 10.0 10.0 "
    "200 200 200 2e10 2e10 2e10 0.06\n"
    "0.5 4 1.0 3 0 3 3 3 0 1e12 1e12 100 12 50 1 0.9 200 10.1 10.1 10.1 "
    "200 200 200 1e10 1e10 1e10 0.05\n"
)


@pytest.fixture
def ct_data_dir(tmp_path: Path) -> Path:
    """Synthesise a tiny Consistent-Trees dataset with valid byte offsets."""
    out = tmp_path / "ctrees"
    out.mkdir()

    tree_file = out / "tree_0_0_0.dat"
    contents = CT_HEADER
    offset_1001 = len(contents.encode()) + len(b"#tree 1001\n")
    contents += TREE_1001
    offset_2001 = len(contents.encode()) + len(b"#tree 2001\n")
    contents += TREE_2001
    tree_file.write_bytes(contents.encode())

    # locations.dat: tree_root_id  file_id  offset  filename
    (out / "locations.dat").write_text(
        "TreeRootID FileID Offset Filename\n"
        f"1001 0 {offset_1001} tree_0_0_0.dat\n"
        f"2001 0 {offset_2001} tree_0_0_0.dat\n"
    )

    # forests.list: tree_root_id forest_id weight
    (out / "forests.list").write_text("TreeRootID ForestID Weight\n1001 100 1.0\n2001 100 1.0\n")
    return out


@pytest.fixture
def sublink_tree_file(tmp_path: Path) -> Path:
    """Synthesise a minimal SubLink-style HDF5 tree file.

    Three subhalos belonging to one forest (RootDescendantID = 100) at
    SnapNum 2, 1, 0, each the sole member of its FOF group (so each is
    its own central). SubhaloMass is in 10^10 Msun/h, positions in kpc/h.
    """
    path = tmp_path / "tree_extended.0.hdf5"
    n = 3
    with h5py.File(path, "w") as f:
        f.create_dataset("SubhaloID", data=np.array([300, 200, 100], dtype=np.int64))
        f.create_dataset("DescendantID", data=np.array([200, 100, -1], dtype=np.int64))
        f.create_dataset("FirstProgenitorID", data=np.array([-1, 300, 200], dtype=np.int64))
        f.create_dataset("NextProgenitorID", data=np.full(n, -1, dtype=np.int64))
        f.create_dataset("RootDescendantID", data=np.full(n, 100, dtype=np.int64))
        f.create_dataset("TreeID", data=np.full(n, 1, dtype=np.int64))
        f.create_dataset("SnapNum", data=np.array([2, 1, 0], dtype=np.int32))
        f.create_dataset("FirstSubhaloInFOFGroupID", data=np.array([300, 200, 100], dtype=np.int64))
        f.create_dataset("SubhaloGrNr", data=np.array([0, 0, 0], dtype=np.int32))
        f.create_dataset("SubfindID", data=np.array([0, 0, 0], dtype=np.int32))
        f.create_dataset("SubhaloMass", data=np.array([10.0, 50.0, 100.0], dtype=np.float32))
        f.create_dataset("SubhaloHalfmassRad", data=np.array([5.0, 10.0, 20.0], dtype=np.float32))
        f.create_dataset(
            "SubhaloPos",
            data=np.array(
                [[1000.0, 2000.0, 3000.0], [1100.0, 2100.0, 3100.0], [1200.0, 2200.0, 3200.0]],
                dtype=np.float32,
            ),
        )
        f.create_dataset(
            "SubhaloVel",
            data=np.array(
                [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0]],
                dtype=np.float32,
            ),
        )
        f.create_dataset(
            "SubhaloSpin",
            data=np.array(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                dtype=np.float32,
            ),
        )
    return path


def _write_sublink_hosts_fixture(
    path: Path,
    *,
    include_first_sub_in_fof: bool,
    include_grnr_subfind: bool,
) -> None:
    """Write a 4-subhalo fixture with a non-trivial FOF group at each snap.

    Layout (all in one forest, RootDescendantID = 100 for simplicity):

    Snap 1: 100 (central, FOF 0, SubfindID 0), 101 (satellite, FOF 0, SubfindID 1)
    Snap 0: 200 (central, FOF 0, SubfindID 0), 201 (satellite, FOF 0, SubfindID 1)

    Descendants: 200 -> 100, 201 -> 101, 100/101 -> -1.
    """
    with h5py.File(path, "w") as f:
        ids = np.array([100, 101, 200, 201], dtype=np.int64)
        desc = np.array([-1, -1, 100, 101], dtype=np.int64)
        snaps = np.array([1, 1, 0, 0], dtype=np.int32)
        f.create_dataset("SubhaloID", data=ids)
        f.create_dataset("DescendantID", data=desc)
        f.create_dataset("FirstProgenitorID", data=np.array([200, 201, -1, -1], dtype=np.int64))
        f.create_dataset("NextProgenitorID", data=np.full(4, -1, dtype=np.int64))
        f.create_dataset("RootDescendantID", data=np.full(4, 100, dtype=np.int64))
        f.create_dataset("TreeID", data=np.full(4, 1, dtype=np.int64))
        f.create_dataset("SnapNum", data=snaps)
        f.create_dataset("SubhaloMass", data=np.array([100.0, 10.0, 80.0, 8.0], dtype=np.float32))
        f.create_dataset(
            "SubhaloHalfmassRad", data=np.array([20.0, 5.0, 18.0, 4.0], dtype=np.float32)
        )
        f.create_dataset(
            "SubhaloPos",
            data=np.array([[0.0] * 3, [1.0] * 3, [0.0] * 3, [1.0] * 3], dtype=np.float32),
        )
        f.create_dataset("SubhaloVel", data=np.zeros((4, 3), dtype=np.float32))
        f.create_dataset("SubhaloSpin", data=np.zeros((4, 3), dtype=np.float32))
        if include_first_sub_in_fof:
            # Central of each FOF group is the lower SubhaloID at the same snap.
            f.create_dataset(
                "FirstSubhaloInFOFGroupID",
                data=np.array([100, 100, 200, 200], dtype=np.int64),
            )
        if include_grnr_subfind:
            f.create_dataset("SubhaloGrNr", data=np.array([0, 0, 0, 0], dtype=np.int32))
            f.create_dataset("SubfindID", data=np.array([0, 1, 0, 1], dtype=np.int32))


@pytest.fixture
def sublink_hosts_file(tmp_path: Path) -> Path:
    """SubLink fixture with both /FirstSubhaloInFOFGroupID and FOF columns."""
    path = tmp_path / "tree_hosts_full.hdf5"
    _write_sublink_hosts_fixture(path, include_first_sub_in_fof=True, include_grnr_subfind=True)
    return path


@pytest.fixture
def sublink_hosts_fof_only_file(tmp_path: Path) -> Path:
    """SubLink fixture without /FirstSubhaloInFOFGroupID, only FOF columns."""
    path = tmp_path / "tree_hosts_fof_only.hdf5"
    _write_sublink_hosts_fixture(path, include_first_sub_in_fof=False, include_grnr_subfind=True)
    return path


@pytest.fixture
def sublink_hosts_no_info_file(tmp_path: Path) -> Path:
    """SubLink fixture with no host-resolution metadata at all."""
    path = tmp_path / "tree_hosts_none.hdf5"
    _write_sublink_hosts_fixture(path, include_first_sub_in_fof=False, include_grnr_subfind=False)
    return path
