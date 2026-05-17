"""Shared pytest fixtures."""

from __future__ import annotations

import struct
from pathlib import Path

import h5py
import numpy as np
import pytest

from astrosylva.readers.lhalotree import LHALO_HALO_DTYPE

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


def _write_minimal_sublink_chunk(
    path: Path,
    *,
    ids: np.ndarray,
    desc: np.ndarray,
    root_desc: np.ndarray,
    snaps: np.ndarray,
    first_sub_in_fof: np.ndarray,
) -> None:
    """Write a small SubLink-style HDF5 file with zeroed-out physical fields.

    Used for building multi-chunk fixtures where only the topology fields
    matter for the test assertions.
    """
    n = ids.shape[0]
    with h5py.File(path, "w") as f:
        f.create_dataset("SubhaloID", data=ids)
        f.create_dataset("DescendantID", data=desc)
        f.create_dataset("FirstProgenitorID", data=np.full(n, -1, dtype=np.int64))
        f.create_dataset("NextProgenitorID", data=np.full(n, -1, dtype=np.int64))
        f.create_dataset("RootDescendantID", data=root_desc)
        f.create_dataset("TreeID", data=np.arange(n, dtype=np.int64))
        f.create_dataset("SnapNum", data=snaps)
        f.create_dataset("FirstSubhaloInFOFGroupID", data=first_sub_in_fof)
        f.create_dataset("SubhaloMass", data=np.ones(n, dtype=np.float32))
        f.create_dataset("SubhaloHalfmassRad", data=np.ones(n, dtype=np.float32))
        f.create_dataset("SubhaloPos", data=np.zeros((n, 3), dtype=np.float32))
        f.create_dataset("SubhaloVel", data=np.zeros((n, 3), dtype=np.float32))
        f.create_dataset("SubhaloSpin", data=np.zeros((n, 3), dtype=np.float32))


@pytest.fixture
def sublink_two_chunks(tmp_path: Path) -> tuple[Path, Path]:
    """Two SubLink chunks where hosts and progenitors cross chunk boundaries.

    Chunk 0 holds the centrals of two snapshots:

    - SubhaloID 100 (RootDescendantID=100, central FOF, desc=-1, snap 1)
    - SubhaloID 200 (RootDescendantID=100, central FOF, desc=100, snap 0)

    Chunk 1 holds the satellites:

    - SubhaloID 101 (RootDescendantID=101, satellite of 100, desc=-1, snap 1)
    - SubhaloID 201 (RootDescendantID=101, satellite of 200, desc=101, snap 0)

    Both ``FirstSubhaloInFOFGroupID`` pointers in chunk 1 reference
    subhalos that live in chunk 0 — they only resolve when both chunks
    are loaded together.
    """
    p0 = tmp_path / "tree.0.hdf5"
    p1 = tmp_path / "tree.1.hdf5"
    _write_minimal_sublink_chunk(
        p0,
        ids=np.array([100, 200], dtype=np.int64),
        desc=np.array([-1, 100], dtype=np.int64),
        root_desc=np.array([100, 100], dtype=np.int64),
        snaps=np.array([1, 0], dtype=np.int32),
        first_sub_in_fof=np.array([100, 200], dtype=np.int64),
    )
    _write_minimal_sublink_chunk(
        p1,
        ids=np.array([101, 201], dtype=np.int64),
        desc=np.array([-1, 101], dtype=np.int64),
        root_desc=np.array([101, 101], dtype=np.int64),
        snaps=np.array([1, 0], dtype=np.int32),
        first_sub_in_fof=np.array([100, 200], dtype=np.int64),
    )
    return p0, p1


@pytest.fixture
def sublink_split_roots_file(tmp_path: Path) -> Path:
    """SubLink fixture where RootDescendantID alone splits a real forest.

    Layout (snap 1 = final):

    Snap 1: 100 (central FOF 0, desc=-1, RootDescendantID=100)
            101 (satellite FOF 0, desc=-1, RootDescendantID=101)
    Snap 0: 200 (central FOF 0, desc=100, RootDescendantID=100)
            201 (satellite FOF 0, desc=101, RootDescendantID=101)

    With RootDescendantID grouping this becomes two forests
    ({100, 200} and {101, 201}). With union-find on
    descendant + host edges, all four belong to one forest because
    101 and 201 share their FOF central (100 and 200 respectively).
    """
    path = tmp_path / "tree_split_roots.hdf5"
    with h5py.File(path, "w") as f:
        ids = np.array([100, 101, 200, 201], dtype=np.int64)
        f.create_dataset("SubhaloID", data=ids)
        f.create_dataset("DescendantID", data=np.array([-1, -1, 100, 101], dtype=np.int64))
        f.create_dataset("FirstProgenitorID", data=np.array([200, 201, -1, -1], dtype=np.int64))
        f.create_dataset("NextProgenitorID", data=np.full(4, -1, dtype=np.int64))
        f.create_dataset("RootDescendantID", data=np.array([100, 101, 100, 101], dtype=np.int64))
        f.create_dataset("TreeID", data=np.array([1, 2, 1, 2], dtype=np.int64))
        f.create_dataset("SnapNum", data=np.array([1, 1, 0, 0], dtype=np.int32))
        f.create_dataset(
            "FirstSubhaloInFOFGroupID",
            data=np.array([100, 100, 200, 200], dtype=np.int64),
        )
        f.create_dataset("SubhaloGrNr", data=np.array([0, 0, 0, 0], dtype=np.int32))
        f.create_dataset("SubfindID", data=np.array([0, 1, 0, 1], dtype=np.int32))
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
    return path


def _make_lhalo_halo(**fields: object) -> np.ndarray:
    h = np.zeros(1, dtype=LHALO_HALO_DTYPE)[0]
    for k, v in fields.items():
        h[k] = v
    return h


def _write_lhalotree_file(path: Path, trees: list[np.ndarray]) -> None:
    """Write an LHaloTree binary chunk.

    ``trees`` is a list of arrays of dtype LHALO_HALO_DTYPE.
    """
    n_trees = len(trees)
    per_tree = np.array([t.size for t in trees], dtype="<i4")
    tot = int(per_tree.sum())
    with path.open("wb") as fh:
        fh.write(struct.pack("<ii", n_trees, tot))
        per_tree.tofile(fh)
        for t in trees:
            t.astype(LHALO_HALO_DTYPE).tofile(fh)


@pytest.fixture
def lhalotree_single_tree(tmp_path: Path) -> Path:
    """One LHaloTree tree with 4 halos: central + satellite at two snaps.

    Local-index layout (LHaloTree convention: progenitors come after the
    descendant)::

        idx  snap  desc  fpro  ffof   role
         0    1     -1    2     0    central, snap 1 (final)
         1    1     -1    3     0    satellite of halo 0, snap 1
         2    0      0    -1    2    progenitor of halo 0, snap 0
         3    0      1    -1    2    progenitor of halo 1, snap 0

    Centrals have FirstHaloInFOFgroup == self_local; satellites point
    at the central in their snap.
    """
    halos = np.stack(
        [
            _make_lhalo_halo(
                Descendant=-1,
                FirstProgenitor=2,
                NextProgenitor=-1,
                FirstHaloInFOFgroup=0,
                NextHaloInFOFgroup=1,
                Len=1000,
                Mvir=10.0,
                Pos=[0.0, 0.0, 0.0],
                Vel=[100.0, 0.0, 0.0],
                Spin=[1.0, 2.0, 3.0],
                SnapNum=1,
                SubHalfMass=0.05,
            ),
            _make_lhalo_halo(
                Descendant=-1,
                FirstProgenitor=3,
                NextProgenitor=-1,
                FirstHaloInFOFgroup=0,
                NextHaloInFOFgroup=-1,
                Len=100,
                Mvir=1.0,
                Pos=[1.0, 1.0, 1.0],
                Vel=[110.0, 0.0, 0.0],
                Spin=[0.1, 0.2, 0.3],
                SnapNum=1,
                SubHalfMass=0.005,
            ),
            _make_lhalo_halo(
                Descendant=0,
                FirstProgenitor=-1,
                NextProgenitor=-1,
                FirstHaloInFOFgroup=2,
                NextHaloInFOFgroup=3,
                Len=800,
                Mvir=8.0,
                Pos=[0.0, 0.0, 0.0],
                Vel=[120.0, 0.0, 0.0],
                Spin=[1.0, 2.0, 3.0],
                SnapNum=0,
                SubHalfMass=0.04,
            ),
            _make_lhalo_halo(
                Descendant=1,
                FirstProgenitor=-1,
                NextProgenitor=-1,
                FirstHaloInFOFgroup=2,
                NextHaloInFOFgroup=-1,
                Len=80,
                Mvir=0.8,
                Pos=[1.0, 1.0, 1.0],
                Vel=[130.0, 0.0, 0.0],
                Spin=[0.1, 0.2, 0.3],
                SnapNum=0,
                SubHalfMass=0.004,
            ),
        ]
    )
    path = tmp_path / "trees_000.0"
    _write_lhalotree_file(path, [halos])
    return path


@pytest.fixture
def lhalotree_two_trees(tmp_path: Path) -> Path:
    """One LHaloTree file with two independent trees (2 halos each)."""
    tree_a = np.stack(
        [
            _make_lhalo_halo(Descendant=-1, FirstHaloInFOFgroup=0, SnapNum=1, Mvir=5.0),
            _make_lhalo_halo(Descendant=0, FirstHaloInFOFgroup=1, SnapNum=0, Mvir=4.0),
        ]
    )
    tree_b = np.stack(
        [
            _make_lhalo_halo(Descendant=-1, FirstHaloInFOFgroup=0, SnapNum=1, Mvir=20.0),
            _make_lhalo_halo(Descendant=0, FirstHaloInFOFgroup=1, SnapNum=0, Mvir=18.0),
        ]
    )
    path = tmp_path / "trees_000.0"
    _write_lhalotree_file(path, [tree_a, tree_b])
    return path


@pytest.fixture
def lhalotree_two_chunks(tmp_path: Path) -> tuple[Path, Path]:
    """Two LHaloTree files, each holding one tree."""
    tree_a = np.stack(
        [
            _make_lhalo_halo(Descendant=-1, FirstHaloInFOFgroup=0, SnapNum=1, Mvir=5.0),
            _make_lhalo_halo(Descendant=0, FirstHaloInFOFgroup=1, SnapNum=0, Mvir=4.0),
        ]
    )
    tree_b = np.stack(
        [
            _make_lhalo_halo(Descendant=-1, FirstHaloInFOFgroup=0, SnapNum=1, Mvir=20.0),
        ]
    )
    p0 = tmp_path / "trees_000.0"
    p1 = tmp_path / "trees_000.1"
    _write_lhalotree_file(p0, [tree_a])
    _write_lhalotree_file(p1, [tree_b])
    return p0, p1


# ---------------------------------------------------------------------------
# AHF fixtures
# ---------------------------------------------------------------------------

# 24 AHF .AHF_halos columns matching _AHF_COLUMNS in the reader:
#   0 ID  1 hostHalo  2 numSubStruct  3 Mvir  4 npart
#   5-7 Xc Yc Zc      8-10 VXc VYc VZc
#   11 Rvir           12-19 unused (filled with 0)
#   20 lambda         21-23 Lx Ly Lz
_AHF_FILLER = " ".join(["0"] * 8)  # columns 12..19


def _ahf_halo_row(
    halo_id: int,
    host: int,
    *,
    mvir: float,
    pos: tuple[float, float, float],
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rvir: float = 100.0,
    spin: float = 0.05,
    angmom: tuple[float, float, float] = (1e10, 1e10, 1e10),
    npart: int = 1000,
) -> str:
    return (
        f"{halo_id} {host} 0 {mvir} {npart} "
        f"{pos[0]} {pos[1]} {pos[2]} "
        f"{vel[0]} {vel[1]} {vel[2]} "
        f"{rvir} {_AHF_FILLER} "
        f"{spin} {angmom[0]} {angmom[1]} {angmom[2]}\n"
    )


@pytest.fixture
def ahf_two_independent_forests(tmp_path: Path) -> list[dict[str, object]]:
    """Two snapshots, two physically independent forests.

    Snap 1 (latest): 100 (central) + 101 (sat of 100) + 200 (independent central)
    Snap 0 (earlier): 1100 (prog of 100) + 1101 (prog of 101, sat of 1100)
                      + 1200 (prog of 200, independent central)

    Mtree links: 1100 -> 100, 1101 -> 101, 1200 -> 200.

    Old "all-in-one-forest" behaviour produces 1 Forest with 6 halos;
    union-find correctly yields two forests {100, 101, 1100, 1101} and
    {200, 1200}.
    """
    snap0 = tmp_path / "snap_00.AHF_halos"
    snap0_mtree = tmp_path / "snap_00.AHF_mtree"
    snap1 = tmp_path / "snap_01.AHF_halos"
    snap0.write_text(
        "# AHF halos header\n"
        + _ahf_halo_row(1100, 0, mvir=1e12, pos=(5.0, 5.0, 5.0))
        + _ahf_halo_row(1101, 1100, mvir=1e11, pos=(5.1, 5.1, 5.1))
        + _ahf_halo_row(1200, 0, mvir=8e11, pos=(10.0, 10.0, 10.0))
    )
    snap1.write_text(
        "# AHF halos header\n"
        + _ahf_halo_row(100, 0, mvir=2e12, pos=(5.0, 5.0, 5.0))
        + _ahf_halo_row(101, 100, mvir=2e11, pos=(5.1, 5.1, 5.1))
        + _ahf_halo_row(200, 0, mvir=1.5e12, pos=(10.0, 10.0, 10.0))
    )
    snap0_mtree.write_text(
        "# halo_id n_shared desc_id\n1100 1000 100\n1101 200 101\n1200 800 200\n"
    )
    return [
        {"halos": str(snap0), "mtree": str(snap0_mtree), "a": 0.5},
        {"halos": str(snap1), "mtree": None, "a": 1.0},
    ]


@pytest.fixture
def ahf_single_snapshot(tmp_path: Path) -> list[dict[str, object]]:
    """One snapshot, three independent halos. Each becomes its own forest."""
    snap = tmp_path / "snap_01.AHF_halos"
    snap.write_text(
        "# AHF halos header\n"
        + _ahf_halo_row(100, 0, mvir=2e12, pos=(5.0, 5.0, 5.0))
        + _ahf_halo_row(200, 0, mvir=1.5e12, pos=(10.0, 10.0, 10.0))
        + _ahf_halo_row(300, 0, mvir=1e12, pos=(15.0, 15.0, 15.0))
    )
    return [{"halos": str(snap), "mtree": None, "a": 1.0}]
