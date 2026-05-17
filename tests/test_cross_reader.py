"""Cross-reader equivalence test.

If two readers consume semantically equivalent input data, they should
produce equivalent :class:`Forest` objects on the canonical fields. This
test catches semantic drift across readers — e.g. a future unit-conversion
change to one reader that doesn't land in the others.

The comparison ignores fields that readers fill from format-specific
quantities: ``scaleRadius`` (CT-only), ``halfMassRadius`` (LHaloTree/SubLink
only), and ``angularMomentum`` / ``spin`` (Bullock convention differs).
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest

from astrosylva.readers import ReaderSource
from astrosylva.readers.ahf import AHFReader
from astrosylva.readers.base import TreeReader
from astrosylva.readers.consistent_trees import ConsistentTreesReader
from astrosylva.readers.lhalotree import LHALO_HALO_DTYPE, LHaloTreeReader
from astrosylva.readers.sublink import SubLinkReader
from astrosylva.schema import HALO_DTYPE, Forest

# ----------------------------------------------------- shared ground truth

# A 3-halo linear thread: a single central halo with two progenitors.
# Physical units match the canonical schema (Mpc/h, km/s, Msun/h, a).
_GROUND_TRUTH: list[dict[str, object]] = [
    # snap 0 (final): root
    {
        "snap": 0,
        "a": 1.0,
        "mass": 2.0e12,
        "pos": (5.0, 5.0, 5.0),
        "vel": (100.0, 100.0, 100.0),
    },
    # snap 1: progenitor of snap-0
    {
        "snap": 1,
        "a": 0.5,
        "mass": 1.0e12,
        "pos": (5.1, 5.1, 5.1),
        "vel": (110.0, 110.0, 110.0),
    },
    # snap 2: progenitor of snap-1
    {
        "snap": 2,
        "a": 0.25,
        "mass": 5.0e11,
        "pos": (5.2, 5.2, 5.2),
        "vel": (120.0, 120.0, 120.0),
    },
]


# ----------------------------------------------------- fixture builders


def _write_ct_fixture(tmp_path: Path) -> Path:
    """Write the ground-truth thread as a Consistent-Trees dataset."""
    out = tmp_path / "ct"
    out.mkdir()
    tree_path = out / "tree_0_0_0.dat"
    contents = (
        "#scale(0) id(1) desc_scale(2) desc_id(3) num_prog(4) pid(5) upid(6) "
        "desc_pid(7) phantom(8) sam_mvir(9) mvir(10) rvir(11) rs(12) vrms(13) "
        "mmp(14) scale_of_last_MM(15) vmax(16) x(17) y(18) z(19) vx(20) vy(21) "
        "vz(22) Jx(23) Jy(24) Jz(25) Spin(26)\n"
        "#Omega_M = 0.3; Omega_L = 0.7; h0 = 0.7\n"
        "#Full box size = 100.000000 Mpc/h\n"
        "#Tree 0 0 0\n"
        "1\n"
    )
    tree_offset = len(contents.encode()) + len(b"#tree 300\n")
    # CT IDs 300/200/100 for snap 0/1/2 respectively (newest -> oldest).
    # Descendant link: 200 -> 300 (snap 1 to snap 0), 100 -> 200 (snap 2 to 1).
    contents += (
        "#tree 300\n"
        # scale id desc_scale desc_id num_prog pid upid desc_pid phantom
        # sam_mvir mvir rvir rs vrms mmp scale_of_last_MM vmax x y z
        # vx vy vz Jx Jy Jz Spin
        "1.0 300 -1 -1 0 -1 -1 -1 0 2e12 2e12 200 20 100 1 1.0 250 "
        "5.0 5.0 5.0 100 100 100 0 0 0 0.05\n"
        "0.5 200 1.0 300 0 -1 -1 -1 0 1e12 1e12 150 15 80 1 0.9 200 "
        "5.1 5.1 5.1 110 110 110 0 0 0 0.05\n"
        "0.25 100 0.5 200 0 -1 -1 -1 0 5e11 5e11 100 10 60 1 0.8 150 "
        "5.2 5.2 5.2 120 120 120 0 0 0 0.05\n"
    )
    tree_path.write_bytes(contents.encode())
    (out / "locations.dat").write_text(
        f"TreeRootID FileID Offset Filename\n300 0 {tree_offset} tree_0_0_0.dat\n"
    )
    (out / "forests.list").write_text("TreeRootID ForestID Weight\n300 300 1.0\n")
    return out


def _write_lhalotree_fixture(tmp_path: Path) -> Path:
    """Write the ground-truth thread as an LHaloTree binary chunk.

    LHaloTree convention: progenitors come after the descendant in the
    array. Local indices 0 (snap 0 root), 1 (snap 1 prog), 2 (snap 2 prog).
    """
    halos = np.zeros(3, dtype=LHALO_HALO_DTYPE)
    for k, gt in enumerate(_GROUND_TRUTH):
        halos[k]["SnapNum"] = gt["snap"]
        halos[k]["FirstHaloInFOFgroup"] = k  # central of its own FOF
        halos[k]["Mvir"] = gt["mass"] / 1e10  # LHaloTree stores 10^10 Msun/h
        halos[k]["Pos"] = gt["pos"]
        halos[k]["Vel"] = gt["vel"]
    # Descendant / progenitor linkage matching the thread.
    halos[0]["Descendant"] = -1
    halos[0]["FirstProgenitor"] = 1
    halos[1]["Descendant"] = 0
    halos[1]["FirstProgenitor"] = 2
    halos[2]["Descendant"] = 1
    halos[2]["FirstProgenitor"] = -1
    halos["NextProgenitor"] = -1
    halos["NextHaloInFOFgroup"] = -1

    path = tmp_path / "trees_000.0"
    with path.open("wb") as fh:
        fh.write(struct.pack("<ii", 1, halos.size))
        np.array([halos.size], dtype="<i4").tofile(fh)
        halos.tofile(fh)
    return path


def _write_sublink_fixture(tmp_path: Path) -> Path:
    """Write the ground-truth thread as a single SubLink-style HDF5 chunk."""
    path = tmp_path / "sublink_tree_extended.0.hdf5"
    n = len(_GROUND_TRUTH)
    # SubhaloIDs: 300 (snap 0), 200 (snap 1), 100 (snap 2). All share the
    # final descendant 300 as RootDescendantID.
    ids = np.array([300, 200, 100], dtype=np.int64)
    desc = np.array([-1, 300, 200], dtype=np.int64)
    first_prog = np.array([200, 100, -1], dtype=np.int64)
    with h5py.File(path, "w") as f:
        f.create_dataset("SubhaloID", data=ids)
        f.create_dataset("DescendantID", data=desc)
        f.create_dataset("FirstProgenitorID", data=first_prog)
        f.create_dataset("NextProgenitorID", data=np.full(n, -1, dtype=np.int64))
        f.create_dataset("RootDescendantID", data=np.full(n, 300, dtype=np.int64))
        f.create_dataset("TreeID", data=np.full(n, 1, dtype=np.int64))
        f.create_dataset(
            "SnapNum", data=np.array([gt["snap"] for gt in _GROUND_TRUTH], dtype=np.int32)
        )
        # Each subhalo is its own FOF central.
        f.create_dataset("FirstSubhaloInFOFGroupID", data=ids)
        # SubhaloMass is stored in 10^10 M_sun/h, reader scales by 1e10.
        f.create_dataset(
            "SubhaloMass",
            data=np.array([gt["mass"] / 1e10 for gt in _GROUND_TRUTH], dtype=np.float32),
        )
        # Half-mass radius (dummy; not compared cross-reader).
        f.create_dataset("SubhaloHalfmassRad", data=np.ones(n, dtype=np.float32))
        # SubhaloPos in kpc/h (reader / 1000 -> Mpc/h).
        f.create_dataset(
            "SubhaloPos",
            data=np.array(
                [[c * 1000.0 for c in gt["pos"]] for gt in _GROUND_TRUTH],
                dtype=np.float32,
            ),
        )
        f.create_dataset(
            "SubhaloVel",
            data=np.array([gt["vel"] for gt in _GROUND_TRUTH], dtype=np.float32),
        )
        f.create_dataset("SubhaloSpin", data=np.zeros((n, 3), dtype=np.float32))
    return path


# AHF helper — same 24-column layout as conftest's _ahf_halo_row, inlined
# here so the cross-reader test file is self-contained.
_AHF_FILLER = " ".join(["0"] * 8)  # columns 12..19


def _ahf_row(halo_id: int, host: int, *, mvir: float, pos: tuple[float, float, float]) -> str:
    return (
        f"{halo_id} {host} 0 {mvir} 1000 "
        f"{pos[0]} {pos[1]} {pos[2]} "
        f"100 100 100 "  # VXc, VYc, VZc — overwritten below per snapshot
        f"100 {_AHF_FILLER} 0.05 1e10 1e10 1e10\n"
    )


def _write_ahf_fixture(tmp_path: Path) -> list[dict[str, object]]:
    """Write the ground-truth thread as an AHF snapshot sequence."""
    # AHF halo IDs: 300 / 200 / 100 (newest first, matches CT for clarity).
    # Snap 2 (oldest): halo 100; mtree links 100 -> 200.
    # Snap 1:          halo 200; mtree links 200 -> 300.
    # Snap 0 (latest): halo 300; no mtree.
    snap_to_id = {0: 300, 1: 200, 2: 100}

    snapshots: list[dict[str, object]] = []
    # Iterate oldest -> latest (AHF reader expects this ordering).
    for snap_index, gt in enumerate(
        sorted(_GROUND_TRUTH, key=lambda g: int(g["snap"]), reverse=True)
    ):
        snap = int(gt["snap"])
        halo_id = snap_to_id[snap]
        halos_path = tmp_path / f"ahf_snap{snap:02d}.AHF_halos"
        # Re-emit the row but overwrite the velocity columns with the
        # ground-truth value for this halo.
        base = _ahf_row(halo_id, 0, mvir=float(gt["mass"]), pos=gt["pos"])
        tokens = base.split()
        # _ahf_row writes VXc/VYc/VZc at positions 8/9/10.
        tokens[8], tokens[9], tokens[10] = (str(v) for v in gt["vel"])
        halos_path.write_text("# header\n" + " ".join(tokens) + "\n")

        mtree_path: str | None = None
        # All snaps except the latest get an .AHF_mtree_idx with a single
        # ProgID DescID pair.
        if snap > 0:
            desc_halo = snap_to_id[snap - 1]
            mtree_p = tmp_path / f"ahf_snap{snap:02d}.AHF_mtree_idx"
            mtree_p.write_text(f"{halo_id} {desc_halo}\n")
            mtree_path = str(mtree_p)
        snapshots.append(
            {
                "halos": str(halos_path),
                "mtree": mtree_path,
                "a": float(gt["a"]),
            }
        )
        del snap_index
    return snapshots


# ----------------------------------------------------- canonicalisation


def _canonicalize(forest: Forest) -> np.ndarray:
    """Re-label ``nodeIndex`` so two readers' Forests can be compared.

    Halos are sorted by (descending expansionFactor, descending mass,
    ascending x) — stable for a non-degenerate thread — then assigned
    nodeIndex 0..N-1 in that order. Descendant and host pointers are
    remapped through the same translation.
    """
    halos = forest.halos
    n = halos.shape[0]
    # np.lexsort treats the LAST key as primary.
    order = np.lexsort(
        (
            halos["position"][:, 0],
            -halos["nodeMass"],
            -halos["expansionFactor"],
        )
    )
    new_ids = np.empty(n, dtype=np.int64)
    for new_id, old_idx in enumerate(order):
        new_ids[old_idx] = new_id
    old_to_new = {int(halos["nodeIndex"][i]): int(new_ids[i]) for i in range(n)}

    canonical = halos.copy()
    canonical["nodeIndex"] = new_ids
    for k in range(n):
        host = int(halos["hostIndex"][k])
        canonical["hostIndex"][k] = old_to_new.get(host, -1)
        desc = int(halos["descendantIndex"][k])
        canonical["descendantIndex"][k] = -1 if desc == -1 else old_to_new.get(desc, -1)
    # Sort the array by canonical nodeIndex so per-row comparisons line up.
    return canonical[np.argsort(canonical["nodeIndex"])]


# ----------------------------------------------------- fixtures


@pytest.fixture
def equivalence_inputs(tmp_path: Path) -> tuple[Path, Path, dict[int, float]]:
    ct_dir = _write_ct_fixture(tmp_path)
    lhalo_path = _write_lhalotree_fixture(tmp_path)
    scale_factors = {gt["snap"]: gt["a"] for gt in _GROUND_TRUTH}
    return ct_dir, lhalo_path, scale_factors


def _build_ct_reader(tmp_path: Path) -> TreeReader:
    ct_dir = _write_ct_fixture(tmp_path)
    return ConsistentTreesReader(
        ReaderSource(
            {
                "input_path": str(ct_dir),
                "forests_path": str(ct_dir / "forests.list"),
                "locations_path": str(ct_dir / "locations.dat"),
            }
        )
    )


def _build_lhalo_reader(tmp_path: Path) -> TreeReader:
    lhalo_path = _write_lhalotree_fixture(tmp_path)
    scale_factors = {gt["snap"]: gt["a"] for gt in _GROUND_TRUTH}
    return LHaloTreeReader(
        ReaderSource({"tree_file": str(lhalo_path)}),
        options={"scale_factors": scale_factors},
    )


def _build_sublink_reader(tmp_path: Path) -> TreeReader:
    sublink_path = _write_sublink_fixture(tmp_path)
    scale_factors = {gt["snap"]: gt["a"] for gt in _GROUND_TRUTH}
    return SubLinkReader(
        ReaderSource({"tree_file": str(sublink_path)}),
        options={"scale_factors": scale_factors},
    )


def _build_ahf_reader(tmp_path: Path) -> TreeReader:
    snapshots = _write_ahf_fixture(tmp_path)
    return AHFReader(ReaderSource({"snapshots": snapshots}))


_ALL_READER_BUILDERS: dict[str, Callable[[Path], TreeReader]] = {
    "consistent_trees": _build_ct_reader,
    "lhalotree": _build_lhalo_reader,
    "sublink": _build_sublink_reader,
    "ahf": _build_ahf_reader,
}


# ----------------------------------------------------- the test


def test_ct_and_lhalotree_produce_equivalent_forests(
    equivalence_inputs: tuple[Path, Path, dict[int, float]],
) -> None:
    """Reading the same merger thread via CT and LHaloTree readers yields
    Forests that agree on canonical schema fields."""
    ct_dir, lhalo_path, scale_factors = equivalence_inputs

    ct_reader = ConsistentTreesReader(
        ReaderSource(
            {
                "input_path": str(ct_dir),
                "forests_path": str(ct_dir / "forests.list"),
                "locations_path": str(ct_dir / "locations.dat"),
            }
        )
    )
    lhalo_reader = LHaloTreeReader(
        ReaderSource({"tree_file": str(lhalo_path)}),
        options={"scale_factors": scale_factors},
    )
    ct_forests = list(ct_reader)
    lhalo_forests = list(lhalo_reader)
    assert len(ct_forests) == 1
    assert len(lhalo_forests) == 1
    assert ct_forests[0].n_halos == lhalo_forests[0].n_halos == len(_GROUND_TRUTH)

    ct = _canonicalize(ct_forests[0])
    lh = _canonicalize(lhalo_forests[0])

    # Topology / identity: exact integer equality.
    for field in ("nodeIndex", "descendantIndex", "hostIndex"):
        np.testing.assert_array_equal(ct[field], lh[field], err_msg=f"mismatch on field {field!r}")

    # Physical scalars: float comparisons.
    for field in ("expansionFactor", "nodeMass"):
        np.testing.assert_allclose(ct[field], lh[field], err_msg=f"mismatch on field {field!r}")

    # Physical vectors: float comparisons (LHaloTree stores Pos/Vel in
    # float32, so allow a touch of tolerance).
    for field in ("position", "velocity"):
        np.testing.assert_allclose(
            ct[field], lh[field], atol=1e-4, err_msg=f"mismatch on field {field!r}"
        )


def test_ct_and_lhalotree_match_the_ground_truth(
    equivalence_inputs: tuple[Path, Path, dict[int, float]],
) -> None:
    """Beyond cross-reader agreement, both readers should match the
    ground-truth values directly — guards against readers agreeing on the
    wrong answer."""
    ct_dir, lhalo_path, scale_factors = equivalence_inputs

    expected = np.empty(len(_GROUND_TRUTH), dtype=HALO_DTYPE)
    # Ground truth sorted by descending a (matches _canonicalize order).
    sorted_gt = sorted(_GROUND_TRUTH, key=lambda g: (-g["a"], -g["mass"]))
    for k, gt in enumerate(sorted_gt):
        expected["nodeIndex"][k] = k
        expected["descendantIndex"][k] = k - 1 if k > 0 else -1
        expected["hostIndex"][k] = k  # central of its own FOF
        expected["expansionFactor"][k] = gt["a"]
        expected["nodeMass"][k] = gt["mass"]
        expected["position"][k] = gt["pos"]
        expected["velocity"][k] = gt["vel"]

    for reader_name, reader in (
        (
            "ConsistentTrees",
            ConsistentTreesReader(
                ReaderSource(
                    {
                        "input_path": str(ct_dir),
                        "forests_path": str(ct_dir / "forests.list"),
                        "locations_path": str(ct_dir / "locations.dat"),
                    }
                )
            ),
        ),
        (
            "LHaloTree",
            LHaloTreeReader(
                ReaderSource({"tree_file": str(lhalo_path)}),
                options={"scale_factors": scale_factors},
            ),
        ),
    ):
        forest = next(iter(reader))
        canon = _canonicalize(forest)
        np.testing.assert_array_equal(
            canon["nodeIndex"], expected["nodeIndex"], err_msg=reader_name
        )
        np.testing.assert_array_equal(
            canon["descendantIndex"], expected["descendantIndex"], err_msg=reader_name
        )
        np.testing.assert_array_equal(
            canon["hostIndex"], expected["hostIndex"], err_msg=reader_name
        )
        np.testing.assert_allclose(
            canon["expansionFactor"], expected["expansionFactor"], err_msg=reader_name
        )
        np.testing.assert_allclose(canon["nodeMass"], expected["nodeMass"], err_msg=reader_name)
        np.testing.assert_allclose(
            canon["position"], expected["position"], atol=1e-4, err_msg=reader_name
        )
        np.testing.assert_allclose(
            canon["velocity"], expected["velocity"], atol=1e-4, err_msg=reader_name
        )


# ---------------------------------------------------- whole-suite equivalence


@pytest.mark.parametrize("reader_name", sorted(_ALL_READER_BUILDERS))
def test_every_reader_matches_ground_truth(reader_name: str, tmp_path: Path) -> None:
    """All four readers, given semantically equivalent input, produce the
    same Forest on the canonical schema fields."""
    expected = np.empty(len(_GROUND_TRUTH), dtype=HALO_DTYPE)
    sorted_gt = sorted(_GROUND_TRUTH, key=lambda g: (-g["a"], -g["mass"]))
    for k, gt in enumerate(sorted_gt):
        expected["nodeIndex"][k] = k
        expected["descendantIndex"][k] = k - 1 if k > 0 else -1
        expected["hostIndex"][k] = k  # central of its own FOF
        expected["expansionFactor"][k] = gt["a"]
        expected["nodeMass"][k] = gt["mass"]
        expected["position"][k] = gt["pos"]
        expected["velocity"][k] = gt["vel"]

    builder = _ALL_READER_BUILDERS[reader_name]
    reader = builder(tmp_path)
    forests = list(reader)
    assert len(forests) == 1, f"{reader_name} produced {len(forests)} forests, expected 1"
    canon = _canonicalize(forests[0])

    np.testing.assert_array_equal(canon["nodeIndex"], expected["nodeIndex"])
    np.testing.assert_array_equal(canon["descendantIndex"], expected["descendantIndex"])
    np.testing.assert_array_equal(canon["hostIndex"], expected["hostIndex"])
    np.testing.assert_allclose(canon["expansionFactor"], expected["expansionFactor"])
    np.testing.assert_allclose(canon["nodeMass"], expected["nodeMass"])
    np.testing.assert_allclose(canon["position"], expected["position"], atol=1e-4)
    np.testing.assert_allclose(canon["velocity"], expected["velocity"], atol=1e-4)


def test_all_four_readers_agree_pairwise(tmp_path: Path) -> None:
    """Every pair of readers produces Forests that match on canonical fields.

    This is the explicit cross-reader contract: any drift introduced to
    one reader without matching changes to the others will trip here.
    """
    canon_by_reader: dict[str, np.ndarray] = {}
    for name, builder in _ALL_READER_BUILDERS.items():
        # Each reader needs its own tmp subdirectory so fixture file
        # names don't collide.
        reader_tmp = tmp_path / name
        reader_tmp.mkdir()
        reader = builder(reader_tmp)
        forests = list(reader)
        assert len(forests) == 1, name
        canon_by_reader[name] = _canonicalize(forests[0])

    names = sorted(canon_by_reader)
    reference = canon_by_reader[names[0]]
    for other in names[1:]:
        rhs = canon_by_reader[other]
        for field in ("nodeIndex", "descendantIndex", "hostIndex"):
            np.testing.assert_array_equal(
                reference[field],
                rhs[field],
                err_msg=f"{names[0]} vs {other}: field {field!r}",
            )
        for field in ("expansionFactor", "nodeMass"):
            np.testing.assert_allclose(
                reference[field],
                rhs[field],
                err_msg=f"{names[0]} vs {other}: field {field!r}",
            )
        for field in ("position", "velocity"):
            np.testing.assert_allclose(
                reference[field],
                rhs[field],
                atol=1e-4,
                err_msg=f"{names[0]} vs {other}: field {field!r}",
            )
