"""Tests for SubLink multi-chunk loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from astrosylva.exceptions import ReaderError
from astrosylva.readers import ReaderSource
from astrosylva.readers.sublink import SubLinkReader

SCALE_FACTORS = {0: 0.5, 1: 1.0}


def _opts(**extra: object) -> dict[str, object]:
    return {"scale_factors": SCALE_FACTORS, **extra}


def _forests_by_id(reader: SubLinkReader) -> dict[int, set[int]]:
    return {f.forest_id: {int(h["nodeIndex"]) for h in f.halos} for f in reader}


def _hosts_by_id(reader: SubLinkReader) -> dict[int, int]:
    out: dict[int, int] = {}
    for forest in reader:
        for h in forest.halos:
            out[int(h["nodeIndex"])] = int(h["hostIndex"])
    return out


def test_two_chunks_union_into_one_forest(
    sublink_two_chunks: tuple[Path, Path],
) -> None:
    """Under union_find, two chunks worth of satellites + centrals collapse
    into a single forest because the cross-chunk host edges close the
    connected component."""
    p0, p1 = sublink_two_chunks
    reader = SubLinkReader(ReaderSource({"tree_files": [str(p0), str(p1)]}), options=_opts())
    forests = _forests_by_id(reader)
    assert forests == {100: {100, 101, 200, 201}}


def test_two_chunks_hosts_resolve_across_chunks(
    sublink_two_chunks: tuple[Path, Path],
) -> None:
    """A satellite in chunk 1 whose host lives in chunk 0 now resolves."""
    p0, p1 = sublink_two_chunks
    reader = SubLinkReader(ReaderSource({"tree_files": [str(p0), str(p1)]}), options=_opts())
    assert _hosts_by_id(reader) == {100: 100, 101: 100, 200: 200, 201: 200}


def test_two_chunks_root_descendant_keeps_them_separate(
    sublink_two_chunks: tuple[Path, Path],
) -> None:
    """Legacy grouping still splits along RootDescendantID lines even when
    every chunk is loaded — the host edge is ignored for grouping."""
    p0, p1 = sublink_two_chunks
    reader = SubLinkReader(
        ReaderSource({"tree_files": [str(p0), str(p1)]}),
        options=_opts(forest_grouping="root_descendant"),
    )
    forests = _forests_by_id(reader)
    assert forests == {100: {100, 200}, 101: {101, 201}}


def test_two_chunks_root_descendant_clamps_satellite_hosts(
    sublink_two_chunks: tuple[Path, Path],
) -> None:
    """Satellite hosts get clamped to self under root_descendant: their real
    host (100/200) is in a different forest's slice."""
    p0, p1 = sublink_two_chunks
    reader = SubLinkReader(
        ReaderSource({"tree_files": [str(p0), str(p1)]}),
        options=_opts(forest_grouping="root_descendant"),
    )
    hosts = _hosts_by_id(reader)
    assert hosts[100] == 100
    assert hosts[200] == 200
    assert hosts[101] == 101  # clamped (host=100 lives in forest 100)
    assert hosts[201] == 201  # clamped (host=200 lives in forest 100)


def test_single_path_via_tree_files(sublink_two_chunks: tuple[Path, Path]) -> None:
    """`tree_files` accepts a single path; only one chunk is loaded so the
    cross-chunk host pointers fall back to self."""
    p0, _ = sublink_two_chunks
    reader = SubLinkReader(ReaderSource({"tree_files": [str(p0)]}), options=_opts())
    forests = _forests_by_id(reader)
    # Only chunk 0's two subhalos appear.
    assert forests == {100: {100, 200}}


def test_tree_file_backcompat(sublink_two_chunks: tuple[Path, Path]) -> None:
    """The legacy `tree_file` source key still works for single-chunk runs."""
    p0, _ = sublink_two_chunks
    reader = SubLinkReader(ReaderSource({"tree_file": str(p0)}), options=_opts())
    assert _forests_by_id(reader) == {100: {100, 200}}


def test_neither_source_key_raises(tmp_path: Path) -> None:
    reader = SubLinkReader(ReaderSource({}), options=_opts())
    with pytest.raises(ReaderError, match="tree_file"):
        list(reader)


def test_empty_tree_files_list_raises(tmp_path: Path) -> None:
    reader = SubLinkReader(ReaderSource({"tree_files": []}), options=_opts())
    with pytest.raises(ReaderError, match="empty chunk list"):
        list(reader)


def test_tree_files_wrong_type_raises(sublink_two_chunks: tuple[Path, Path]) -> None:
    reader = SubLinkReader(ReaderSource({"tree_files": 42}), options=_opts())
    with pytest.raises(ReaderError, match="tree_files"):
        list(reader)
