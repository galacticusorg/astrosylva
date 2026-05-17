"""Tests for SubLink forest grouping (union-find vs RootDescendantID)."""

from __future__ import annotations

from pathlib import Path

import pytest

from astrosylva.exceptions import ReaderError
from astrosylva.readers import ReaderSource
from astrosylva.readers.sublink import SubLinkReader

SCALE_FACTORS = {0: 0.5, 1: 1.0}


def _source(path: Path) -> ReaderSource:
    return ReaderSource({"tree_file": str(path)})


def _opts(**extra: object) -> dict[str, object]:
    return {"scale_factors": SCALE_FACTORS, **extra}


def _forests_by_id(reader: SubLinkReader) -> dict[int, set[int]]:
    return {f.forest_id: {int(h["nodeIndex"]) for h in f.halos} for f in reader}


def test_union_find_unites_satellites_with_central(sublink_split_roots_file: Path) -> None:
    """Default (union_find): a satellite that never merges still shares a forest
    with its central, because the FOF edge ties them together."""
    reader = SubLinkReader(_source(sublink_split_roots_file), options=_opts())
    forests = _forests_by_id(reader)
    assert forests == {100: {100, 101, 200, 201}}


def test_union_find_is_the_default(sublink_split_roots_file: Path) -> None:
    """Same as above but make the default explicit so future config changes
    that flip the default are caught here."""
    reader = SubLinkReader(
        _source(sublink_split_roots_file), options=_opts(forest_grouping="union_find")
    )
    forests = _forests_by_id(reader)
    assert forests == {100: {100, 101, 200, 201}}


def test_root_descendant_legacy_splits_satellite(sublink_split_roots_file: Path) -> None:
    """Legacy grouping splits the FOF-connected halos along RootDescendantID lines."""
    reader = SubLinkReader(
        _source(sublink_split_roots_file),
        options=_opts(forest_grouping="root_descendant"),
    )
    forests = _forests_by_id(reader)
    assert forests == {100: {100, 200}, 101: {101, 201}}


def test_root_descendant_hosts_clamped_per_forest(sublink_split_roots_file: Path) -> None:
    """Under the legacy grouping, the satellite's central is in another forest,
    so the host pointer must be clamped to self."""
    reader = SubLinkReader(
        _source(sublink_split_roots_file),
        options=_opts(forest_grouping="root_descendant"),
    )
    hosts: dict[int, int] = {}
    for forest in reader:
        for h in forest.halos:
            hosts[int(h["nodeIndex"])] = int(h["hostIndex"])
    # Forest {100, 200}: each is its own central.
    assert hosts[100] == 100
    assert hosts[200] == 200
    # Forest {101, 201}: satellite's host (100, 200) lives in the other forest,
    # so we fall back to self.
    assert hosts[101] == 101
    assert hosts[201] == 201


def test_union_find_hosts_resolve_within_unified_forest(
    sublink_split_roots_file: Path,
) -> None:
    """Under union-find, the satellite and central share a forest so the host
    pointer survives clamping."""
    reader = SubLinkReader(_source(sublink_split_roots_file), options=_opts())
    forest = next(iter(reader))
    hosts = {int(h["nodeIndex"]): int(h["hostIndex"]) for h in forest.halos}
    assert hosts == {100: 100, 101: 100, 200: 200, 201: 200}


def test_forest_grouping_rejects_bad_value(sublink_split_roots_file: Path) -> None:
    with pytest.raises(ReaderError, match="forest_grouping"):
        SubLinkReader(
            _source(sublink_split_roots_file),
            options=_opts(forest_grouping="weird"),
        )


def test_root_descendant_grouping_matches_legacy_count(sublink_tree_file: Path) -> None:
    """For a linear thread the two modes agree (one forest, 3 halos)."""
    scales = {0: 0.05, 1: 0.5, 2: 1.0}
    union = SubLinkReader(_source(sublink_tree_file), options={"scale_factors": scales})
    legacy = SubLinkReader(
        _source(sublink_tree_file),
        options={"scale_factors": scales, "forest_grouping": "root_descendant"},
    )
    assert len(union) == len(legacy) == 1
    assert _forests_by_id(union) == _forests_by_id(legacy) == {100: {100, 200, 300}}
