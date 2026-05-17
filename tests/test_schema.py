"""Schema sanity tests."""

from __future__ import annotations

import numpy as np
import pytest

from astrosylva.schema import HALO_DTYPE, Forest, Metadata


def test_halo_dtype_field_names() -> None:
    assert set(HALO_DTYPE.names) == {
        "nodeIndex",
        "descendantIndex",
        "hostIndex",
        "expansionFactor",
        "nodeMass",
        "scaleRadius",
        "halfMassRadius",
        "position",
        "velocity",
        "angularMomentum",
        "spin",
    }


def test_forest_rejects_wrong_dtype() -> None:
    bad = np.zeros(3, dtype=np.float64)
    with pytest.raises(ValueError, match="HALO_DTYPE"):
        Forest(forest_id=0, halos=bad)


def test_forest_rejects_2d() -> None:
    bad = np.zeros((2, 3), dtype=HALO_DTYPE)
    with pytest.raises(ValueError, match="1-D"):
        Forest(forest_id=0, halos=bad)


def test_metadata_groups_layout() -> None:
    m = Metadata(cosmology={"HubbleParam": 0.7})
    groups = m.groups()
    assert "/cosmology" in groups
    assert groups["/cosmology"]["HubbleParam"] == 0.7
    assert set(groups) == {
        "/cosmology",
        "/units",
        "/groupFinder",
        "/simulation",
        "/forestHalos",
    }


def test_metadata_halo_trees_attrs_emit_to_forestHalos_group() -> None:
    """``haloTrees`` config attributes end up on the ``/forestHalos``
    group, matching the legacy C tool's HDF5 layout."""
    m = Metadata(
        halo_trees={
            "haloMassesIncludeSubhalos": 1,
            "forestsAreSelfContained": 1,
            "treesHaveSubhalos": 1,
            "velocitiesIncludeHubbleFlow": 0,
        }
    )
    forest_halos_attrs = m.groups()["/forestHalos"]
    assert forest_halos_attrs["haloMassesIncludeSubhalos"] == 1
    assert forest_halos_attrs["forestsAreSelfContained"] == 1
    assert forest_halos_attrs["treesHaveSubhalos"] == 1
    assert forest_halos_attrs["velocitiesIncludeHubbleFlow"] == 0
