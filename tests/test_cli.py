"""End-to-end CLI tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import yaml
from click.testing import CliRunner

from astrosylva.cli import main


def test_readers_subcommand_lists_consistent_trees() -> None:
    result = CliRunner().invoke(main, ["readers"])
    assert result.exit_code == 0
    assert "consistent_trees" in result.output


def test_validate_subcommand(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "reader": {"name": "consistent_trees", "source": {}, "options": {}},
                "writer": {"output_path": str(tmp_path / "out.hdf5")},
            }
        )
    )
    result = CliRunner().invoke(main, ["validate", str(cfg_path)])
    assert result.exit_code == 0
    assert "ok" in result.output


def test_convert_end_to_end(tmp_path: Path, ct_data_dir: Path) -> None:
    out_path = tmp_path / "galacticus.hdf5"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "reader": {
                    "name": "consistent_trees",
                    "source": {
                        "input_path": str(ct_data_dir),
                        "forests_path": str(ct_data_dir / "forests.list"),
                        "locations_path": str(ct_data_dir / "locations.dat"),
                    },
                    "options": {},
                },
                "writer": {"output_path": str(out_path)},
                "metadata": {
                    "simulation": {"boxSize": 100000.0},
                    "groupFinder": {"groupFinderCode": "rockstar"},
                    "haloTrees": {
                        "haloMassesIncludeSubhalos": 1,
                        "forestsAreSelfContained": 1,
                        "treesHaveSubhalos": 1,
                        "velocitiesIncludeHubbleFlow": 0,
                    },
                },
            }
        )
    )
    result = CliRunner().invoke(main, ["convert", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    with h5py.File(out_path, "r") as f:
        assert f["forestHalos/nodeIndex"].shape == (4,)
        # boxSize from config persists.
        assert f["simulation"].attrs["boxSize"] == 100000.0
        # Cosmology came from the reader.
        assert f["cosmology"].attrs["Omega0"] == 0.27
        # haloTrees attrs from the YAML land on the /forestHalos group
        # (matching the legacy C tool's layout).
        forest_halos_attrs = dict(f["forestHalos"].attrs)
        assert forest_halos_attrs["haloMassesIncludeSubhalos"] == 1
        assert forest_halos_attrs["forestsAreSelfContained"] == 1
        assert forest_halos_attrs["treesHaveSubhalos"] == 1
        assert forest_halos_attrs["velocitiesIncludeHubbleFlow"] == 0
