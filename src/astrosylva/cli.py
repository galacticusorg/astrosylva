"""``astrosylva`` command-line entry point."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import click

from astrosylva.config import Config, load_config
from astrosylva.exceptions import AstroSylvaError, MetadataConflictWarning
from astrosylva.readers import ReaderSource, discover_readers, get_reader
from astrosylva.schema import Metadata
from astrosylva.writers import GalacticusWriter


@click.group(help="Convert halo merger trees to the Galacticus HDF5 format.")
@click.version_option(package_name="astrosylva")
def main() -> None:
    pass


@main.command(help="Run a conversion described by a YAML config file.")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def convert(config_path: Path) -> None:
    try:
        cfg = load_config(config_path)
    except AstroSylvaError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)

    try:
        _run_conversion(cfg)
    except AstroSylvaError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)


@main.command(help="List discovered reader plugins.")
def readers() -> None:
    registry = discover_readers()
    seen: dict[str, type] = {}
    for cls in registry.values():
        seen.setdefault(cls.name, cls)
    for canonical, cls in sorted(seen.items()):
        aliases = ", ".join(cls.aliases) or "-"
        click.echo(f"{canonical:24s} aliases: {aliases}")


@main.command(help="Parse and validate a config file without writing anything.")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(config_path: Path) -> None:
    try:
        load_config(config_path)
    except AstroSylvaError as exc:
        click.echo(f"invalid: {exc}", err=True)
        sys.exit(2)
    click.echo(f"ok: {config_path}")


def _run_conversion(cfg: Config) -> None:
    reader_cls = get_reader(cfg.reader.name)
    reader = reader_cls(ReaderSource(cfg.reader.source), cfg.reader.options)

    reader_meta = reader.metadata()
    config_meta = Metadata(
        cosmology=dict(cfg.metadata.cosmology),
        units=dict(cfg.metadata.units),
        halo_trees=dict(cfg.metadata.halo_trees),
        group_finder=dict(cfg.metadata.group_finder),
        simulation=dict(cfg.metadata.simulation),
        forest_halos_attrs=dict(cfg.metadata.forest_halos_attrs),
    )
    merged = _merge_metadata(reader_meta, config_meta)

    cfg.writer.output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(reader)
    click.echo(f"converting {total} forests -> {cfg.writer.output_path}")
    with GalacticusWriter(cfg.writer.output_path, merged) as writer:
        for i, forest in enumerate(reader, start=1):
            writer.write_forest(forest)
            if i % 100 == 0 or i == total:
                click.echo(f"  wrote forest {i}/{total}")
    click.echo("done.")


def _merge_metadata(reader_meta: Metadata, config_meta: Metadata) -> Metadata:
    """Merge config and reader metadata. Reader wins on conflict; warn."""
    out = Metadata(format_version=reader_meta.format_version or config_meta.format_version)
    for field_name in (
        "cosmology",
        "units",
        "halo_trees",
        "group_finder",
        "simulation",
        "forest_halos_attrs",
    ):
        config_dict: dict[str, Any] = dict(getattr(config_meta, field_name))
        reader_dict: dict[str, Any] = dict(getattr(reader_meta, field_name))
        merged: dict[str, Any] = dict(config_dict)
        for key, reader_val in reader_dict.items():
            if key in config_dict and config_dict[key] != reader_val:
                warnings.warn(
                    f"{field_name}.{key}: reader value {reader_val!r} "
                    f"overrides config value {config_dict[key]!r}",
                    MetadataConflictWarning,
                    stacklevel=2,
                )
            merged[key] = reader_val
        setattr(out, field_name, merged)
    return out


if __name__ == "__main__":  # pragma: no cover
    main()
