"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from astrosylva.exceptions import ConfigError


class ReaderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class WriterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: Path
    options: dict[str, Any] = Field(default_factory=dict)


class MetadataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cosmology: dict[str, Any] = Field(default_factory=dict)
    units: dict[str, Any] = Field(default_factory=dict)
    halo_trees: dict[str, Any] = Field(default_factory=dict, alias="haloTrees")
    group_finder: dict[str, Any] = Field(default_factory=dict, alias="groupFinder")
    simulation: dict[str, Any] = Field(default_factory=dict)
    forest_halos_attrs: dict[str, Any] = Field(default_factory=dict, alias="forestHalosAttrs")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reader: ReaderConfig
    writer: WriterConfig
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")
    try:
        return Config.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"Invalid config {path}: {exc}") from exc
