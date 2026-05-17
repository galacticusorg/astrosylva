"""Canonical in-memory schema for halos, forests, and run metadata.

All readers must return halo arrays with :data:`HALO_DTYPE`. Units are:

- length:   Mpc/h, comoving
- mass:     M_sun/h
- velocity: km/s, peculiar (no Hubble flow)
- spin:     dimensionless (Bullock)
- a:        scale factor (dimensionless)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

HALO_DTYPE = np.dtype(
    [
        ("nodeIndex", "<i8"),
        ("descendantIndex", "<i8"),
        ("hostIndex", "<i8"),
        ("expansionFactor", "<f8"),
        ("nodeMass", "<f8"),
        ("scaleRadius", "<f8"),
        ("position", "<f8", (3,)),
        ("velocity", "<f8", (3,)),
        ("angularMomentum", "<f8", (3,)),
        ("spin", "<f8"),
    ]
)


@dataclass(frozen=True)
class Forest:
    """A single self-contained merger-tree forest."""

    forest_id: int
    halos: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.halos.dtype != HALO_DTYPE:
            raise ValueError(f"Forest.halos must have dtype HALO_DTYPE, got {self.halos.dtype}")
        if self.halos.ndim != 1:
            raise ValueError(f"Forest.halos must be 1-D, got ndim={self.halos.ndim}")

    @property
    def n_halos(self) -> int:
        return int(self.halos.shape[0])


@dataclass
class Metadata:
    """Run metadata grouped to mirror the Galacticus HDF5 layout."""

    cosmology: dict[str, Any] = field(default_factory=dict)
    units: dict[str, Any] = field(default_factory=dict)
    halo_trees: dict[str, Any] = field(default_factory=dict)
    group_finder: dict[str, Any] = field(default_factory=dict)
    simulation: dict[str, Any] = field(default_factory=dict)
    forest_halos_attrs: dict[str, Any] = field(default_factory=dict)
    format_version: int = 2

    def groups(self) -> dict[str, dict[str, Any]]:
        """Return a mapping of HDF5 group path -> attributes dict."""
        return {
            "/cosmology": self.cosmology,
            "/units": self.units,
            "/groupFinder": self.group_finder,
            "/simulation": self.simulation,
            "/forestHalos": self.forest_halos_attrs,
        }


DEFAULT_UNITS: dict[str, Any] = {
    "lengthUnitsInSI": 3.08568e22,
    "lengthHubbleExponent": -1,
    "lengthScaleFactorExponent": 1,
    "massUnitsInSI": 1.9891e30,
    "massHubbleExponent": -1,
    "massScaleFactorExponent": 0,
    "timeUnitsInSI": 3.1536e16,
    "timeHubbleExponent": 0,
    "timeScaleFactorExponent": 0,
    "velocityUnitsInSI": 1000.0,
    "velocityHubbleExponent": 0,
    "velocityScaleFactorExponent": 0,
}
