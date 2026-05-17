"""astrosylva: convert halo merger trees to the Galacticus HDF5 input format."""

from importlib.metadata import PackageNotFoundError, version

from astrosylva.exceptions import (
    AstroSylvaError,
    ConfigError,
    MetadataConflictWarning,
    ReaderError,
)
from astrosylva.schema import HALO_DTYPE, Forest, Metadata

try:
    __version__ = version("astrosylva")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "HALO_DTYPE",
    "AstroSylvaError",
    "ConfigError",
    "Forest",
    "Metadata",
    "MetadataConflictWarning",
    "ReaderError",
    "__version__",
]
