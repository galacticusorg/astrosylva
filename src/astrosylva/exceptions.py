"""Exception and warning types for astrosylva."""

from __future__ import annotations


class AstroSylvaError(Exception):
    """Base class for all astrosylva errors."""


class ConfigError(AstroSylvaError):
    """Raised when the user-supplied configuration is invalid."""


class ReaderError(AstroSylvaError):
    """Raised when a reader cannot parse its input."""


class WriterError(AstroSylvaError):
    """Raised when the writer cannot produce its output."""


class MetadataConflictWarning(UserWarning):
    """Reader-introspected metadata conflicts with config-supplied metadata.

    The reader value wins; the warning surfaces both values so the user can
    decide whether the config is wrong.
    """
