"""Reader abstract base class and entry-point discovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, ClassVar

from astrosylva.exceptions import ReaderError
from astrosylva.schema import Forest, Metadata


@dataclass
class ReaderSource:
    """Free-form mapping of paths/handles that locate a reader's input.

    Each reader documents which keys it expects (e.g. Consistent-Trees needs
    ``input_path``, ``forests_path``, ``locations_path``).
    """

    paths: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.paths.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.paths:
            raise ReaderError(f"Missing required source key: {key!r}")
        return self.paths[key]


class TreeReader(ABC):
    """Abstract base for all tree-format readers.

    Subclasses must:

    - set the class attribute :attr:`name` (and optionally :attr:`aliases`),
    - return reader-introspected metadata from :meth:`metadata`,
    - yield :class:`Forest` objects from ``__iter__``,
    - report the forest count via ``__len__``.
    """

    name: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]] = ()

    def __init__(self, source: ReaderSource, options: dict[str, Any] | None = None) -> None:
        self.source = source
        self.options = options or {}

    @abstractmethod
    def metadata(self) -> Metadata:
        """Return whatever metadata the reader can introspect from its input."""

    def defaults(self) -> Metadata:
        """Reader-supplied default metadata.

        Unlike :meth:`metadata`, these values are *not* introspected from
        the input data — they're per-format conventions that fill in a
        Galacticus output's optional attributes when the user hasn't
        spelled them out in their YAML. Config values silently override
        defaults; introspected values warn on conflict with config.

        Subclasses override this to ship their format's defaults; the
        base implementation returns an empty :class:`Metadata`.
        """
        return Metadata()

    @abstractmethod
    def __iter__(self) -> Iterator[Forest]:
        """Yield :class:`Forest` objects in any order."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of forests this reader will yield."""


def discover_readers() -> dict[str, type[TreeReader]]:
    """Return a mapping of reader name (and aliases) to reader class.

    Discovered via the ``astrosylva.readers`` entry point group.
    """
    registry: dict[str, type[TreeReader]] = {}
    eps = entry_points(group="astrosylva.readers")
    for ep in eps:
        cls = ep.load()
        if not isinstance(cls, type) or not issubclass(cls, TreeReader):
            raise ReaderError(
                f"Entry point {ep.name!r} did not resolve to a TreeReader subclass: {cls!r}"
            )
        registry[cls.name] = cls
        for alias in cls.aliases:
            registry[alias] = cls
    return registry


def get_reader(name: str) -> type[TreeReader]:
    """Look up a reader class by name or alias."""
    registry = discover_readers()
    if name not in registry:
        available = sorted(set(registry))
        raise ReaderError(f"Unknown reader {name!r}. Available: {available}")
    return registry[name]
