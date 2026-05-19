API reference
=============

This page documents the public Python API. Hand-written guides:

- :doc:`schema` — the canonical halo / forest / metadata contract.
- :doc:`readers` — reader plugins and their format-specific options.
- :doc:`integration` — running against ytree's sample data.

Schema
------

The on-disk and in-memory contract every reader speaks.

.. data:: astrosylva.HALO_DTYPE
   :type: numpy.dtype

   The structured ``numpy`` dtype every reader yields and the writer
   consumes. See :doc:`schema` for the full field list and units.

.. autoclass:: astrosylva.Forest
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: astrosylva.Metadata
   :members:
   :undoc-members:
   :show-inheritance:

Reader framework
----------------

The abstract base class and the entry-point discovery helpers.

.. autoclass:: astrosylva.readers.TreeReader
   :members:
   :show-inheritance:

.. autoclass:: astrosylva.readers.ReaderSource
   :members:
   :show-inheritance:

.. autofunction:: astrosylva.readers.discover_readers

.. autofunction:: astrosylva.readers.get_reader

Bundled readers
---------------

.. autoclass:: astrosylva.readers.consistent_trees.ConsistentTreesReader
   :members:
   :show-inheritance:

.. autoclass:: astrosylva.readers.lhalotree.LHaloTreeReader
   :members:
   :show-inheritance:

.. autoclass:: astrosylva.readers.sublink.SubLinkReader
   :members:
   :show-inheritance:

.. autoclass:: astrosylva.readers.ahf.AHFReader
   :members:
   :show-inheritance:

Writer
------

.. autoclass:: astrosylva.writers.GalacticusWriter
   :members:
   :show-inheritance:

Configuration
-------------

YAML config loading + the pydantic models that validate it.

.. autofunction:: astrosylva.config.load_config

.. autoclass:: astrosylva.config.Config
   :members:
   :show-inheritance:

.. autoclass:: astrosylva.config.ReaderConfig
   :members:
   :show-inheritance:

.. autoclass:: astrosylva.config.WriterConfig
   :members:
   :show-inheritance:

.. autoclass:: astrosylva.config.MetadataConfig
   :members:
   :show-inheritance:

Exceptions
----------

.. autoexception:: astrosylva.AstroSylvaError
   :show-inheritance:

.. autoexception:: astrosylva.ConfigError
   :show-inheritance:

.. autoexception:: astrosylva.ReaderError
   :show-inheritance:

.. autoexception:: astrosylva.exceptions.WriterError
   :show-inheritance:

.. autoexception:: astrosylva.MetadataConflictWarning
   :show-inheritance:
