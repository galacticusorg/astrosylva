Writing a reader plugin
=======================

Readers are discovered via the ``astrosylva.readers`` entry-point group.
Adding one is three small pieces of work: subclass
:class:`astrosylva.readers.TreeReader`, register an entry point, and
yield :class:`astrosylva.Forest` objects.

Skeleton
--------

.. code-block:: python

   from astrosylva.readers import TreeReader
   from astrosylva.schema import HALO_DTYPE, Forest, Metadata

   class MyReader(TreeReader):
       name = "myformat"
       aliases = ("mf",)

       def metadata(self) -> Metadata:
           return Metadata(cosmology={"HubbleParam": 0.7})

       def __len__(self) -> int:
           return self._n_forests

       def __iter__(self):
           for forest_id, halos in self._walk():
               yield Forest(forest_id=forest_id, halos=halos)

Entry point
-----------

In ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."astrosylva.readers"]
   myformat = "mypkg.reader:MyReader"

Reader responsibilities
-----------------------

A reader **must**:

1. Convert all units to the canonical ones documented in :doc:`schema`.
2. Remap "no host" sentinels to ``hostIndex == nodeIndex``.
3. Populate any metadata it can introspect from its input.
4. Validate its required ``source`` keys via
   :meth:`astrosylva.readers.ReaderSource.require`.

Bundled readers
---------------

================== ==============================================
Reader             Required ``source`` keys
================== ==============================================
``consistent_trees`` ``input_path``, ``forests_path``, ``locations_path``
``lhalotree``        ``input_path``
``sublink``          ``tree_file``
``ahf``              ``snapshots`` (list of dicts)
================== ==============================================
