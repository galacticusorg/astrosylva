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
``lhalotree``        ``tree_file`` (single) or ``tree_files`` (list)
``sublink``          ``tree_file`` (single) or ``tree_files`` (list)
``ahf``              ``snapshots`` (list of dicts)
================== ==============================================

LHaloTree
~~~~~~~~~

Reads the 104-byte Millennium / L-Galaxies binary halo struct. Each
*tree* in the LHaloTree file maps to one Galacticus :class:`Forest`;
local pointers (``Descendant``, ``FirstHaloInFOFgroup``) are rewritten
into a global ``nodeIndex`` space (a running counter across trees and
chunks). Scale-factor lookup uses the same options as SubLink
(``snapshot_table`` / ``scale_factors`` / ``redshifts``,
``strict_scale_factors``).

Status: proxies in use. ``scaleRadius`` comes from ``SubHalfMass``
(half-mass radius), ``angularMomentum`` from the ``Spin`` vector
(Millennium's specific-J), and dimensionless ``spin`` is left at 0.0.

SubLink multi-chunk loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A SubLink run is normally sharded into many ``tree_extended.<N>.hdf5``
files. Pass the full list as ``source.tree_files``:

.. code-block:: yaml

   reader:
     name: sublink
     source:
       tree_files:
         - tree_extended.0.hdf5
         - tree_extended.1.hdf5
         - tree_extended.2.hdf5

The reader loads every chunk and runs forest grouping over the union.
Cross-chunk host and descendant pointers resolve as long as both ends
are in the file list. The legacy single-file key ``source.tree_file``
is still accepted.

SubLink scale-factor table
~~~~~~~~~~~~~~~~~~~~~~~~~~

SubLink HDF5 files store ``SnapNum`` but not the per-snapshot scale
factor. Supply one of, in priority order:

1. ``source.snapshot_table``: path to a whitespace-delimited file
   ``snap_num  (scale_factor | redshift)``. Control the second column's
   meaning with ``options.snapshot_table_quantity`` (``"scale_factor"``
   default, or ``"redshift"``).
2. ``options.scale_factors``: inline ``{snap_num: a}`` mapping.
3. ``options.redshifts``: inline ``{snap_num: z}`` mapping (converted
   to ``a = 1/(1+z)``). Mutually exclusive with ``scale_factors``.

By default a missing table or missing snap raises
:class:`ReaderError`. Set ``options.strict_scale_factors: false`` to
downgrade to a warning (missing values become ``NaN``).

SubLink forest grouping
~~~~~~~~~~~~~~~~~~~~~~~

A Galacticus forest must be self-contained: every gravitational
interaction that affects a halo's evolution should be inside the same
forest. ``RootDescendantID`` alone is not enough — a satellite that is
disrupted before merging with its host has a different
``RootDescendantID`` from the host, even though they shared a FOF
group for most of cosmic history.

``options.forest_grouping``:

- ``"union_find"`` (default): connected components of the union of
  descendant edges and host edges. Forest IDs are the smallest
  ``RootDescendantID`` in each component.
- ``"root_descendant"``: legacy behaviour — one forest per
  ``RootDescendantID``. Satellites that never merge end up in their
  own forest, and their host pointers get clamped to self.

SubLink host-pointer resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Galacticus needs each subhalo's ``hostIndex`` to point to its FOF
central. Control how the reader resolves this with
``options.host_resolution``:

- ``"auto"`` (default): use ``/FirstSubhaloInFOFGroupID`` if present,
  else compute from ``/SubhaloGrNr`` + ``/SubfindID``
  (``SubfindID == 0`` marks the central in each
  ``(SnapNum, SubhaloGrNr)`` bucket), else fall back to self-host with
  a warning.
- ``"field"``: require ``/FirstSubhaloInFOFGroupID``; raise otherwise.
- ``"fof_compute"``: require ``/SubhaloGrNr`` + ``/SubfindID``; raise
  otherwise.
- ``"self"``: every subhalo is its own host (silent).

Hosts pointing to subhalos in a different SubLink chunk are silently
remapped to self, matching Galacticus's "no host" convention.
