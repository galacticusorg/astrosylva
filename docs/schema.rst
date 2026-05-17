Canonical schema
================

All readers convert their native fields into a single in-memory schema
before the writer emits HDF5. Sticking to this contract is what makes
new readers cheap to add.

Units
-----

============== ==============================================
Quantity       Units
============== ==============================================
length         Mpc/h (comoving)
mass           M_sun/h
velocity       km/s (peculiar; no Hubble flow)
spin           dimensionless (Bullock)
expansion ``a`` dimensionless scale factor
============== ==============================================

``Halo`` dtype
--------------

.. code-block:: python

   HALO_DTYPE = np.dtype([
       ("nodeIndex",        "<i8"),
       ("descendantIndex",  "<i8"),
       ("hostIndex",        "<i8"),
       ("expansionFactor",  "<f8"),
       ("nodeMass",         "<f8"),
       ("scaleRadius",      "<f8"),
       ("position",         "<f8", (3,)),
       ("velocity",         "<f8", (3,)),
       ("angularMomentum",  "<f8", (3,)),
       ("spin",             "<f8"),
   ])

``Forest``
----------

A :class:`astrosylva.Forest` packages one self-contained merger tree:

.. code-block:: python

   @dataclass(frozen=True)
   class Forest:
       forest_id: int
       halos: np.ndarray   # dtype = HALO_DTYPE
       weight: float = 1.0

Galacticus conventions
----------------------

- Halos with no host have ``hostIndex == nodeIndex`` (never ``-1``).
- Halos with no descendant have ``descendantIndex == -1``.
- Per-forest indices in ``/forestIndex`` are written in the order
  forests are yielded by the reader.

Metadata
--------

The :class:`astrosylva.Metadata` dataclass mirrors the Galacticus HDF5
group layout. Reader-introspected values win over config-supplied
values; conflicts emit a :class:`astrosylva.MetadataConflictWarning`.
