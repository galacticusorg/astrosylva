astrosylva
==========

*Carry merger trees from the forest to Galacticus.*

A Python library and CLI for converting halo merger-tree catalogues
(Consistent-Trees, LHaloTree, SubLink, AHF) into the Galacticus HDF5
input format.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   schema
   readers
   integration
   api

Installation
------------

.. code-block:: bash

   pip install astrosylva

Quickstart
----------

.. code-block:: bash

   astrosylva convert config.yaml

See :doc:`schema` for the canonical halo / metadata contract and
:doc:`readers` for how to add a reader plugin.
