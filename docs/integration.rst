Integration testing
===================

ytree sample data
-----------------

`ytree <https://ytree.readthedocs.io/>`_ publishes a collection of
real-world merger-tree samples covering every format astrosylva
supports (Consistent-Trees, LHaloTree, SubLink, AHF, and others). The
collection is hosted on the
`yt Hub <https://girder.hub.yt/#collection/59835a1ee2a67400016a2cda>`_
and is too large to bundle with this repository.

Once you've downloaded and unpacked the data, point an environment
variable at the top-level directory:

.. code-block:: bash

   export ASTROSYLVA_YTREE_DATA=/path/to/ytree_data
   pytest tests/test_ytree_samples.py -v

Each test discovers the per-format sub-directory inside that root
(``consistent_trees/``, ``lhalotree/``, ``sublink/``, ``ahf_halos/``)
and skips when the expected files aren't present. With no environment
variable set, the whole file skips silently — CI runs are unaffected.

The smoke tests assert only that each reader loads at least one
forest with at least one halo; they're a way to verify that the
parsers don't choke on real data, not a substitute for the
per-reader unit tests.
