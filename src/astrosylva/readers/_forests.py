"""Shared forest-grouping helpers.

Multiple readers need to partition halos into self-contained Galacticus
forests. The canonical definition is the connected components of the
union of two relations:

- descendant edges  (``descendantIndex(i) == nodeIndex(j)``)
- host edges        (``hostIndex(i)        == nodeIndex(j)``)

This module owns the union-find implementation so SubLink, AHF, and any
future reader stay consistent.
"""

from __future__ import annotations

import numpy as np


def clamp_hosts_to_forest(hosts: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    """Remap host pointers that fall outside the current forest's nodes to self.

    Galacticus's "no host" convention is ``hostIndex == nodeIndex``; this
    helper enforces that wherever a satellite's central is in a different
    forest (or simply not in the chunk being loaded).
    """
    in_forest = np.isin(hosts, node_ids)
    out = np.array(hosts, copy=True)
    out[~in_forest] = node_ids[~in_forest]
    return out


def group_by_root_descendant(root_desc: np.ndarray) -> dict[int, np.ndarray]:
    """Legacy grouping: each distinct ``root_desc`` value is its own forest."""
    out: dict[int, list[int]] = {}
    for i, rd in enumerate(root_desc):
        out.setdefault(int(rd), []).append(i)
    return {fid: np.array(idxs, dtype=np.int64) for fid, idxs in sorted(out.items())}


def group_by_union_find(
    node_ids: np.ndarray,
    root_desc: np.ndarray,
    descendants: np.ndarray,
    hosts: np.ndarray,
) -> dict[int, np.ndarray]:
    """Union-find over the union of descendant edges and host edges.

    Returns a mapping ``{forest_id: indices}`` where ``forest_id`` is the
    minimum ``root_desc`` value in each connected component. Callers that
    do not have a separate ``RootDescendantID`` concept can pass
    ``node_ids`` for ``root_desc``.
    """
    n = node_ids.shape[0]
    id_to_idx: dict[int, int] = {int(nid): i for i, nid in enumerate(node_ids)}
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        d_idx = id_to_idx.get(int(descendants[i]))
        if d_idx is not None:
            union(i, d_idx)
        h_idx = id_to_idx.get(int(hosts[i]))
        if h_idx is not None:
            union(i, h_idx)

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    labeled: dict[int, list[int]] = {}
    for indices in components.values():
        forest_id = min(int(root_desc[i]) for i in indices)
        # Components occasionally collide on forest_id (e.g. duplicate
        # root_desc values); merging is the safe option — dropping
        # halos would be silent corruption.
        labeled.setdefault(forest_id, []).extend(indices)
    return {fid: np.array(sorted(idxs), dtype=np.int64) for fid, idxs in sorted(labeled.items())}
