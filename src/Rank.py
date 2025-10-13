from __future__ import annotations
from reactive_graph import ReactiveGraph, GraphMatplotlibView

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np
import networkx as nx

try:
    # Optional: only imported for the demo
    from reactive_graph import ReactiveGraph, GraphMatplotlibView  # type: ignore
except Exception:
    ReactiveGraph = object  # type: ignore


@dataclass
class RankResult:
    scores: Dict[Hashable, float]
    order: List[Hashable]
    trajectory: List[Dict[Hashable, float]]  # score per node at each iteration




class PageRank:
    def __init__(self) -> None:
        self.last_result: Optional[RankResult] = None

    @staticmethod
    def _node_order(G: nx.Graph) -> List[Hashable]:
        # Stable, explicit node order for matrix ops
        return list(G.nodes())

    @staticmethod
    def _adjacency(G: nx.Graph, nodes: List[Hashable]) -> np.ndarray:
        """
        Build a weighted adjacency matrix A where A[i, j] represents
        the weight of edge j -> i (column j influences row i).
        Uses edge attribute 'value' if present, otherwise 1.0.
        """
        n = len(nodes)
        index = {node: i for i, node in enumerate(nodes)}
        A = np.zeros((n, n), dtype=float)
        # For directed graphs, use direction; for undirected, add contributions both ways.
        if G.is_directed():
            for u, v, data in G.edges(data=True):
                w = float(data.get("value", 1.0))
                A[index[v], index[u]] += w  # column u flows to row v
        else:
            for u, v, data in G.edges(data=True):
                w = float(data.get("value", 1.0))
                iu, iv = index[u], index[v]
                A[iu, iv] += w
                A[iv, iu] += w
        return A

    @staticmethod
    def _column_stochastic(M: np.ndarray) -> np.ndarray:
        """Return a column-stochastic version of M (safe when a column sums to 0)."""
        col_sums = M.sum(axis=0)
        # Avoid division by zero: if a column sum is 0, leave that column as zeros for now.
        safe = col_sums.copy()
        safe[safe == 0.0] = 1.0
        return M / safe

    def _power_iteration(
        self,
        P: np.ndarray,
        nodes: List[Hashable],
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
        live_callback: Optional[callable] = None,
    ) -> RankResult:
        n = len(nodes)
        r = np.full(n, 1.0 / n, dtype=float)
        teleport = (1.0 - damping) / n
        trajectory: List[Dict[Hashable, float]] = []

        # Identify empty columns to redistribute their score each step
        empty_cols = np.where(P.sum(axis=0) == 0.0)[0]

        for _ in range(max_iter):
            prev = r.copy()
            r = teleport + damping * (P @ prev)

            # If there are empty columns, redistribute their previous score evenly
            if len(empty_cols) > 0:
                r += damping * prev[empty_cols].sum() / n

            # Track snapshot
            snapshot = {nodes[i]: float(f"{r[i]:.4f}") for i in range(n)}
            trajectory.append(snapshot)

            # Live streaming of intermediate scores
            if live_callback is not None:
                live_callback(snapshot)

            # Convergence check (L1 distance)
            if np.abs(r - prev).sum() < tol:
                break

        scores = {nodes[i]: float(r[i]) for i in range(n)}
        result = RankResult(scores=scores, order=nodes, trajectory=trajectory)
        self.last_result = result
        return result

    def rank(
        self,
        graph: ReactiveGraph,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
        sort_desc: bool = False,
        live_update: bool = True,
        update_attr: str = "value",
        sleep_between_iters: float = 0.0,
    ) -> RankResult:
        """
        Compute PageRank scores over the current graph.
        If live_update=True, writes each iteration's scores back to the graph node attribute
        `update_attr` so any reactive view will update automatically.
        """
        if not hasattr(graph, "nx"):
            raise TypeError("Expected a ReactiveGraph-like object with a .nx property")
        G: nx.Graph = graph.nx

        nodes = self._node_order(G)
        if len(nodes) == 0:
            return RankResult(scores={}, order=[], trajectory=[])

        A = self._adjacency(G, nodes)
        P = self._column_stochastic(A)

        def push(snapshot: Dict[Hashable, float]) -> None:
            if not live_update:
                return
            # Write scores into node attributes; this triggers reactive redraws.
            for node, score in snapshot.items():
                try:
                    graph.set_node_attrs(node, **{update_attr: score})
                except Exception:
                    # Best-effort: ignore nodes that might have been removed mid-run
                    pass
            if sleep_between_iters > 0.0:
                # Lightweight pacing so humans can see the updates
                import time as _t
                _t.sleep(sleep_between_iters)

        result = self._power_iteration(
            P,
            nodes,
            damping=damping,
            max_iter=max_iter,
            tol=tol,
            live_callback=push if live_update else None,
        )

        if sort_desc:
            # Return a new RankResult with sorted order
            ordered = sorted(result.scores.items(), key=lambda kv: kv[1], reverse=True)
            scores_sorted = dict(ordered)
            order_sorted = [k for k, _ in ordered]
            trajectory_sorted: List[Dict[Hashable, float]] = []
            for snap in result.trajectory:
                trajectory_sorted.append({k: snap.get(k, 0.0) for k in order_sorted})
            result = RankResult(scores=scores_sorted, order=order_sorted, trajectory=trajectory_sorted)

        return result



# ----------------------------
# Demo
# ----------------------------
if __name__ == "__main__":
    # Minimal live demo that shows auto-updating visual while PageRank runs.
    try:
        import threading
        import matplotlib.pyplot as plt

        g = ReactiveGraph(directed=True)  # directed example
        view = GraphMatplotlibView(
            g,
            layout="spring",
            node_value_key="value",   # visualize current score
            edge_value_key="value",   # show edge weights
            interval_ms=150,
        )

        # Seed nodes + weighted edges
        for n in range(1, 7):
            g.add_node(n, value=1.0 / 6)  # initial value for visualization
        edges = [
            (1, 2, 4.0),
            (2, 3, 2.0),
            (3, 1, 1.5),
            (3, 4, 1.0),
            (4, 5, 3.0),
            (5, 3, 5.0),
            (5, 6, 2.5),
            (6, 1, 1.0),
        ]
        for u, v, w in edges:
            g.add_edge(u, v, value=w)

        # Start the live view
        threading.Thread(target=lambda: view.start(block=True), daemon=True).start()

        # Run PageRank with live updates
        pr = PageRank()
        threading.Thread(
            target=lambda: pr.rank(g, damping=0.85, max_iter=50, tol=1e-8, live_update=True, update_attr="value", sleep_between_iters=0.2),
            daemon=True,
        ).start()

        # Keep process alive until figure closes
        plt.show()
    except Exception as e:
        print("Demo could not start:", e)
