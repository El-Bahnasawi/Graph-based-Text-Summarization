from __future__ import annotations

import threading
from typing import Callable, Dict, Hashable, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ----------------------------
# Reactive Graph Model
# ----------------------------
class ReactiveGraph:
    """A graph with node/edge attributes that notifies listeners on any change."""

    def __init__(self, directed: bool = False):
        self._G = nx.DiGraph() if directed else nx.Graph()
        self._listeners: List[Callable[[], None]] = []
        self._lock = threading.RLock()

    # ---- Listener management ----
    def subscribe(self, fn: Callable[[], None]) -> None:
        with self._lock:
            self._listeners.append(fn)

    def unsubscribe(self, fn: Callable[[], None]) -> None:
        with self._lock:
            self._listeners = [f for f in self._listeners if f is not fn]

    def _notify(self) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn()
            except Exception:
                pass  # never let a bad listener crash the model

    # ---- Node operations ----
    def add_node(self, node: Hashable, **attrs) -> None:
        with self._lock:
            self._G.add_node(node, **attrs)
        self._notify()

    def remove_node(self, node: Hashable) -> None:
        with self._lock:
            if node in self._G:
                self._G.remove_node(node)
        self._notify()

    def set_node_attrs(self, node: Hashable, **attrs) -> None:
        with self._lock:
            if node not in self._G:
                raise KeyError(f"Node {node!r} not found")
            self._G.nodes[node].update(attrs)
        self._notify()

    # ---- Edge operations ----
    def add_edge(self, u: Hashable, v: Hashable, **attrs) -> None:
        with self._lock:
            self._G.add_edge(u, v, **attrs)
        self._notify()

    def remove_edge(self, u: Hashable, v: Hashable) -> None:
        with self._lock:
            if self._G.has_edge(u, v):
                self._G.remove_edge(u, v)
        self._notify()

    def set_edge_attrs(self, u: Hashable, v: Hashable, **attrs) -> None:
        with self._lock:
            if not self._G.has_edge(u, v):
                raise KeyError(f"Edge {(u, v)!r} not found")
            self._G.edges[u, v].update(attrs)
        self._notify()

    # ---- Accessors ----
    @property
    def nx(self) -> nx.Graph:
        return self._G

    def nodes(self):
        return self._G.nodes

    def edges(self):
        return self._G.edges

    def __len__(self):
        return self._G.number_of_nodes()


# ----------------------------
# Matplotlib View (Reactive)
# ----------------------------
class GraphMatplotlibView:
    """
    A live-updating Matplotlib view for a ReactiveGraph.

    - Subscribes to graph changes and marks itself 'dirty'.
    - Uses FuncAnimation to redraw when dirty (max FPS set by `interval_ms`).
    - Colors: sentences (blue), names (green)
    - Edge colors: sent↔sent (black), sent↔name (gray), name↔name (red)
    """

    def __init__(
        self,
        model: ReactiveGraph,
        layout: str = "spring",
        node_value_key: Optional[str] = "value",
        edge_value_key: Optional[str] = "value",
        interval_ms: int = 250,
        k: Optional[float] = None,
        seed: Optional[int] = 42,
        show_legend: bool = True,
    ):
        self.model = model
        self.layout = layout
        self.node_value_key = node_value_key
        self.edge_value_key = edge_value_key
        self.interval_ms = max(50, int(interval_ms))
        self.k = k
        self.seed = seed
        self.show_legend = show_legend

        self._dirty = True
        self._lock = threading.RLock()

        self.fig, self.ax = plt.subplots(figsize=(7, 5), dpi=120)
        self.ax.set_axis_off()
        self.pos: Dict[Hashable, Tuple[float, float]] = {}
        self.ani: Optional[FuncAnimation] = None

        self.model.subscribe(self._on_model_change)

    def _on_model_change(self) -> None:
        with self._lock:
            self._dirty = True

    def _compute_layout(self) -> Dict[Hashable, Tuple[float, float]]:
        G = self.model.nx
        if self.layout == "spring":
            return nx.spring_layout(G, k=self.k, seed=self.seed)
        if self.layout == "kamada_kawai":
            return nx.kamada_kawai_layout(G)
        if self.layout == "circular":
            return nx.circular_layout(G)
        if self.layout == "shell":
            return nx.shell_layout(G)
        if self.layout == "spectral":
            return nx.spectral_layout(G)
        return nx.spring_layout(G, k=self.k, seed=self.seed)

    def _draw(self, _frame: int) -> None:
        redraw = False
        with self._lock:
            if self._dirty:
                self._dirty = False
                redraw = True
        if not redraw:
            return

        G = self.model.nx

        # Recompute layout when nodes change (or for force-directed layouts)
        if set(self.pos.keys()) != set(G.nodes) or self.layout in {"spring", "kamada_kawai"}:
            self.pos = self._compute_layout()

        self.ax.clear()
        self.ax.set_axis_off()

        # Node sizes from score
        vals: List[float] = []
        for n in G.nodes:
            v = G.nodes[n].get(self.node_value_key, 1.0) if self.node_value_key else 1.0
            try:
                vals.append(float(v))
            except Exception:
                vals.append(1.0)
        if vals:
            mn, mx = min(vals), max(vals)
            rng = (mx - mn) or 1.0
            sizes = [300 + 1000 * (v - mn) / rng for v in vals]
        else:
            sizes = 300

        # Node colors by kind
        node_colors: List[str] = []
        for n in G.nodes:
            kind = G.nodes[n].get("kind", "sent")
            node_colors.append("#2ca02c" if kind == "name" else "#1f77b4")  # green vs blue

        nx.draw_networkx_nodes(G, self.pos, ax=self.ax, node_size=sizes, node_color=node_colors)

        # Edge groups & widths
        sent_sent_edges, sent_name_edges, name_name_edges = [], [], []
        widths = {}
        for u, v, d in G.edges(data=True):
            ku, kv = G.nodes[u].get("kind", "sent"), G.nodes[v].get("kind", "sent")
            w = float(d.get(self.edge_value_key, 1.0)) if self.edge_value_key else 1.0
            widths[(u, v)] = 0.5 + 2.0 * (w / (1.0 + w))
            if ku == "name" and kv == "name":
                name_name_edges.append((u, v))
            elif ku == "sent" and kv == "sent":
                sent_sent_edges.append((u, v))
            else:
                sent_name_edges.append((u, v))

        if sent_sent_edges:
            nx.draw_networkx_edges(G, self.pos, edgelist=sent_sent_edges, ax=self.ax,
                                   width=[widths[e] for e in sent_sent_edges], edge_color="#000000")
        if sent_name_edges:
            nx.draw_networkx_edges(G, self.pos, edgelist=sent_name_edges, ax=self.ax,
                                   width=[widths[e] for e in sent_name_edges], edge_color="#777777")
        if name_name_edges:
            nx.draw_networkx_edges(G, self.pos, edgelist=name_name_edges, ax=self.ax,
                                   width=[widths[e] for e in name_name_edges], edge_color="#d62728")

        # Node labels: score for sentences; readable text for names
        labels: Dict[Hashable, str] = {}
        for n in G.nodes:
            if G.nodes[n].get("kind", "sent") == "name":
                labels[n] = str(G.nodes[n].get("label", str(n)))
            else:
                labels[n] = f"{G.nodes[n].get(self.node_value_key, 0):.4f}"
        nx.draw_networkx_labels(G, self.pos, labels=labels, ax=self.ax, font_size=8)

        # Edge labels (weights)
        if self.edge_value_key:
            edge_labels = {(u, v): str(d.get(self.edge_value_key, "")) for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, self.pos, edge_labels=edge_labels, ax=self.ax, font_size=7)

        # Title
        self.ax.set_title("Reactive Graph (auto-updates on changes)")

        # ---------- Legend (nodes + edges) ----------
        if self.show_legend:
            node_patches = [
                Patch(facecolor="#1f77b4", edgecolor="none", label="Sentence"),
                Patch(facecolor="#2ca02c", edgecolor="none", label="Name"),
            ]
            edge_lines = [
                Line2D([0], [0], color="#000000", lw=2, label="Sent ↔ Sent"),
                Line2D([0], [0], color="#777777", lw=2, label="Sent ↔ Name"),
                Line2D([0], [0], color="#d62728", lw=2, label="Name ↔ Name"),
            ]
            legend_handles = node_patches + edge_lines
            self.ax.legend(
                handles=legend_handles,
                loc="upper right",
                frameon=True,
                fancybox=True,
                framealpha=0.85,
                title="Legend",
            )

    def start(self, block: bool = True) -> Optional[FuncAnimation]:
        """Start the animation loop. Returns the animation handle."""
        self.ani = FuncAnimation(self.fig, self._draw, interval=self.interval_ms, cache_frame_data=False)
        if block:
            plt.show()
        return self.ani

    def stop(self) -> None:
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None
        plt.close(self.fig)

    def __enter__(self):
        self.start(block=False)
        return self

    def __exit__(self, exc_type, exc, tb):
        pass
