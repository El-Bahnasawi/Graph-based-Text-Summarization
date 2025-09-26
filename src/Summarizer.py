# summarizer.py
# Requirements:
#   pip install numpy networkx matplotlib
#   (optional) pip install spacy && python -m spacy download en_core_web_sm
#   (optional, depending on embedder choice): torch, transformers, scikit-learn
#
# This script:
#   1) splits input text into sentences
#   2) builds embeddings via sent_embd.create_embedder(kind=...)
#   3) builds an undirected weighted graph with THREE edge types:
#        a) sentence–sentence (similarity-based)
#        b) sentence–name (name present in sentence)
#        c) name–name (names co-occur in a sentence)
#   4) runs PageRank with live updates to node "value"
#   5) prints top-k sentences in original order

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np

from src.reactive_graph import ReactiveGraph, GraphMatplotlibView
from src.sent_embd import create_embedder
from src.Rank import PageRank  # uses graph's 'value' attr for live updates

# --------------------
# Config
# --------------------
VALUE_KEY = "value"         # node score attribute (PageRank updates this)
EDGE_KEY = "value"          # edge weight attribute (used by PR implementation)
SIM_THRESHOLD = 0.50        # min cosine sim for sentence–sentence edge
EMBEDDER_KIND = "bert"     # e.g., "tfidf", "bow", "bert", "sbert"
TOP_K = 3                   # sentences to print in summary


# --------------------
# Utilities
# --------------------
def split_into_sentences(text: str) -> List[str]:
    """Very simple sentence splitter; good enough for demos."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Split on ., !, ? that look like sentence ends
    parts = re.split(r"(?<=[\.!\?])\s+(?=[A-Z0-9\"'])", text)
    # Keep punctuation and drop empty
    sents = [s.strip() for s in parts if s.strip()]
    return sents


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / n


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    Xn = l2_normalize(X.astype(np.float64))
    S = np.matmul(Xn, Xn.T)
    # zero the diagonal for clarity (no self-edges from similarity builder)
    np.fill_diagonal(S, 0.0)
    return S


# --------------------
# Graph construction: sentence–sentence by similarity
# --------------------
def build_graph_from_similarity(
    g: ReactiveGraph,
    sents: List[str],
    S: np.ndarray,
    threshold: float,
) -> None:
    """Add sentence nodes and sentence–sentence edges where sim >= threshold."""
    n = len(sents)
    for i in range(n):
        g.add_node(i, text=sents[i], kind="sent", **{VALUE_KEY: 1.0 / max(n, 1)})

    for i in range(n):
        for j in range(i + 1, n):
            w = float(S[i, j])
            if w >= threshold:
                g.add_edge(i, j, **{EDGE_KEY: round(w, 3)})


# --------------------
# Name extraction + edges (NEW)
# --------------------
def _extract_names_per_sentence(sents: List[str]) -> List[List[str]]:
    """
    Returns a list (per sentence) of detected 'names'.
    Tries spaCy PERSON/ORG/GPE; falls back to a simple Title-case regex.
    """
    try:
        import spacy  # type: ignore
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            raise ImportError

        out: List[List[str]] = []
        for doc in nlp.pipe(
            sents,
            disable=["tagger", "lemmatizer", "morphologizer", "attribute_ruler"],
        ):
            names = []
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE"}:
                    nm = ent.text.strip()
                    if nm:
                        names.append(nm)
            out.append(names)
        return out
    except Exception:
        # Regex fallback: sequences of TitleCase tokens (allows hyphen/apostrophe)
        pat = re.compile(r"(?:[A-Z][a-z]+(?:[-'][A-Z][a-z]+)*)")
        out = []
        for s in sents:
            tokens = [m.group(0) for m in pat.finditer(s)]
            out.append(tokens)
        return out


def add_sentence_name_edges(
    g: ReactiveGraph,
    sents: List[str],
    names_per_sent: List[List[str]],
    edge_key: str = EDGE_KEY,
    sent_name_weight: float = 1.0,
) -> None:
    """
    Ensures name nodes exist; adds:
      - sentence<->name edges (weight accumulates per occurrence)
      - name<->name edges for co-occurrence within the same sentence (weight = count)
    """
    def _name_node_id(name: str) -> str:
        return f"NAME: {name}"

    # Sentence <-> Name
    for i, names in enumerate(names_per_sent):
        for nm in names:
            nm = nm.strip()
            if not nm:
                continue
            nid = _name_node_id(nm)
            if nid not in g.nx:
                g.add_node(nid, label=nm, kind="name", **{VALUE_KEY: 0.0})

            if g.nx.has_edge(i, nid):
                w = float(g.nx.edges[i, nid].get(edge_key, 0.0)) + sent_name_weight
                g.set_edge_attrs(i, nid, **{edge_key: round(w, 3)})
            elif g.nx.has_edge(nid, i):
                w = float(g.nx.edges[nid, i].get(edge_key, 0.0)) + sent_name_weight
                g.set_edge_attrs(nid, i, **{edge_key: round(w, 3)})
            else:
                g.add_edge(i, nid, **{edge_key: float(sent_name_weight)})

    # Name <-> Name co-occurrence
    from collections import Counter
    pair_counts: Counter[Tuple[str, str]] = Counter()
    for names in names_per_sent:
        uniq = sorted(set(n.strip() for n in names if n.strip()))
        for a_idx in range(len(uniq)):
            for b_idx in range(a_idx + 1, len(uniq)):
                a, b = uniq[a_idx], uniq[b_idx]
                pair_counts[(a, b)] += 1

    for (a, b), c in pair_counts.items():
        na, nb = _name_node_id(a), _name_node_id(b)
        if na not in g.nx:
            g.add_node(na, label=a, kind="name", **{VALUE_KEY: 0.0})
        if nb not in g.nx:
            g.add_node(nb, label=b, kind="name", **{VALUE_KEY: 0.0})
        if g.nx.has_edge(na, nb):
            w = float(g.nx.edges[na, nb].get(edge_key, 0.0)) + float(c)
            g.set_edge_attrs(na, nb, **{edge_key: round(w, 3)})
        elif g.nx.has_edge(nb, na):
            w = float(g.nx.edges[nb, na].get(edge_key, 0.0)) + float(c)
            g.set_edge_attrs(nb, na, **{edge_key: round(w, 3)})
        else:
            g.add_edge(na, nb, **{edge_key: float(c)})


# --------------------
# Orchestration
# --------------------
def summarize_text(
    text: str,
    k: int = TOP_K,
    sim_threshold: float = SIM_THRESHOLD,
    embedder_kind: str = EMBEDDER_KIND,
    embedder: Optional[any] = None,  # Allow passing pre-initialized embedder
    enable_visualization: bool = True  # Allow disabling visualization
) -> Tuple[List[int], List[str]]:  # Changed return type
    """
    Builds the graph, runs PageRank live, and returns indices of top-k sentences
    in original order. Also prints the selected sentences.
    """
    sents = split_into_sentences(text)
    if not sents:
        print("No sentences found.")
        return [], []

    # Embeddings - use provided embedder or create new one
    if embedder is None:
        embedder = create_embedder(embedder_kind)
    X = embedder.encode(sents)  # (n, d) numpy array expected
    S = cosine_similarity_matrix(X)

    # Graph + live view (optional)
    g = ReactiveGraph(directed=False)
    view = None
    if enable_visualization:
        view = GraphMatplotlibView(
            g,
            layout="spring",
            node_value_key=VALUE_KEY,
            edge_value_key=EDGE_KEY,
            interval_ms=150,
            seed=7,
        )
        view.start(block=False)

    # 1) sentence–sentence
    build_graph_from_similarity(g, sents, S, sim_threshold)

    # 2) sentence–name + 3) name–name
    names_per_sent = _extract_names_per_sentence(sents)
    add_sentence_name_edges(g, sents, names_per_sent)

    # Run PageRank with live updates into node attr "value"
    pr = PageRank()
    res = pr.rank(
        g,
        damping=0.85,
        max_iter=50,
        tol=1e-6,
        sort_desc=True,
        live_update=enable_visualization,  # Only update if visualization is enabled
        update_attr=VALUE_KEY,
        sleep_between_iters=0.15 if enable_visualization else 0,  # No sleep if no visualization
    )

    # Select top-k sentence indices but preserve original input order
    sent_scores = {i: s for i, s in res.scores.items() if isinstance(i, int)}
    top_sorted = sorted(sent_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    top_ids = sorted([i for i, _ in top_sorted])
    top_sentences = [sents[i] for i in top_ids]  # Extract the actual sentences

    if enable_visualization:
        print("\n=== PageRank scores (node -> score) ===")
        for i, sc in sorted(sent_scores.items(), key=lambda kv: -kv[1]):
            print(f" {i} -> {sc:.4f} | {sents[i]}")

        print(f"\n=== Top {k} sentences (in input order) ===")
        for i in top_ids:
            print(f"- {sents[i]}")

    return top_ids, top_sentences  # Return both indices and sentences


# --------------------
# Demo
# --------------------
if __name__ == "__main__":
    demo_text = (
        "Large language models are transforming natural language processing. "
        "However, deploying them efficiently requires careful system design. "
        "OpenAI and Google have both proposed optimizations. "
        "In our approach, each sentence becomes a node and edges reflect cosine similarity. "
        "We construct an undirected, weighted graph with a similarity threshold of 0.5. "
        "Then we run PageRank with live visualization to identify influential sentences. "
        "Finally, we select the top three sentences as the extractive summary. "
        "This pipeline balances interpretability, performance, and practicality."
    )
    top_indices, top_sentences = summarize_text(demo_text, k=3)
    import matplotlib.pyplot as plt
    plt.show()