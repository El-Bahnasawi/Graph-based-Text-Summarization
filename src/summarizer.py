# summarizer.py
from __future__ import annotations
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np

from reactive_graph import ReactiveGraph
from sent_embd import create_embedder
from Rank import PageRank

@dataclass
class SummaryConfig:
    """Configuration DTO for summarization parameters"""
    k: int = 3
    sim_threshold: float = 0.5
    embedder_kind: str = "sbert"
    use_sentence_edges: bool = True
    use_sent_name_edges: bool = True
    use_name_name_edges: bool = True
    damping: float = 0.85
    max_iter: int = 50

@dataclass
class SummaryResult:
    """Result DTO for summarization output"""
    top_indices: List[int]
    top_sentences: List[str]
    graph: ReactiveGraph
    sentence_scores: Dict[int, float]
    processing_time: float

class TextProcessor(ABC):
    """Abstract base class for text processing operations"""
    
    @abstractmethod
    def split_into_sentences(self, text: str) -> List[str]:
        pass
    
    @abstractmethod
    def extract_names(self, sentences: List[str]) -> List[List[str]]:
        pass

class DefaultTextProcessor(TextProcessor):
    """Default implementation of text processing operations"""
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using basic rules"""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        parts = re.split(r"(?<=[\.!\?])\s+(?=[A-Z0-9\"'])", text)
        return [s.strip() for s in parts if s.strip()]
    
    def extract_names(self, sentences: List[str]) -> List[List[str]]:
        """Extract named entities from sentences"""
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
            except Exception:
                raise ImportError("spaCy model not found")
            
            names_per_sent = []
            for doc in nlp.pipe(
                sentences,
                disable=["tagger", "lemmatizer", "morphologizer", "attribute_ruler"],
            ):
                names = []
                for ent in doc.ents:
                    if ent.label_ in {"PERSON", "ORG", "GPE"}:
                        nm = ent.text.strip()
                        if nm:
                            names.append(nm)
                names_per_sent.append(names)
            return names_per_sent
        except Exception:
            # Fallback to regex for title case words
            pat = re.compile(r"(?:[A-Z][a-z]+(?:[-'][A-Z][a-z]+)*)")
            return [[m.group(0) for m in pat.finditer(s)] for s in sentences]

class GraphBuilder:
    """Handles graph construction with different component types"""
    
    def __init__(self, value_key: str = "value", edge_key: str = "value"):
        self.value_key = value_key
        self.edge_key = edge_key
    
    def build_sentence_nodes(self, graph: ReactiveGraph, sentences: List[str]) -> None:
        """Add sentence nodes to the graph"""
        n = len(sentences)
        for i in range(n):
            graph.add_node(i, text=sentences[i], kind="sent", **{self.value_key: 1.0 / max(n, 1)})
    
    def build_sentence_edges(self, graph: ReactiveGraph, similarity_matrix: np.ndarray, 
                           threshold: float) -> None:
        """Add sentence-sentence edges based on similarity"""
        n = similarity_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                w = float(similarity_matrix[i, j])
                if w >= threshold:
                    graph.add_edge(i, j, **{self.edge_key: round(w, 3)})
    
    def build_entity_edges(self, graph: ReactiveGraph, sentences: List[str], 
                          names_per_sent: List[List[str]]) -> None:
        """Add entity-related edges (sentence-entity and entity-entity)"""
        from collections import Counter
        
        # Build sentence-entity edges
        for i, names in enumerate(names_per_sent):
            for nm in names:
                nm = nm.strip()
                if not nm:
                    continue
                nid = f"NAME: {nm}"
                if nid not in graph.nx:
                    graph.add_node(nid, label=nm, kind="name", **{self.value_key: 0.0})
                graph.add_edge(i, nid, **{self.edge_key: 1.0})
        
        # Build entity-entity edges
        pair_counts = Counter()
        for names in names_per_sent:
            uniq = sorted(set(n.strip() for n in names if n.strip()))
            for a_idx in range(len(uniq)):
                for b_idx in range(a_idx + 1, len(uniq)):
                    a, b = uniq[a_idx], uniq[b_idx]
                    pair_counts[(a, b)] += 1
        
        for (a, b), c in pair_counts.items():
            na, nb = f"NAME: {a}", f"NAME: {b}"
            if na not in graph.nx:
                graph.add_node(na, label=a, kind="name", **{self.value_key: 0.0})
            if nb not in graph.nx:
                graph.add_node(nb, label=b, kind="name", **{self.value_key: 0.0})
            graph.add_edge(na, nb, **{self.edge_key: float(c)})

class EmbeddingService:
    """Handles sentence embedding operations"""
    
    @staticmethod
    def l2_normalize(mat: np.ndarray) -> np.ndarray:
        """Normalize matrix using L2 norm"""
        n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / n
    
    @staticmethod
    def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix from embeddings"""
        normalized = EmbeddingService.l2_normalize(embeddings.astype(np.float64))
        similarity = np.matmul(normalized, normalized.T)
        np.fill_diagonal(similarity, 0.0)  # No self-edges
        return similarity

class GraphSummarizer:
    """Main orchestrator for graph-based text summarization"""
    
    def __init__(self, text_processor: Optional[TextProcessor] = None,
                 graph_builder: Optional[GraphBuilder] = None):
        self.text_processor = text_processor or DefaultTextProcessor()
        self.graph_builder = graph_builder or GraphBuilder()
        self.embedding_service = EmbeddingService()
    
    def summarize(self, text: str, config: SummaryConfig, 
                  embedder: Optional[any] = None) -> SummaryResult:
        """
        Main summarization method that orchestrates the entire process
        """
        import time
        start_time = time.time()
        
        # Step 1: Text processing
        sentences = self.text_processor.split_into_sentences(text)
        if not sentences:
            return SummaryResult([], [], ReactiveGraph(), {}, 0.0)
        
        # Step 2: Embedding
        if embedder is None:
            embedder = create_embedder(config.embedder_kind)
        embeddings = embedder.encode(sentences)
        similarity_matrix = self.embedding_service.cosine_similarity_matrix(embeddings)
        
        # Step 3: Graph construction
        graph = ReactiveGraph(directed=False)
        
        # Add sentence nodes
        self.graph_builder.build_sentence_nodes(graph, sentences)
        
        # Add edges based on configuration
        if config.use_sentence_edges:
            self.graph_builder.build_sentence_edges(graph, similarity_matrix, config.sim_threshold)
        
        if config.use_sent_name_edges or config.use_name_name_edges:
            names_per_sent = self.text_processor.extract_names(sentences)
            self.graph_builder.build_entity_edges(graph, sentences, names_per_sent)
        
        # Step 4: PageRank scoring
        pr = PageRank()
        result = pr.rank(
            graph,
            damping=config.damping,
            max_iter=config.max_iter,
            tol=1e-6,
            sort_desc=True,
            live_update=False,
            update_attr=self.graph_builder.value_key,
            sleep_between_iters=0
        )
        
        # Step 5: Extract top sentences
        sentence_scores = {i: s for i, s in result.scores.items() if isinstance(i, int)}
        top_sorted = sorted(sentence_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:config.k]
        top_indices = sorted([i for i, _ in top_sorted])
        top_sentences = [sentences[i] for i in top_indices]
        
        processing_time = time.time() - start_time
        
        return SummaryResult(
            top_indices=top_indices,
            top_sentences=top_sentences,
            graph=graph,
            sentence_scores=sentence_scores,
            processing_time=processing_time
        )

# Legacy function for backward compatibility
def summarize_text(text: str, k: int = 3, sim_threshold: float = 0.5,
                   embedder_kind: str = "sbert", embedder: Optional[any] = None,
                   enable_visualization: bool = True) -> Tuple[List[int], List[str], ReactiveGraph]:
    """Legacy function maintaining original interface"""
    config = SummaryConfig(k=k, sim_threshold=sim_threshold, embedder_kind=embedder_kind)
    summarizer = GraphSummarizer()
    result = summarizer.summarize(text, config, embedder)
    return result.top_indices, result.top_sentences, result.graph