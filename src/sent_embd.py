# embeddings.py
# Strategy + Factory for sentence embeddings
# Requires: numpy
# Optional per embedder:
#   - BERT/SBERT: torch, transformers (and sentence-transformers if using sbert)
#   - TF-IDF / BoW: scikit-learn

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Type
import numpy as np
import torch
import pickle
import os


# ---------- Base Strategy ----------
class SentenceEmbedder:
    """Strategy interface for sentence embedding backends."""
    def encode(self, sentences: List[str]) -> np.ndarray:
        """Return array shape [N, D]. Must be float32/float64."""
        raise NotImplementedError
    
    def save(self, model_path: str) -> None:
        """Save the model to disk (optional for some embedders)."""
        pass
    
    def load(self, model_path: str) -> None:
        """Load the model from disk (optional for some embedders)."""
        pass


# ---------- Utilities ----------
def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


# ---------- Concrete Strategies ----------
@dataclass
class BERTEmbedder(SentenceEmbedder):
    """Mean-pooled BERT/Transformer embeddings via HuggingFace."""
    model_name: str = "bert-base-uncased"
    normalize: bool = True
    device: str = "cpu"  # Default to CPU

    _tok: Any = None
    _mdl: Any = None

    def __post_init__(self):
        try:
            import torch  # noqa
            from transformers import AutoTokenizer, AutoModel  # noqa
        except Exception as e:
            raise ImportError(
                "BERTEmbedder requires 'torch' and 'transformers'. "
                "Install with: pip install torch transformers"
            ) from e

    def _ensure_loaded(self):
        if self._tok is not None and self._mdl is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._mdl = AutoModel.from_pretrained(self.model_name)
        self._mdl.to(self.device).eval()

    @torch.no_grad()  # type: ignore  # added at runtime when torch is present
    def encode(self, sentences: List[str]) -> np.ndarray:  # type: ignore
        import torch
        self._ensure_loaded()
        enc = self._tok(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self._mdl(**enc).last_hidden_state  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        summed = (out * mask).sum(dim=1)            # [B, H]
        counts = mask.sum(dim=1).clamp(min=1)       # [B, 1]
        emb = summed / counts
        emb = emb.detach().cpu().numpy()
        return _l2_normalize(emb) if self.normalize else emb


@dataclass
class SbertEmbedder(SentenceEmbedder):
    """Sentence-Transformers encoder (e.g., all-MiniLM-L6-v2)."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    device: str = "cpu"  # Default to CPU

    _mdl: Any = None

    def __post_init__(self):
        try:
            # sentence-transformers pulls torch + transformers under the hood
            import sentence_transformers  # noqa
        except Exception as e:
            raise ImportError(
                "SbertEmbedder requires 'sentence-transformers'. "
                "Install with: pip install sentence-transformers"
            ) from e

    def _ensure_loaded(self):
        from sentence_transformers import SentenceTransformer
        if self._mdl is None:
            self._mdl = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, sentences: List[str]) -> np.ndarray:
        self._ensure_loaded()
        emb = np.asarray(self._mdl.encode(sentences, convert_to_numpy=True, normalize_embeddings=False))
        return _l2_normalize(emb) if self.normalize else emb


@dataclass
class TfidfEmbedder(SentenceEmbedder):
    """TF-IDF sparse embeddings via scikit-learn (L2-normalized)."""
    max_features: Optional[int] = 20000
    ngram_range: tuple = (1, 2)
    lowercase: bool = True
    analyzer: str = "word"
    binary: bool = False
    normalize: bool = True
    model_path: Optional[str] = None  # Path to save/load the model

    _vec: Any = None

    def __post_init__(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa
        except Exception as e:
            raise ImportError(
                "TfidfEmbedder requires scikit-learn. "
                "Install with: pip install scikit-learn"
            ) from e

    def fit(self, corpus: List[str]) -> None:
        """Fit the vectorizer on a corpus of documents."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=self.lowercase,
            analyzer=self.analyzer,
            binary=self.binary,
        )
        self._vec.fit(corpus)

    def encode(self, sentences: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        if self._vec is None:
            if self.model_path and os.path.exists(self.model_path):
                self.load(self.model_path)
            else:
                # Fallback: fit on the provided sentences (not ideal but maintains compatibility)
                self.fit(sentences)
        
        X = self._vec.transform(sentences)
        X = X.astype(np.float32)
        dense = X.toarray()
        return _l2_normalize(dense) if self.normalize else dense

    def save(self, model_path: str) -> None:
        """Save the fitted vectorizer to disk."""
        if self._vec is not None:
            with open(model_path, 'wb') as f:
                pickle.dump(self._vec, f)
        else:
            raise ValueError("No fitted vectorizer to save. Call fit() first.")

    def load(self, model_path: str) -> None:
        """Load a fitted vectorizer from disk."""
        with open(model_path, 'rb') as f:
            self._vec = pickle.load(f)


@dataclass
class BowEmbedder(SentenceEmbedder):
    """Bag-of-Words count vectors via scikit-learn (optionally normalized)."""
    max_features: Optional[int] = 20000
    ngram_range: tuple = (1, 1)
    lowercase: bool = True
    analyzer: str = "word"
    binary: bool = False
    normalize: bool = True
    model_path: Optional[str] = None  # Path to save/load the model

    _vec: Any = None

    def __post_init__(self):
        try:
            from sklearn.feature_extraction.text import CountVectorizer  # noqa
        except Exception as e:
            raise ImportError(
                "BowEmbedder requires scikit-learn. "
                "Install with: pip install scikit-learn"
            ) from e

    def fit(self, corpus: List[str]) -> None:
        """Fit the vectorizer on a corpus of documents."""
        from sklearn.feature_extraction.text import CountVectorizer
        self._vec = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=self.lowercase,
            analyzer=self.analyzer,
            binary=self.binary,
        )
        self._vec.fit(corpus)

    def encode(self, sentences: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import CountVectorizer
        
        if self._vec is None:
            if self.model_path and os.path.exists(self.model_path):
                self.load(self.model_path)
            else:
                # Fallback: fit on the provided sentences (not ideal but maintains compatibility)
                self.fit(sentences)
        
        X = self._vec.transform(sentences)
        dense = X.astype(np.float32).toarray()
        return _l2_normalize(dense) if self.normalize else dense

    def save(self, model_path: str) -> None:
        """Save the fitted vectorizer to disk."""
        if self._vec is not None:
            with open(model_path, 'wb') as f:
                pickle.dump(self._vec, f)
        else:
            raise ValueError("No fitted vectorizer to save. Call fit() first.")

    def load(self, model_path: str) -> None:
        """Load a fitted vectorizer from disk."""
        with open(model_path, 'rb') as f:
            self._vec = pickle.load(f)


# ---------- Factory ----------
_REGISTRY: Dict[str, Type[SentenceEmbedder]] = {
    "bert": BERTEmbedder,
    "sbert": SbertEmbedder,
    "tfidf": TfidfEmbedder,
    "bow": BowEmbedder,
}

def create_embedder(kind: str, **kwargs) -> SentenceEmbedder:
    """
    Factory: create an embedder by name.

    Examples:
        create_embedder("bert", model_name="bert-base-uncased")
        create_embedder("sbert", model_name="sentence-transformers/all-MiniLM-L6-v2")
        create_embedder("tfidf", ngram_range=(1,2))
        create_embedder("bow", max_features=10000)
    """
    key = kind.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown embedder '{kind}'. Available: {list(_REGISTRY)}")
    cls = _REGISTRY[key]
    return cls(**kwargs)