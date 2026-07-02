"""
Embedding layer for resume <-> role matching.

Tries to use `sentence-transformers` (all-MiniLM-L6-v2) for true semantic
embeddings. This requires downloading model weights from huggingface.co on
first use -- if that host is unreachable (e.g. sandboxed/offline environment,
or a corporate network without HF access) we transparently fall back to a
TF-IDF + Truncated SVD embedding, which needs no network access at all.

This mirrors the "graceful offline fallback" pattern used throughout this
project (see week7 RAG pipeline notes) so the notebook always runs end to
end regardless of network conditions.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

EMBED_DIM = 256


class EmbeddingBackend:
    """Unified interface: .fit(corpus) then .encode(texts) -> np.ndarray."""

    def __init__(self, prefer_transformer: bool = True, model_name: str = "all-MiniLM-L6-v2"):
        self.mode = None
        self.model = None
        self._tfidf = None
        self._svd = None

        if prefer_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.mode = "sentence-transformer"
            except Exception as e:
                print(f"[embeddings] sentence-transformers unavailable ({type(e).__name__}: "
                      f"{str(e)[:120]}...) -> falling back to TF-IDF+SVD embeddings.")
                self.mode = None

        if self.mode is None:
            self.mode = "tfidf-svd"

    def fit(self, corpus: list[str]):
        """Only needed for the tfidf-svd fallback; no-op for sentence-transformers."""
        if self.mode == "tfidf-svd":
            self._tfidf = TfidfVectorizer(
                max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2
            )
            X = self._tfidf.fit_transform(corpus)
            n_components = max(1, min(EMBED_DIM, X.shape[1] - 1, X.shape[0] - 1))
            self._svd = TruncatedSVD(n_components=n_components, random_state=42)
            self._svd.fit(X)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.mode == "sentence-transformer":
            result = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return np.asarray(result)
        else:
            if self._tfidf is None:
                raise RuntimeError("Call .fit(corpus) before .encode() in tfidf-svd mode.")
            X = self._tfidf.transform(texts)
            emb = self._svd.transform(X)
            # normalize for cosine similarity
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return emb / norms


def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between two sets of embeddings."""
    return cosine_similarity(a, b)


if __name__ == "__main__":
    backend = EmbeddingBackend(prefer_transformer=True)
    print("Backend mode:", backend.mode)
    corpus = ["python developer with django experience and rest apis",
              "java spring boot backend engineer with microservices",
              "frontend react typescript developer building web apps",
              "python data scientist with pandas and machine learning",
              "devops engineer using docker kubernetes and aws"]
    backend.fit(corpus)
    emb = backend.encode(corpus)
    print("Embedding shape:", emb.shape)
    print("Self-similarity (should be ~1 on diagonal):")
    print(np.round(similarity(emb, emb), 2))
