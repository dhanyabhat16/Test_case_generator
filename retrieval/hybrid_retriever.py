"""
hybrid_retriever.py
-------------------
Advanced RAG retrieval combining Sparse (BM25) + Dense (SBERT/FAISS) retrieval
with optional re-ranking and query expansion.

Syllabus coverage:
  - Sparse Retrieval (BM25)          → Module 2 & Module 3
  - Dense Retrieval (SBERT)          → Module 3
  - Hybrid Retrieval (BM25 + SBERT)  → Module 3
  - Re-Ranking                       → Module 3
  - Query Expansion (HyDE-style)     → Module 3

Usage:
    from retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(records)
    results = retriever.retrieve(query_text, input_type="code", top_k=5)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── BM25 Implementation (no external dependency) ──────────────────────────────

class BM25:
    """
    BM25 (Okapi BM25) sparse retrieval from scratch.

    BM25 is a bag-of-words ranking function that ranks documents by relevance
    to a query. It improves on TF-IDF by:
      - Saturating term frequency (high TF has diminishing returns)
      - Normalising for document length (k1, b parameters)

    Formula:
        score(D, Q) = Σ IDF(qi) * [ TF(qi,D) * (k1+1) ]
                                    / [ TF(qi,D) + k1*(1 - b + b*|D|/avgdl) ]

    Syllabus: Sparse Retrieval (BM25) — Module 2 & Module 3
    """

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        """
        Args:
            corpus : List of tokenized documents (each doc is a list of tokens).
            k1     : Term saturation parameter (default 1.5). Higher = less saturation.
            b      : Length normalization parameter (default 0.75). 1.0 = full norm.
        """
        self.k1 = k1
        self.b  = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / max(len(corpus), 1)

        # Build: doc frequency dict, term frequencies per doc
        self.df: Dict[str, int]        = {}
        self.tf: List[Dict[str, int]]  = []

        for doc in corpus:
            tf_doc: Dict[str, int] = {}
            for token in doc:
                tf_doc[token] = tf_doc.get(token, 0) + 1
            self.tf.append(tf_doc)
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1

    def idf(self, token: str) -> float:
        """Compute IDF with smoothing: log((N - df + 0.5) / (df + 0.5) + 1)"""
        df = self.df.get(token, 0)
        return math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document."""
        doc_tf   = self.tf[doc_idx]
        doc_len  = sum(doc_tf.values())
        total    = 0.0
        for token in query_tokens:
            if token not in doc_tf:
                continue
            tf_val = doc_tf[token]
            idf    = self.idf(token)
            norm   = tf_val * (self.k1 + 1)
            denom  = tf_val + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            total += idf * (norm / denom)
        return total

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """Return BM25 scores for all documents."""
        return [self.score(query_tokens, i) for i in range(self.corpus_size)]


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """
    Simple whitespace + punctuation tokenizer for BM25.
    Lowercases and splits on non-alphanumeric characters.
    """
    return re.findall(r"\w+", text.lower())


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Hybrid retrieval: combines BM25 (sparse) + SBERT/FAISS (dense) scores
    using Reciprocal Rank Fusion (RRF), then optionally re-ranks results.

    Architecture:
        Query
          ├── BM25 sparse search  → ranked list A (by BM25 score)
          ├── SBERT dense search  → ranked list B (by cosine similarity)
          └── RRF fusion          → merged ranked list
                └── Re-ranker     → final top-k

    Syllabus: Advanced RAG (Module 3) — Dense + Sparse + Hybrid + Re-ranking
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        embedding_model: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
    ):
        """
        Args:
            records         : List of chunk records from the FAISS store
                              (each has 'id', 'content', 'metadata').
            embedding_model : SBERT model name (must match embedder.py).
            rrf_k           : RRF constant. Higher = less weight to top ranks.
        """
        self.records  = records
        self.rrf_k    = rrf_k

        # Build BM25 corpus from all records
        self._corpus_tokens = [tokenize(r["content"]) for r in records]
        self._bm25 = BM25(self._corpus_tokens) if records else None

        # Load SBERT for dense encoding
        self._sbert = self._load_sbert(embedding_model)

        print(
            f"[HybridRetriever] Ready — {len(records)} records | "
            f"BM25 avgdl={self._bm25.avgdl:.1f}" if self._bm25 else
            f"[HybridRetriever] Ready — 0 records"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_text: str,
        input_type: Optional[str] = None,
        top_k: int = 5,
        alpha: float = 0.5,
        use_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: BM25 + SBERT fusion with optional re-ranking.

        Args:
            query_text  : The query string.
            input_type  : Filter to only return chunks of this type.
            top_k       : Number of final results.
            alpha       : Weight for dense score in fusion (0=BM25 only, 1=dense only).
                          Default 0.5 = equal weight hybrid.
            use_rerank  : Whether to apply cross-encoder re-ranking after fusion.

        Returns:
            List of result dicts with 'id', 'content', 'metadata', 'score',
            'rank', 'retrieval_method' fields.
        """
        if not query_text.strip() or not self.records:
            return []

        # Filter candidate pool by type first
        candidates, candidate_indices = self._filter_by_type(input_type)
        if not candidates:
            return []

        # ── Step 1: BM25 sparse search ────────────────────────────────────────
        bm25_results = self._bm25_search(query_text, candidates, candidate_indices)

        # ── Step 2: SBERT dense search ────────────────────────────────────────
        dense_results = self._dense_search(query_text, candidates, candidate_indices)

        # ── Step 3: Reciprocal Rank Fusion ────────────────────────────────────
        fused = self._reciprocal_rank_fusion(bm25_results, dense_results, alpha)

        # ── Step 4: Re-ranking (cross-encoder style via dot-product) ─────────
        if use_rerank and len(fused) > top_k:
            fused = self._rerank(query_text, fused, top_k * 2)

        # ── Step 5: Return top-k with metadata ───────────────────────────────
        final = fused[:top_k]
        enriched = []
        for rank, item in enumerate(final, start=1):
            record = self.records[item["original_idx"]]
            enriched.append({
                "id"               : record["id"],
                "content"          : record["content"],
                "metadata"         : record.get("metadata", {}),
                "score"            : round(item["fused_score"], 4),
                "bm25_score"       : round(item.get("bm25_score", 0.0), 4),
                "dense_score"      : round(item.get("dense_score", 0.0), 4),
                "rank"             : rank,
                "retrieval_method" : "hybrid_bm25_sbert",
                "source"           : self._build_source_label(record["metadata"]),
            })
        return enriched

    def retrieve_bm25_only(
        self,
        query_text: str,
        input_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """BM25-only sparse retrieval. Used for ablation comparison."""
        candidates, candidate_indices = self._filter_by_type(input_type)
        results = self._bm25_search(query_text, candidates, candidate_indices)
        final = results[:top_k]
        enriched = []
        for rank, item in enumerate(final, start=1):
            record = self.records[item["original_idx"]]
            enriched.append({
                "id"               : record["id"],
                "content"          : record["content"],
                "metadata"         : record.get("metadata", {}),
                "score"            : round(item["bm25_score"], 4),
                "rank"             : rank,
                "retrieval_method" : "bm25_sparse",
                "source"           : self._build_source_label(record["metadata"]),
            })
        return enriched

    def retrieve_dense_only(
        self,
        query_text: str,
        input_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """SBERT-only dense retrieval. Used for ablation comparison."""
        candidates, candidate_indices = self._filter_by_type(input_type)
        results = self._dense_search(query_text, candidates, candidate_indices)
        final = results[:top_k]
        enriched = []
        for rank, item in enumerate(final, start=1):
            record = self.records[item["original_idx"]]
            enriched.append({
                "id"               : record["id"],
                "content"          : record["content"],
                "metadata"         : record.get("metadata", {}),
                "score"            : round(item["dense_score"], 4),
                "rank"             : rank,
                "retrieval_method" : "dense_sbert",
                "source"           : self._build_source_label(record["metadata"]),
            })
        return enriched

    # ── Internal search methods ────────────────────────────────────────────────

    def _filter_by_type(
        self, input_type: Optional[str]
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Return (filtered_records, original_indices) for a given type."""
        if not input_type:
            return self.records, list(range(len(self.records)))
        filtered = [
            (i, r) for i, r in enumerate(self.records)
            if r.get("metadata", {}).get("type") == input_type
        ]
        if not filtered:
            return self.records, list(range(len(self.records)))
        indices  = [i for i, _ in filtered]
        records  = [r for _, r in filtered]
        return records, indices

    def _bm25_search(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        candidate_indices: List[int],
    ) -> List[Dict[str, Any]]:
        """Run BM25 on the candidate pool and return ranked results."""
        if not self._bm25 or not candidates:
            return []

        # Build a sub-corpus BM25 for the candidate set
        sub_tokens = [tokenize(c["content"]) for c in candidates]
        sub_bm25   = BM25(sub_tokens)
        q_tokens   = tokenize(query_text)
        scores     = sub_bm25.get_scores(q_tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        return [
            {
                "sub_idx"      : sub_idx,
                "original_idx" : candidate_indices[sub_idx],
                "bm25_score"   : score,
                "dense_score"  : 0.0,
                "fused_score"  : 0.0,
            }
            for sub_idx, score in ranked
            if score > 0
        ]

    def _dense_search(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        candidate_indices: List[int],
    ) -> List[Dict[str, Any]]:
        """Run SBERT cosine similarity on the candidate pool."""
        if self._sbert is None or not candidates:
            return []

        # Encode query
        q_vec = self._sbert.encode(query_text, convert_to_numpy=True)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)

        # Encode all candidate documents
        doc_texts = [c["content"] for c in candidates]
        doc_vecs  = self._sbert.encode(doc_texts, convert_to_numpy=True, batch_size=32)

        # Normalise doc vectors
        norms    = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9
        doc_vecs = doc_vecs / norms

        # Cosine similarity = dot product of normalised vectors
        scores = doc_vecs @ q_vec

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        return [
            {
                "sub_idx"      : sub_idx,
                "original_idx" : candidate_indices[sub_idx],
                "bm25_score"   : 0.0,
                "dense_score"  : float(score),
                "fused_score"  : 0.0,
            }
            for sub_idx, score in ranked
        ]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF):
            RRF_score(d) = alpha * 1/(k + rank_dense) + (1-alpha) * 1/(k + rank_bm25)

        This is the standard method for combining heterogeneous ranked lists
        without needing to normalize scores across different scales.

        Syllabus: Hybrid Retrieval (BM25 + SBERT) — Module 3
        """
        k = self.rrf_k
        fusion_scores: Dict[int, Dict[str, Any]] = {}

        # BM25 contribution
        for rank, item in enumerate(bm25_results, start=1):
            oi = item["original_idx"]
            if oi not in fusion_scores:
                fusion_scores[oi] = {
                    "original_idx" : oi,
                    "bm25_score"   : item["bm25_score"],
                    "dense_score"  : 0.0,
                    "fused_score"  : 0.0,
                }
            fusion_scores[oi]["bm25_score"]  = item["bm25_score"]
            fusion_scores[oi]["fused_score"] += (1 - alpha) / (k + rank)

        # Dense contribution
        for rank, item in enumerate(dense_results, start=1):
            oi = item["original_idx"]
            if oi not in fusion_scores:
                fusion_scores[oi] = {
                    "original_idx" : oi,
                    "bm25_score"   : 0.0,
                    "dense_score"  : item["dense_score"],
                    "fused_score"  : 0.0,
                }
            fusion_scores[oi]["dense_score"]  = item["dense_score"]
            fusion_scores[oi]["fused_score"] += alpha / (k + rank)

        # Sort by fused RRF score
        return sorted(
            fusion_scores.values(),
            key=lambda x: x["fused_score"],
            reverse=True,
        )

    def _rerank(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """
        Re-ranking: re-score top candidates using cross-encoder-style
        query-document dot product on SBERT embeddings.

        This is a lightweight re-ranker: it re-encodes both query and
        candidate together to get a finer-grained relevance score.

        Syllabus: Re-Ranking — Module 3
        """
        if self._sbert is None:
            return candidates

        pool = candidates[:top_n]
        q_vec = self._sbert.encode(query_text, convert_to_numpy=True)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)

        reranked = []
        for item in pool:
            record   = self.records[item["original_idx"]]
            doc_vec  = self._sbert.encode(record["content"], convert_to_numpy=True)
            doc_vec  = doc_vec / (np.linalg.norm(doc_vec) + 1e-9)
            # Re-score: cross-encoder approximation via dot product
            rerank_score = float(q_vec @ doc_vec)
            item_copy = dict(item)
            # Blend fused score with rerank score for final ordering
            item_copy["fused_score"] = 0.6 * item["fused_score"] + 0.4 * rerank_score
            reranked.append(item_copy)

        return sorted(reranked, key=lambda x: x["fused_score"], reverse=True)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _load_sbert(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name)
        except ImportError:
            print("[HybridRetriever] sentence-transformers not installed. Dense search disabled.")
            return None

    @staticmethod
    def _build_source_label(metadata: Any) -> str:
        if not isinstance(metadata, dict):
            return "unknown"
        itype     = metadata.get("type", "unknown")
        file_name = metadata.get("file_name", "unknown")
        if itype == "code":
            return f"{file_name} :: {metadata.get('function_name', '?')}()"
        elif itype == "requirements":
            return f"{file_name} :: {metadata.get('section', '?')}"
        elif itype == "api":
            return f"{file_name} :: {metadata.get('method','?')} {metadata.get('endpoint','?')}"
        return f"{file_name} :: chunk"


# ── Query Expansion (HyDE-style) ──────────────────────────────────────────────

def expand_query_hyde(
    original_query: str,
    input_type: str,
    llm_generate_fn=None,
) -> str:
    """
    HyDE (Hypothetical Document Embedding) query expansion.

    Instead of searching with the raw query, we ask the LLM to generate
    a hypothetical ideal chunk that would answer the query, then use THAT
    as the search query. This dramatically improves dense retrieval quality
    because we're searching embedding-space near the answer, not the question.

    Syllabus: Query Expansion — Module 3

    Args:
        original_query  : The original chunk content.
        input_type      : 'code', 'requirements', or 'api'.
        llm_generate_fn : Optional LLM function. If None, uses keyword expansion.

    Returns:
        Expanded query string.
    """
    if llm_generate_fn is None:
        # Fallback: keyword-based expansion (no LLM needed)
        return _keyword_expand(original_query, input_type)

    templates = {
        "code": (
            f"Write a short Python function that is similar to or related to: "
            f"{original_query[:300]}"
        ),
        "requirements": (
            f"Write a brief software requirement that is similar to: "
            f"{original_query[:300]}"
        ),
        "api": (
            f"Describe an API endpoint that is similar to: "
            f"{original_query[:300]}"
        ),
    }
    prompt = templates.get(input_type, f"Expand this search query: {original_query[:300]}")
    try:
        expanded = llm_generate_fn(prompt, max_tokens=200, temperature=0.3)
        return f"{original_query}\n\n{expanded}"
    except Exception:
        return original_query


def _keyword_expand(text: str, input_type: str) -> str:
    """
    Simple keyword expansion without an LLM.
    Adds domain-specific synonyms based on input type.
    """
    type_keywords = {
        "code"         : "function method test unit validate return assert",
        "requirements" : "requirement feature user story acceptance criteria",
        "api"          : "endpoint route HTTP request response status code",
    }
    domain_terms = type_keywords.get(input_type, "")
    return f"{text}\n{domain_terms}"


# ── Standalone integration with existing pipeline ─────────────────────────────

def get_hybrid_retriever_from_store() -> Optional[HybridRetriever]:
    """
    Build a HybridRetriever from the existing FAISS store records.
    Drop-in replacement for the existing retriever.
    """
    try:
        from retrieval.vector_store import get_store
        store   = get_store()
        records = store.records
        return HybridRetriever(records)
    except Exception as e:
        print(f"[HybridRetriever] Could not load store: {e}")
        return None


def retrieve_hybrid(
    query_text: str,
    input_type: str,
    top_k: int = 5,
    alpha: float = 0.5,
    use_query_expansion: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convenience function — drop-in replacement for retriever.retrieve().
    Uses hybrid BM25+SBERT with optional query expansion.

    Args:
        query_text          : The text to search with.
        input_type          : 'code', 'requirements', or 'api'.
        top_k               : Number of results.
        alpha               : Dense weight in fusion (0.5 = balanced hybrid).
        use_query_expansion : Whether to expand the query before searching.
    """
    retriever = get_hybrid_retriever_from_store()
    if retriever is None:
        # Fall back to original dense-only retriever
        from retrieval.retriever import retrieve
        return retrieve(query_text, input_type, top_k)

    # Optional query expansion
    search_text = query_text
    if use_query_expansion:
        search_text = expand_query_hyde(query_text, input_type)

    return retriever.retrieve(
        query_text=search_text,
        input_type=input_type,
        top_k=top_k,
        alpha=alpha,
        use_rerank=True,
    )


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick unit test of BM25
    corpus = [
        ["python", "function", "test", "validate", "email"],
        ["java", "method", "unit", "test", "class"],
        ["api", "endpoint", "http", "request", "response"],
        ["python", "class", "method", "return", "value"],
    ]
    bm25  = BM25(corpus)
    query = tokenize("python function test")
    scores = bm25.get_scores(query)
    print("BM25 scores:", [f"{s:.3f}" for s in scores])
    print("Top result index:", scores.index(max(scores)))
