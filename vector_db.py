import math
import os
import uuid
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FilterSelector

from schemas import TextSegment, RAGSearchResult, ContextPassage

class LexicalScorerBM25:
    """Lightweight BM25 implementation for lexical relevance ranking."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.corpus_size: int = 0
        self.tokenized_corpus: List[List[str]] = []
        self.idf_map: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        return [word.lower() for word in text.split() if word.isalnum() or len(word) > 1]

    def fit(self, documents: List[str]):
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.corpus_size = len(self.tokenized_corpus)
        if self.corpus_size == 0:
            return

        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lens) / self.corpus_size

        doc_freqs: Dict[str, int] = {}
        for doc in self.tokenized_corpus:
            for word in set(doc):
                doc_freqs[word] = doc_freqs.get(word, 0) + 1

        for word, freq in doc_freqs.items():
            self.idf_map[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str) -> List[float]:
        query_words = self._tokenize(query)
        scores = [0.0] * self.corpus_size

        for word in query_words:
            if word not in self.idf_map:
                continue
            idf_val = self.idf_map[word]
            for idx, doc in enumerate(self.tokenized_corpus):
                term_freq = doc.count(word)
                if term_freq == 0:
                    continue
                numerator = term_freq * (self.k1 + 1.0)
                denominator = term_freq + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[idx] / self.avg_doc_len))
                scores[idx] += idf_val * (numerator / denominator)

        return scores

_qdrant_client_singleton: Optional[QdrantClient] = None
_cross_encoder_registry = {}

def get_qdrant_client(url: Optional[str] = None, storage_path: str = "qdrant_storage") -> QdrantClient:
    """Returns a shared, thread-safe instance of QdrantClient."""
    global _qdrant_client_singleton
    if _qdrant_client_singleton is not None:
        return _qdrant_client_singleton
    
    if url:
        _qdrant_client_singleton = QdrantClient(url=url, timeout=30)
    else:
        try:
            _qdrant_client_singleton = QdrantClient(path=storage_path)
        except Exception:
            _qdrant_client_singleton = QdrantClient(location=":memory:")
            
    return _qdrant_client_singleton

_active_collections: Set[str] = set()

class QdrantEngine:
    """Handles communication with Qdrant and implements search dispatchers."""
    
    def __init__(self, collection_name: str | None = None, vector_dim: int = 384):
        url = os.getenv("QDRANT_URL")
        storage_path = os.getenv("QDRANT_PATH", "qdrant_storage")

        self.client = get_qdrant_client(url=url, storage_path=storage_path)
        self.collection = collection_name or os.getenv("QDRANT_COLLECTION", "docs")
        self.vector_dim = vector_dim

        global _active_collections
        if self.collection not in _active_collections:
            try:
                if not self.client.collection_exists(self.collection):
                    self.client.create_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
                    )
            except Exception:
                pass
            _active_collections.add(self.collection)

    def reset_collection(self):
        """Wipes the existing collection to reset indices."""
        try:
            if self.client.collection_exists(self.collection):
                self.client.delete(
                    collection_name=self.collection,
                    points_selector=FilterSelector(filter=Filter())
                )
        except Exception:
            pass

        try:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
        except Exception:
            pass

        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
        except Exception:
            pass

    def store_segments(self, segments: List[TextSegment], vectors: List[List[float]]):
        """Persists segment text and dense vectors into Qdrant."""
        if len(segments) != len(vectors):
            raise ValueError("Size mismatch between segments and embedding vectors.")
        if not segments:
            return

        points = []
        for i, segment in enumerate(segments):
            # Generate stable ID based on segment parameters
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{segment.doc_id}:{segment.page_num}:{segment.seq_index}:{hash(segment.content)}"
                )
            )
            payload = {
                "text": segment.content,
                "source": segment.doc_id,
                "page": segment.page_num,
                "chunk_id": segment.seq_index,
            }
            points.append(PointStruct(id=point_id, vector=vectors[i], payload=payload))

        self.client.upsert(self.collection, points=points)

    def list_documents(self) -> List[str]:
        """Scans the collection and extracts unique source document names."""
        docs = set()
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection,
                limit=1000,
                with_payload=["source"],
                with_vectors=False
            )
            for record in records:
                if record.payload and "source" in record.payload:
                    docs.add(record.payload["source"])
        except Exception:
            pass
        return list(docs)

    def _query(self, vector: List[float], limit: int = 5, retrieve_vectors: bool = False):
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                with_payload=True,
                with_vectors=retrieve_vectors,
                limit=limit,
            )
        else:
            res = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                with_payload=True,
                with_vectors=retrieve_vectors,
                limit=limit,
            )
            return getattr(res, "points", [])

    def search_dense(self, vector: List[float], limit: int = 5) -> List[ContextPassage]:
        """Runs standard dense vector search."""
        hits = self._query(vector, limit=limit)
        results = []
        for hit in hits:
            payload = getattr(hit, "payload", None) or {}
            content = payload.get("text", "")
            doc_id = payload.get("source", "Unknown")
            page_num = int(payload.get("page", 1))
            seq_index = int(payload.get("chunk_id", 0))
            score = float(getattr(hit, "score", 0.0))

            if content:
                results.append(
                    ContextPassage(
                        content=content,
                        doc_id=doc_id,
                        page_num=page_num,
                        seq_index=seq_index,
                        relevance_score=score
                    )
                )
        return results

    def search_mmr(
        self,
        vector: List[float],
        limit: int = 5,
        diversity_factor: float = 0.7,
        candidate_pool_size: int = 15
    ) -> List[ContextPassage]:
        """Executes MMR search balancing match relevance and embedding diversity."""
        pool = self._query(vector, limit=candidate_pool_size, retrieve_vectors=True)
        if not pool:
            return []

        pool_vecs = [np.array(p.vector) for p in pool if getattr(p, "vector", None) is not None]
        if not pool_vecs:
            return self.search_dense(vector, limit=limit)

        selected: List[int] = []
        unselected: List[int] = list(range(len(pool_vecs)))

        for _ in range(min(limit, len(pool_vecs))):
            max_mmr = -float("inf")
            target_idx = -1

            for idx in unselected:
                relevance = pool[idx].score

                if not selected:
                    max_similarity = 0.0
                else:
                    similarities = [
                        float(
                            np.dot(pool_vecs[idx], pool_vecs[sel_idx])
                            / (np.linalg.norm(pool_vecs[idx]) * np.linalg.norm(pool_vecs[sel_idx]) + 1e-9)
                        )
                        for sel_idx in selected
                    ]
                    max_similarity = max(similarities)

                mmr_score = diversity_factor * relevance - (1.0 - diversity_factor) * max_similarity

                if mmr_score > max_mmr:
                    max_mmr = mmr_score
                    target_idx = idx

            if target_idx != -1:
                selected.append(target_idx)
                unselected.remove(target_idx)

        results = []
        for idx in selected:
            hit = pool[idx]
            payload = getattr(hit, "payload", None) or {}
            results.append(
                ContextPassage(
                    content=payload.get("text", ""),
                    doc_id=payload.get("source", "Unknown"),
                    page_num=int(payload.get("page", 1)),
                    seq_index=int(payload.get("chunk_id", 0)),
                    relevance_score=float(hit.score)
                )
            )
        return results

    def search_hybrid(
        self,
        query: str,
        vector: List[float],
        limit: int = 5,
        fusion_k: int = 60
    ) -> List[ContextPassage]:
        """Combines BM25 lexical ranking and dense vector search via RRF."""
        dense_hits = self._query(vector, limit=limit * 4)
        if not dense_hits:
            return []

        # Run BM25 on the dense retrieval pool
        corpus = [getattr(h, "payload", {}).get("text", "") for h in dense_hits]
        bm25 = LexicalScorerBM25()
        bm25.fit(corpus)
        lexical_scores = bm25.score(query)

        bm25_sorted_indices = np.argsort(lexical_scores)[::-1]
        bm25_ranks = {dense_hits[idx].id: rank + 1 for rank, idx in enumerate(bm25_sorted_indices)}
        dense_ranks = {h.id: rank + 1 for rank, h in enumerate(dense_hits)}

        fused_scores: Dict[str, float] = {}
        hit_registry = {h.id: h for h in dense_hits}

        for point_id in hit_registry:
            d_rank = dense_ranks.get(point_id, len(dense_hits) + 1)
            l_rank = bm25_ranks.get(point_id, len(dense_hits) + 1)
            # Reciprocal Rank Fusion formula
            score = (1.0 / (fusion_k + d_rank)) + (1.0 / (fusion_k + l_rank))
            fused_scores[point_id] = score

        top_point_ids = sorted(fused_scores.keys(), key=lambda pid: fused_scores[pid], reverse=True)[:limit]

        results = []
        for pid in top_point_ids:
            hit = hit_registry[pid]
            payload = getattr(hit, "payload", None) or {}
            results.append(
                ContextPassage(
                    content=payload.get("text", ""),
                    doc_id=payload.get("source", "Unknown"),
                    page_num=int(payload.get("page", 1)),
                    seq_index=int(payload.get("chunk_id", 0)),
                    relevance_score=float(fused_scores[pid])
                )
            )
        return results

    def search_multi_query(
        self,
        query_vectors: List[List[float]],
        limit: int = 5
    ) -> List[ContextPassage]:
        """Queries database using query variants, resolving duplicates by keeping highest score."""
        registry: Dict[str, ContextPassage] = {}

        for vec in query_vectors:
            hits = self.search_dense(vec, limit=limit * 2)
            for hit in hits:
                key = f"{hit.doc_id}:{hit.page_num}:{hit.seq_index}"
                if key not in registry or hit.relevance_score > registry[key].relevance_score:
                    registry[key] = hit

        sorted_passages = sorted(registry.values(), key=lambda p: p.relevance_score, reverse=True)
        return sorted_passages[:limit]

    def search_rerank(
        self,
        query: str,
        vector: List[float],
        limit: int = 5,
        candidate_count: int = 10
    ) -> List[ContextPassage]:
        """Two-step search: Dense retrieval followed by Cross-Encoder scoring."""
        dense_results = self.search_dense(vector, limit=candidate_count)
        if not dense_results:
            return []

        try:
            global _cross_encoder_registry
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            if model_name not in _cross_encoder_registry:
                from sentence_transformers import CrossEncoder
                _cross_encoder_registry[model_name] = CrossEncoder(model_name)
            
            reranker = _cross_encoder_registry[model_name]
            pairs = [[query, passage.content] for passage in dense_results]
            scores = reranker.predict(pairs)

            for idx, passage in enumerate(dense_results):
                passage.relevance_score = float(scores[idx])

            reranked_list = sorted(dense_results, key=lambda x: x.relevance_score, reverse=True)
            return reranked_list[:limit]
        except Exception:
            return dense_results[:limit]

    def search(
        self,
        query: str,
        query_vector: List[float],
        strategy: str = "dense",
        top_k: int = 5,
        variants_vectors: Optional[List[List[float]]] = None
    ) -> RAGSearchResult:
        """Unified entry point to perform search across all supported strategies."""
        strategy = strategy.lower().strip()
        
        if strategy == "mmr":
            passages = self.search_mmr(query_vector, limit=top_k)
        elif strategy == "hybrid":
            passages = self.search_hybrid(query, query_vector, limit=top_k)
        elif strategy == "multi_query" and variants_vectors:
            passages = self.search_multi_query(variants_vectors, limit=top_k)
        elif strategy == "reranker":
            passages = self.search_rerank(query, query_vector, limit=top_k)
        else:
            passages = self.search_dense(query_vector, limit=top_k)

        contexts = [p.content for p in passages]
        sources = list({f"{p.doc_id} (Page {p.page_num})" for p in passages})

        return RAGSearchResult(
            contexts=contexts,
            sources=sources,
            passages=passages,
            strategy=strategy
        )
