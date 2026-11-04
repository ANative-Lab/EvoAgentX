"""Vector retrieval with request-local parameters and consistent node lookup."""
from contextlib import nullcontext

from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import VectorIndexRetriever

from .base import BaseRetrieverWrapper
from evoagentx.rag.schema import Query, RagResult, Corpus


class VectorRetriever(BaseRetrieverWrapper):
    """Retrieve from one corpus without sharing mutable query parameters."""

    def __init__(self, index: BaseIndex, top_k: int = 5, chunk_class=None):
        super().__init__()
        self.index = index
        self.top_k = top_k
        self.chunk_class = chunk_class
        self._mutation_lock = getattr(index, "_evoagentx_mutation_lock", None)
        # Retain the public compatibility attribute, but never mutate/reuse it
        # for concurrent requests with different top_k values.
        self.retriever = self._new_retriever(self.top_k)

    def _new_retriever(self, top_k):
        """Construct a retriever whose settings belong to a single request."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return VectorIndexRetriever(index=self.index, similarity_top_k=top_k)

    def _lock(self):
        """Use the corpus lock when this index supports transactional writes."""
        return self._mutation_lock if self._mutation_lock is not None else nullcontext()

    def _result(self, query, nodes):
        """Convert detached retrieved nodes without changing stored metadata."""
        if self.chunk_class is None:
            raise ValueError("chunk_class not set - RAGEngine must pass the chunk class")
        chunks = []
        scores = []
        for scored in nodes or []:
            node = scored.node.model_copy(deep=True)
            chunk = self.chunk_class.from_llama_node(node)
            score = scored.score if scored.score is not None else 0.0
            chunk.metadata.similarity_score = score
            chunks.append(chunk)
            scores.append(score)
        return RagResult(
            corpus=Corpus(chunks=chunks),
            scores=scores,
            metadata={"query": query.query_str, "retriever": "vector"},
        )

    async def aretrieve(self, query: Query) -> RagResult:
        """Embed asynchronously, then search and resolve nodes under one lock.

        No lock is held across an await. This allows another task to finish
        ingesting while a slow embedding request is in flight; after embedding,
        the query sees one complete committed state of vectors and docstore.
        """
        bundle = query.to_QueryBundle()
        if bundle.embedding is None and self.index.vector_store.is_embedding_query:
            bundle.embedding = await self.index._embed_model.aget_agg_embedding_from_queries(
                bundle.embedding_strs
            )
        with self._lock():
            retriever = self._new_retriever(query.top_k if query.top_k is not None else self.top_k)
            nodes = retriever.retrieve(bundle)
            return self._result(query, nodes)

    def retrieve(self, query: Query) -> RagResult:
        """Run a synchronous request with the same isolation guarantees."""
        bundle = query.to_QueryBundle()
        if bundle.embedding is None and self.index.vector_store.is_embedding_query:
            bundle.embedding = self.index._embed_model.get_agg_embedding_from_queries(
                bundle.embedding_strs
            )
        with self._lock():
            retriever = self._new_retriever(query.top_k if query.top_k is not None else self.top_k)
            return self._result(query, retriever.retrieve(bundle))

    def get_retriever(self) -> VectorIndexRetriever:
        """Return the compatibility retriever with the configured defaults."""
        with self._lock():
            return self._new_retriever(self.top_k)
