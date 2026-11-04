"""Milvus vector store integration tests."""

# ruff: noqa: E402

from pathlib import Path
import importlib.util
import sys
import types
from unittest.mock import patch

import pytest
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import MetadataFilters, VectorStoreQuery

pytest.importorskip("pymilvus")


def install_optional_embedding_stubs() -> None:
    """Stub unused optional embedding packages when running focused tests."""
    if importlib.util.find_spec("sentence_transformers") is None:
        sentence_transformers = types.ModuleType("sentence_transformers")

        class SentenceTransformer:
            pass

        sentence_transformers.SentenceTransformer = SentenceTransformer
        sys.modules["sentence_transformers"] = sentence_transformers

    if importlib.util.find_spec("voyageai") is None:
        voyageai = types.ModuleType("voyageai")

        class AsyncClient:
            pass

        voyageai.AsyncClient = AsyncClient
        sys.modules["voyageai"] = voyageai


install_optional_embedding_stubs()

from evoagentx.rag.indexings.base import IndexType
from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.rag_config import (
    ChunkerConfig,
    EmbeddingConfig,
    IndexConfig,
    RAGConfig,
    ReaderConfig,
    RetrievalConfig,
)
from evoagentx.rag.schema import Corpus, Query, TextChunk
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import DBConfig, StoreConfig, VectorStoreConfig
from evoagentx.storages.vectore_stores import VectorStoreFactory


class DeterministicEmbedding(BaseEmbedding):
    """Small deterministic embedding model for vector-store integration tests."""

    def __init__(self) -> None:
        super().__init__(model_name="deterministic-test", embed_batch_size=10)

    def _embed(self, text: str) -> list[float]:
        text = text.lower()
        if "milvus" in text:
            return [1.0, 0.0]
        if "faiss" in text:
            return [0.0, 1.0]
        return [0.5, 0.5]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    @property
    def dimensions(self) -> int:
        return 2


class DeterministicEmbeddingWrapper:
    """Embedding wrapper that avoids external API calls in tests."""

    def __init__(self) -> None:
        self.model_name = "deterministic-test"
        self._embedding_model = DeterministicEmbedding()

    def get_embedding_model(self) -> BaseEmbedding:
        return self._embedding_model

    def validate_model(self, provider: str, model_name: str) -> bool:
        return model_name == self.model_name

    @property
    def dimensions(self) -> int:
        return 2


def milvus_config(tmp_path: Path, collection_name: str = "evoagentx_test") -> VectorStoreConfig:
    return VectorStoreConfig(
        vector_name="milvus",
        dimensions=2,
        milvus_uri=str(tmp_path / "milvus.db"),
        milvus_collection_name=collection_name,
        milvus_metric_type="IP",
        milvus_consistency_level="Strong",
        milvus_overwrite=True,
    )


def test_milvus_vector_store_add_query_filter_and_delete(tmp_path: Path) -> None:
    wrapper = VectorStoreFactory().create(
        store_type="milvus",
        store_config=milvus_config(tmp_path).model_dump(),
    )
    vector_store = wrapper.get_vector_store()

    vector_store.add(
        [
            TextNode(
                id_="milvus-node",
                text="Milvus stores vectors.",
                embedding=[1.0, 0.0],
                metadata={"topic": "milvus"},
            ),
            TextNode(
                id_="faiss-node",
                text="FAISS stores vectors.",
                embedding=[0.0, 1.0],
                metadata={"topic": "faiss"},
            ),
        ]
    )

    result = vector_store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0], similarity_top_k=2)
    )
    assert result.ids == ["milvus-node", "faiss-node"]
    assert result.similarities[0] > result.similarities[1]

    filtered = vector_store.query(
        VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts([{"key": "topic", "value": "milvus"}]),
        )
    )
    assert filtered.ids == ["milvus-node"]

    vector_store.delete_nodes(["milvus-node"])
    after_delete = vector_store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0], similarity_top_k=2)
    )
    assert after_delete.ids == ["faiss-node"]


def test_rag_engine_uses_milvus_vector_store(tmp_path: Path) -> None:
    store_config = StoreConfig(
        dbConfig=DBConfig(db_name="sqlite", path=str(tmp_path / "storage.db")),
        vectorConfig=milvus_config(tmp_path, collection_name="evoagentx_rag_test"),
        graphConfig=None,
        path=str(tmp_path / "index_cache"),
    )
    storage_handler = StorageHandler(storageConfig=store_config)
    rag_config = RAGConfig(
        reader=ReaderConfig(),
        chunker=ChunkerConfig(strategy="simple", chunk_size=512, chunk_overlap=0),
        embedding=EmbeddingConfig(
            provider="openai",
            model_name="deterministic-test",
            api_key="dummy-key",
        ),
        index=IndexConfig(index_type="vector"),
        retrieval=RetrievalConfig(
            retrivel_type="vector",
            postprocessor_type="simple",
            top_k=1,
            similarity_cutoff=None,
        ),
    )

    with patch(
        "evoagentx.rag.rag.EmbeddingFactory.create",
        return_value=DeterministicEmbeddingWrapper(),
    ):
        rag_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)
        corpus = Corpus(
            corpus_id="test_corpus",
            chunks=[
                TextChunk(text="Milvus provides scalable vector search.", chunk_id="milvus"),
                TextChunk(text="FAISS provides local vector search.", chunk_id="faiss"),
            ],
        )
        rag_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id="test_corpus")

        result = rag_engine.query(
            Query(query_str="Milvus vector database", top_k=1, similarity_cutoff=None),
            corpus_id="test_corpus",
        )

    assert len(result.corpus.chunks) == 1
    assert result.corpus.chunks[0].chunk_id == "milvus"
