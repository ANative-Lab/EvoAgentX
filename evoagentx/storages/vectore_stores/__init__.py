from typing import Dict, Any

from llama_index.core.vector_stores.types import VectorStore

from .base import VectorStoreType, VectorStoreBase
from .faiss import FaissVectorStoreWrapper
from evoagentx.core.logging import logger

__all__ = ['VectorStoreBase', 'FaissVectorStoreWrapper', 'VectorStoreFactory']

class VectorStoreFactory:
    """Factory for creating vector stores."""

    def create(self, store_type: str, store_config: Dict[str, Any] = None) -> VectorStore:
        store_config = store_config or {}
        if store_type == VectorStoreType.FAISS:
            dimensions = store_config.get("dimensions")
            if not dimensions or not isinstance(dimensions, int):
                raise ValueError("FAISS requires a valid dimension")
            vector_store = FaissVectorStoreWrapper(**store_config)
        # elif store_type == VectorStoreType.QDRANT:
        #     qdrant_url = store_config.get("qdrant_url")
        #     qdrant_api_key = store_config.get("qdrant_api_key")
        #     collection_name = store_config.get("qdrant_collection_name", "default_collection")
        #     if not qdrant_url:
        #         raise ValueError("Qdrant requires a valid URL")
        #     client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        #     vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        elif store_type == VectorStoreType.MILVUS:
            dimensions = store_config.get("dimensions")
            if not dimensions or not isinstance(dimensions, int):
                raise ValueError("Milvus requires a valid dimension")
            from .milvus import MilvusVectorStoreWrapper

            vector_store = MilvusVectorStoreWrapper(**store_config)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        logger.info(f"Created vector store: {store_type}")
        return vector_store
