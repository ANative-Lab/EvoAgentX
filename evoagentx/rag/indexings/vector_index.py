"""Corpus-owned vector indexing with atomic updates to all node catalogs."""
from contextlib import contextmanager
from typing import List, Dict, Any, Union, Optional, Sequence

from llama_index.core.schema import BaseNode, ImageNode, NodeWithScore
from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.embeddings import BaseEmbedding

from .base import BaseIndexWrapper, IndexType
from ._transaction import JournaledKVStore
from evoagentx.rag.schema import Chunk, ImageChunk
from evoagentx.storages.base import StorageHandler


class VectorIndexing(BaseIndexWrapper):
    """Own a FAISS namespace and keep its LlamaIndex catalogs consistent.

    ``storage_handler.vector_store`` supplies configuration, not shared corpus
    data. Embeddings are prepared before mutation. Native changes are journaled
    and LlamaIndex metadata is restored if any stage of a write fails.
    """

    def __init__(
        self,
        embed_model: BaseEmbedding,
        storage_handler: StorageHandler,
        index_config: Dict[str, Any] = None,
    ):
        super().__init__()
        self.index_type = IndexType.VECTOR
        self.embed_model = embed_model
        self.storage_handler = storage_handler
        self.index_config = index_config or {}
        self.id_to_node = {}
        self._create_storage_context()
        self.index = VectorStoreIndex(
            nodes=[],
            embed_model=self.embed_model,
            storage_context=self.storage_context,
            show_progress=self.index_config.get("show_progress", False),
        )
        # The same lock must cover search and docstore resolution. Keeping the
        # LlamaIndex object stable preserves callers of get_index().
        self.index._evoagentx_mutation_lock = self.mutation_lock

    def _create_storage_context(self):
        """Create an independent backend using the supported FAISS wrapper."""
        template = self.storage_handler.vector_store
        if template is None:
            raise ValueError("VectorIndexing requires a configured vector backend")
        if not callable(getattr(template, "create_empty", None)):
            raise TypeError("Vector backend must support isolated create_empty() stores")
        self.vector_store = template.create_empty()
        backend = self.vector_store.get_vector_store()
        self.mutation_lock = backend.mutation_lock
        self._catalog = JournaledKVStore()
        self.storage_context = StorageContext.from_defaults(
            vector_store=backend,
            docstore=SimpleDocumentStore(simple_kvstore=self._catalog),
            index_store=SimpleIndexStore(simple_kvstore=self._catalog),
        )
        return self.storage_context

    def get_index(self) -> VectorStoreIndex:
        """Return the stable LlamaIndex object backing this corpus."""
        return self.index

    @contextmanager
    def _write_transaction(self, node_ids):
        """Journal changed native vectors, catalog records and index entries.

        The undo state is proportional to the batch, not the corpus. LlamaIndex
        still serializes its index-ID mapping on each write, as it did before;
        no additional full corpus or native embedding copy is introduced here.
        """
        with self.mutation_lock:
            mapping = self.index.index_struct.nodes_dict
            previous = {identifier: mapping.get(identifier) for identifier in node_ids}
            try:
                with self.vector_store.get_vector_store().transaction():
                    with self._catalog.transaction():
                        yield
            except BaseException:
                for identifier, old_value in previous.items():
                    if old_value is None:
                        mapping.pop(identifier, None)
                    else:
                        mapping[identifier] = old_value
                raise

    @staticmethod
    def _prepare_nodes(nodes):
        """Copy inputs and keep the final value of duplicate IDs in a batch."""
        prepared = {}
        for node in nodes:
            if isinstance(node, NodeWithScore):
                node = node.node
            if isinstance(node, (Chunk, ImageChunk)):
                node = node.to_llama_node()
            if not isinstance(node, BaseNode):
                raise TypeError("Expected Chunk, ImageChunk, BaseNode or NodeWithScore")
            if not node.node_id:
                raise ValueError("Node IDs must be nonempty")
            prepared[node.node_id] = node.model_copy(deep=True)
        return list(prepared.values())

    def insert_nodes(self, nodes: List[Union[Chunk, BaseNode]]) -> Sequence[str]:
        """Insert or replace a batch without losing the prior state on failure.

        The final occurrence of a duplicate ID wins. Callers retain ownership
        of their input objects; neither metadata nor embeddings are mutated.
        Errors propagate to the engine, which retains its historical ``[]``
        failure result. Direct wrapper callers receive the original exception.
        """
        prepared = self._prepare_nodes(nodes)
        if not prepared:
            return []
        # No live node has been deleted when embedding raises.
        embedded = self.index._get_node_with_embedding(nodes=prepared)
        # Validate before entering the native transaction, including embeddings
        # supplied by callers rather than generated by the embedding model.
        self.vector_store.get_vector_store()._prepare_batch(embedded)
        identifiers = [node.node_id for node in embedded]
        with self._write_transaction(identifiers):
            existing = [identifier for identifier in identifiers if identifier in self.id_to_node]
            if existing:
                self.index.delete_nodes(existing, delete_from_docstore=True)
            self.index.insert_nodes(embedded)
            self.id_to_node.update({node.node_id: node for node in embedded})
        return identifiers

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete selected nodes and their docstore/index references atomically.

        Explicit IDs take precedence over metadata filters, preserving the
        existing selection contract. Unknown IDs are harmless.
        """
        with self.mutation_lock:
            if node_ids is not None:
                selected = [identifier for identifier in dict.fromkeys(node_ids) if identifier in self.id_to_node]
            elif metadata_filters:
                selected = [
                    identifier for identifier, node in self.id_to_node.items()
                    if all(node.metadata.get(key) == value for key, value in metadata_filters.items())
                ]
            else:
                return
            if not selected:
                return
            with self._write_transaction(selected):
                self.index.delete_nodes(selected, delete_from_docstore=True)
                for identifier in selected:
                    self.id_to_node.pop(identifier)

    async def aload(self, nodes: List[Union[Chunk, BaseNode]]) -> Sequence[str]:
        """Load a batch using the same transactional path as insertion."""
        return self.insert_nodes(nodes)

    def load(self, nodes: List[Union[Chunk, BaseNode]]) -> Sequence[str]:
        """Load without starting a nested event loop."""
        return self.insert_nodes(nodes)

    def clear(self) -> None:
        """Clear this corpus without modifying another index's backend."""
        self.delete_nodes(node_ids=list(self.id_to_node))

    def snapshot_nodes(self) -> List[BaseNode]:
        """Return detached nodes from one consistent state for persistence."""
        with self.mutation_lock:
            return [node.model_copy(deep=True) for node in self.id_to_node.values()]

    async def _get(self, node_id: str) -> Optional[Union[Chunk, ImageChunk]]:
        """Read one detached chunk from the committed cache."""
        with self.mutation_lock:
            node = self.id_to_node.get(node_id)
            if node is None:
                return None
            node = node.model_copy(deep=True)
        chunk_class = ImageChunk if isinstance(node, ImageNode) else Chunk
        return chunk_class.from_llama_node(node)

    async def get(self, node_ids: Sequence[str]) -> List[Union[Chunk, ImageChunk]]:
        """Read selected chunks in requested order from one committed state."""
        with self.mutation_lock:
            nodes = [self.id_to_node[identifier].model_copy(deep=True) for identifier in node_ids if identifier in self.id_to_node]
        return [
            (ImageChunk if isinstance(node, ImageNode) else Chunk).from_llama_node(node)
            for node in nodes
        ]
