"""FAISS storage with stable node IDs and reversible vector mutations."""
from contextlib import contextmanager
from threading import RLock
from typing import Any, Iterator, List, Optional, Union, Literal

import faiss
import numpy as np
from pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.vector_stores.faiss import FaissMapVectorStore, FaissVectorStore

from .base import VectorStoreBase


class TransactionalFaissMapVectorStore(FaissMapVectorStore):
    """Keep FAISS IDs unique after deletions and undo unsuccessful batches.

    The upstream adapter allocates IDs from ``ntotal``. That value decreases
    when a node is deleted, so an insertion can reuse a surviving vector's ID.
    This adapter allocates monotonically and keeps an undo journal containing
    only the vectors affected by a mutation. It does not clone the index.

    Transactions also cover a caller's subsequent docstore/index mutations.
    They serialize access to native FAISS using the same lock as retrieval.
    """

    _next_id: int = PrivateAttr(default=0)
    _journal: Optional[List[Any]] = PrivateAttr(default=None)
    _mutation_lock: Any = PrivateAttr(default_factory=RLock)

    def __init__(self, faiss_index):
        """Use native IVF IDs directly, and an IDMap for flat indexes.

        IVF already supports explicit IDs. Wrapping it in an IndexIDMap makes
        deletion compact a second ID table independently of IVF's stored IDs.
        A native hash direct map also permits reconstructing touched vectors
        for the rollback journal without copying the whole vector corpus.
        """
        if isinstance(faiss_index, faiss.IndexIVFFlat):
            faiss_index.set_direct_map_type(faiss.DirectMap.Hashtable)
            FaissVectorStore.__init__(self, faiss_index=faiss_index)
            self._node_id_to_faiss_id_map = {}
            self._faiss_id_to_node_id_map = {}
        else:
            super().__init__(faiss_index=faiss_index)

    @property
    def mutation_lock(self):
        """Return the lock protecting vectors and their associated node catalog."""
        return self._mutation_lock

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit vector changes together, or undo them in reverse order.

        Nested calls participate in the outer transaction. A rollback failure
        is raised explicitly: callers must not treat that backend as healthy.
        The counter deliberately does not rewind after rollback, so failed
        attempts cannot cause a later numeric-ID collision.
        """
        with self._mutation_lock:
            if self._journal is not None:
                yield
                return
            self._journal = []
            try:
                yield
            except BaseException as original:
                rollback_errors = []
                for undo in reversed(self._journal):
                    try:
                        undo()
                    except Exception as error:
                        rollback_errors.append(error)
                if rollback_errors:
                    raise RuntimeError(
                        "FAISS rollback failed; recreate the index from its saved corpus"
                    ) from original
                raise
            finally:
                self._journal = None

    def _prepare_batch(self, nodes):
        """Validate the entire batch before touching native state."""
        unique = {}
        for node in nodes:
            if not isinstance(node, BaseNode):
                raise TypeError("FAISS nodes must be LlamaIndex BaseNode instances")
            if not isinstance(node.node_id, str) or not node.node_id:
                raise ValueError("FAISS node IDs must be nonempty strings")
            unique[node.node_id] = node
        if not unique:
            return [], np.empty((0, self.client.d), dtype=np.float32)
        try:
            vectors = np.asarray(
                [node.get_embedding() for node in unique.values()], dtype=np.float32
            )
        except (TypeError, ValueError) as error:
            raise ValueError("FAISS embeddings must be equally sized numeric vectors") from error
        if vectors.shape != (len(unique), self.client.d):
            raise ValueError(f"FAISS embeddings must have dimension {self.client.d}")
        if not np.isfinite(vectors).all():
            raise ValueError("FAISS embeddings must contain only finite numbers")
        return list(unique), np.ascontiguousarray(vectors)

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Atomically insert a batch, replacing duplicate logical node IDs."""
        node_ids, vectors = self._prepare_batch(nodes)
        if not node_ids:
            return []
        with self.transaction():
            self.delete_nodes(node_ids)
            # Also handle maps loaded by the upstream persistence constructor.
            if self._next_id == 0 and self._faiss_id_to_node_id_map:
                self._next_id = max(self._faiss_id_to_node_id_map) + 1
            numeric_ids = np.arange(
                self._next_id, self._next_id + len(node_ids), dtype=np.int64
            )
            self._next_id += len(node_ids)

            def undo_add():
                # Record before add_with_ids: even a partially applied native
                # insertion is removable because these IDs were never live.
                self.client.remove_ids(numeric_ids)
                for node_id, numeric_id in zip(node_ids, numeric_ids):
                    if self._node_id_to_faiss_id_map.get(node_id) == int(numeric_id):
                        self._node_id_to_faiss_id_map.pop(node_id, None)
                    self._faiss_id_to_node_id_map.pop(int(numeric_id), None)

            self._journal.append(undo_add)
            self.client.add_with_ids(vectors, numeric_ids)
            for node_id, numeric_id in zip(node_ids, numeric_ids):
                self._node_id_to_faiss_id_map[node_id] = int(numeric_id)
                self._faiss_id_to_node_id_map[int(numeric_id)] = node_id
        return node_ids

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete logical IDs while journaling their native vectors for undo."""
        if filters is not None:
            raise NotImplementedError("FAISS metadata filters are not supported")
        if node_ids is None:
            raise ValueError("node_ids must be provided to delete nodes")
        with self.transaction():
            existing = {
                node_id: self._node_id_to_faiss_id_map[node_id]
                for node_id in dict.fromkeys(node_ids)
                if node_id in self._node_id_to_faiss_id_map
            }
            if not existing:
                return
            numeric_ids = np.asarray(list(existing.values()), dtype=np.int64)
            vectors = np.asarray(
                [self.client.reconstruct(int(identifier)) for identifier in numeric_ids],
                dtype=np.float32,
            )

            def undo_delete():
                # Remove first so rollback is safe if native deletion failed
                # after removing only some of the requested IDs.
                self.client.remove_ids(numeric_ids)
                self.client.add_with_ids(vectors, numeric_ids)
                self._node_id_to_faiss_id_map.update(existing)
                self._faiss_id_to_node_id_map.update(
                    {identifier: node_id for node_id, identifier in existing.items()}
                )

            self._journal.append(undo_delete)
            self.client.remove_ids(numeric_ids)
            for node_id, numeric_id in existing.items():
                self._node_id_to_faiss_id_map.pop(node_id)
                self._faiss_id_to_node_id_map.pop(numeric_id)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a node using the adapter's node-ID contract."""
        self.delete_nodes([ref_doc_id], **delete_kwargs)

    def query(self, query, **kwargs):
        """Serialize native search with mutations."""
        with self._mutation_lock:
            return super().query(query, **kwargs)


class FaissVectorStoreWrapper(VectorStoreBase):
    """Create independent FAISS stores from a reusable backend configuration."""

    def __init__(
        self,
        dimensions: int = 1536,
        metrics: Union[Literal["flat_l2", "ivf_flat"]] = "flat_l2",
        **kwargs,
    ):
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("FAISS dimensions must be a positive integer")
        self.dimensions = dimensions
        self.metrics = metrics
        self.faiss_index = self._create_index()
        native = self.faiss_index if isinstance(self.faiss_index, faiss.IndexIVFFlat) else faiss.IndexIDMap2(self.faiss_index)
        self.vector_store = TransactionalFaissMapVectorStore(faiss_index=native)

    def _create_index(self) -> faiss.Index:
        """Build the configured native index."""
        if self.metrics == "flat_l2":
            return faiss.IndexFlatL2(self.dimensions)
        if self.metrics == "ivf_flat":
            quantizer = faiss.IndexFlatL2(self.dimensions)
            return faiss.IndexIVFFlat(quantizer, self.dimensions, 100)
        raise ValueError(f"Unsupported FAISS index type: {self.metrics}")

    def create_empty(self) -> "FaissVectorStoreWrapper":
        """Create a corpus-owned store without copying any existing vectors."""
        empty = type(self)(dimensions=self.dimensions, metrics=self.metrics)
        if isinstance(self.faiss_index, faiss.IndexIVFFlat) and self.faiss_index.is_trained:
            # Retain the trained centroids and query settings, without cloning
            # the template's vector data. Only the small quantizer is copied.
            source = self.faiss_index
            native = faiss.IndexIVFFlat(
                faiss.clone_index(source.quantizer), self.dimensions,
                source.nlist, source.metric_type,
            )
            native.is_trained = True
            native.nprobe = source.nprobe
            empty.faiss_index = native
            empty.vector_store = TransactionalFaissMapVectorStore(faiss_index=native)
        return empty

    def get_vector_store(self) -> TransactionalFaissMapVectorStore:
        """Return the LlamaIndex-compatible backend."""
        return self.vector_store

    async def aload(self, node: BaseNode) -> None:
        """Load a node once without replacing an existing logical ID."""
        if not isinstance(node, BaseNode):
            raise ValueError("node must be a LlamaIndex BaseNode")
        with self.vector_store.mutation_lock:
            if node.node_id not in self.vector_store._node_id_to_faiss_id_map:
                self.vector_store.add([node])
