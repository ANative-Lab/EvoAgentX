import asyncio
from uuid import uuid4
from datetime import datetime
from threading import RLock
from typing import List, Union, Optional, Sequence, Dict, Any, Tuple
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode, ImageNode

from .rag_config import RAGConfig
from .readers import LLamaIndexReader, MultimodalReader
from .indexings import IndexFactory, BaseIndexWrapper
from .chunkers import ChunkFactory
from .embeddings import EmbeddingFactory
from .retrievers import RetrieverFactory, BaseRetrieverWrapper
from .postprocessors import PostprocessorFactory
from .indexings.base import IndexType
from .retrievers.base import RetrieverType
from .schema import Corpus, ChunkMetadata, IndexMetadata, Query, RagResult, ImageChunk, TextChunk
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.schema import IndexStore, expand_index_record
from .snapshots import (
    IndexSnapshot, load_file_snapshots,
    save_file_snapshot, snapshot_from_record,
)
from evoagentx.models.base_model import BaseLLM
from evoagentx.core.logging import logger


class RAGEngine:
    def __init__(self, config: RAGConfig, storage_handler: StorageHandler, llm: Optional[BaseLLM] = None):
        self.config = config
        self.storage_handler = storage_handler
        self._registry_lock = RLock()
        self.embedding_factory = EmbeddingFactory()
        self.index_factory = IndexFactory()
        self.chunk_factory = ChunkFactory()
        self.retriever_factory = RetrieverFactory()
        self.postprocessor_factory = PostprocessorFactory()
        self.llm = llm  # LLM for entity extractor
        
        # Set chunk class based on modality config
        logger.info(f"RAGEngine modality config: {self.config.modality}")
        
        if self.config.modality == "multimodal":
            self.chunk_class = ImageChunk
        else:
            self.chunk_class = TextChunk

        # Initialize reader based on modality
        if self.config.modality == "multimodal":
            self.reader = MultimodalReader(
                recursive=self.config.reader.recursive,
                exclude_hidden=self.config.reader.exclude_hidden,
                num_files_limits=self.config.reader.num_files_limit,
                errors=self.config.reader.errors
            )
        else:
            self.reader = LLamaIndexReader(
                recursive=self.config.reader.recursive,
                exclude_hidden=self.config.reader.exclude_hidden,
                num_workers=self.config.num_workers,
                num_files_limits=self.config.reader.num_files_limit,
                custom_metadata_function=self.config.reader.custom_metadata_function,
                extern_file_extractor=self.config.reader.extern_file_extractor,
                errors=self.config.reader.errors,
                encoding=self.config.reader.encoding
            )

        # Initialize embedding model. 
        self.embed_model = self.embedding_factory.create(
            provider=self.config.embedding.provider,
            model_config=self.config.embedding.model_dump(exclude_unset=True),
        )

        # Dynamic Check the dimensions in StorageHandler
        if (self.storage_handler.vector_store is not None) and (self.embed_model.dimensions is not None):
            if self.storage_handler.storageConfig.vectorConfig.dimensions != self.embed_model.dimensions:
                logger.warning("The dimensions in vector_store is not equal with embed_model. Reiniliaze vector_store.")
                self.storage_handler.storageConfig.vectorConfig.dimensions = self.embed_model.dimensions
                self.storage_handler._init_vector_store()

        # Initialize chunker (skip for multimodal)
        if self.config.modality == "multimodal":
            self.chunker = None  # No chunking for images
        else:
            self.chunker = self.chunk_factory.create(
                strategy=self.config.chunker.strategy,
                embed_model=self.embed_model.get_embedding_model(),
                chunker_config={
                    "chunk_size": self.config.chunker.chunk_size,
                    "chunk_overlap": self.config.chunker.chunk_overlap,
                    "max_chunks": self.config.chunker.max_chunks
                }
            )

        # Initialize indices and retrievers
        self.indices: Dict[str, Dict[str, BaseIndexWrapper]] = {}  # Nested: {corpus_id: {index_type: index}}
        self.retrievers: Dict[str, Dict[str, BaseRetrieverWrapper]] = {}  # Nested: {corpus_id: {index_type: retriever}}

    def read(self, file_paths: Union[Sequence[str], str], 
             exclude_files: Optional[Union[str, List, Tuple, Sequence]] = None,
             filter_file_by_suffix: Optional[Union[str, List, Tuple, Sequence]] = None,
             merge_by_file: bool = False,
             show_progress: bool = False,
             corpus_id: str = None) -> Corpus:
        """Load and chunk documents from files.

        Reads files from specified paths, processes them into documents, and chunks them into a Corpus.

        Args:
            file_paths (Union[Sequence[str], str]): Path(s) to files or directories.
            exclude_files (Optional[Union[str, List, Tuple, Sequence]]): Files to exclude.
            filter_file_by_suffix (Optional[Union[str, List, Tuple, Sequence]]): Filter files by suffix (e.g., '.pdf').
            merge_by_file (bool): Merge documents by file.
            show_progress (bool): Show loading progress.
            corpus_id (Optional[str]): Identifier for the corpus. Defaults to a UUID if None.

        Returns:
            Corpus: The chunked corpus containing processed document chunks.

        Raises:
            Exception: If document reading or chunking fails.
        """
        try:
            corpus_id = corpus_id or str(uuid4())
            documents = self.reader.load(
                file_paths=file_paths,
                exclude_files=exclude_files,
                filter_file_by_suffix=filter_file_by_suffix,
                merge_by_file=merge_by_file,
                show_progress=show_progress
            )
            if self.config.modality == "multimodal":
                # No chunking - convert ImageDocuments directly to ImageChunks
                image_chunks = []
                for doc in documents:
                    # Get image path from document - try multiple possible attributes
                    image_path = getattr(doc, 'image_path', None) or doc.metadata.get('file_path')
                    image_mimetype = getattr(doc, 'image_mimetype', None)
                    
                    image_chunk = self.chunk_class(
                        image_path=image_path,
                        image_mimetype=image_mimetype,
                        chunk_id=doc.metadata.get('file_name', f'img_{len(image_chunks)}'),
                        metadata=ChunkMetadata(
                            doc_id=doc.metadata.get('file_name', f'doc_{len(image_chunks)}'),
                            corpus_id=corpus_id,
                            **doc.metadata
                        )
                    )
                    image_chunks.append(image_chunk)
                corpus = Corpus(chunks=image_chunks, corpus_id=corpus_id)
                logger.info(f"Read {len(documents)} multimodal documents (no chunking) for corpus {corpus_id}")
            else:
                corpus = self.chunker.chunk(documents)
                corpus.corpus_id = corpus_id
                logger.info(f"Read {len(documents)} documents and created {len(corpus.chunks)} chunks for corpus {corpus_id}")
            return corpus
        except Exception as e:
            logger.error(f"Failed to read documents for corpus {corpus_id}: {str(e)}")
            raise

    def _new_index(self, index_type: str) -> BaseIndexWrapper:
        """Construct an unregistered index using the engine's embedding model."""
        return self.index_factory.create(
            index_type=index_type,
            embed_model=self.embed_model.get_embedding_model(),
            storage_handler=self.storage_handler,
            index_config=self.config.index.model_dump(exclude_unset=True) if self.config.index else {},
            llm=self.llm,
        )

    def _new_retriever(self, index_type: str, index: BaseIndexWrapper):
        """Bind a retriever to the same corpus-owned index and node catalog."""
        retriever_type = RetrieverType.GRAPH if index_type == IndexType.GRAPH else RetrieverType.VECTOR
        return self.retriever_factory.create(
            retriever_type=retriever_type,
            llm=self.llm,
            index=index.get_index(),
            graph_store=index.get_index().storage_context.graph_store,
            embed_model=self.embed_model.get_embedding_model(),
            query=Query(query_str="", top_k=self.config.retrieval.top_k),
            storage_handler=self.storage_handler,
            chunk_class=self.chunk_class,
        )

    @staticmethod
    def _owned_nodes(nodes, corpus_id, index_type):
        """Copy a caller's nodes before assigning index ownership metadata."""
        source = nodes.to_llama_nodes() if isinstance(nodes, Corpus) else nodes
        owned = []
        for node in source:
            if isinstance(node, NodeWithScore):
                node = node.node
            if isinstance(node, (TextChunk, ImageChunk)):
                node = node.to_llama_node()
            if not isinstance(node, BaseNode):
                raise TypeError("Expected Corpus, chunks, BaseNode or NodeWithScore inputs")
            node = node.model_copy(deep=True)
            node.metadata.update(corpus_id=corpus_id, index_type=index_type)
            owned.append(node)
        return owned

    def add(self, index_type: str, nodes: Union[Corpus, List[NodeWithScore], List[TextNode], List[ImageNode]],
            corpus_id: str = None) -> Sequence[str]:
        """Insert a batch and publish a new corpus only after successful ingestion.

        Uses the Corpus identity when no explicit corpus_id is supplied. Input
        nodes remain caller-owned. As before, ingestion failures return an empty
        list; transactional vector writes retain the previous committed state.
        """
        try:
            if corpus_id is None:
                corpus_id = str(nodes.corpus_id) if isinstance(nodes, Corpus) else str(uuid4())
            if not isinstance(corpus_id, str) or not corpus_id:
                raise ValueError("corpus_id must be a nonempty string")
            owned = self._owned_nodes(nodes, corpus_id, index_type)
            if not owned:
                return []
            with self._registry_lock:
                index = self.indices.get(corpus_id, {}).get(index_type)
                new = index is None
                if new:
                    index = self._new_index(index_type)
                    retriever = self._new_retriever(index_type, index)
                identifiers = index.insert_nodes(owned)
                if new:
                    self.indices.setdefault(corpus_id, {})[index_type] = index
                    self.retrievers.setdefault(corpus_id, {})[index_type] = retriever
            return identifiers
        except Exception as error:
            logger.error(f"Failed to add nodes to {index_type} in corpus {corpus_id}: {error}")
            return []

    def delete(self, corpus_id: str, index_type: Optional[str] = None, 
               node_ids: Optional[Union[str, List[str]]] = None, 
               metadata_filters: Optional[Dict[str, Any]] = None) -> None:
        """Delete nodes or an entire index from a corpus.

        Removes specific nodes by ID or metadata filters, or deletes the entire index if no filters are provided.

        Args:
            corpus_id (str): Identifier for the corpus.
            index_type (Optional[IndexType]): Specific index type to delete from. If None, affects all indices.
            node_ids (Union[str, Optional[List[str]]]): List of node IDs to delete.
            metadata_filters (Optional[Dict[str, Any]]): Metadata filters to select nodes for deletion.

        Raises:
            Exception: If deletion fails.
        """
        try:
            with self._registry_lock:
                if corpus_id not in self.indices:
                    logger.warning(f"No indices found for corpus {corpus_id}")
                    return

                target_indices = [index_type] if index_type is not None else self.indices[corpus_id].keys()
                for idx_type in list(target_indices):  # Use list to avoid runtime modification issues
                    if idx_type not in self.indices[corpus_id]:
                        logger.warning(f"Index type {idx_type} not found for corpus {corpus_id}")
                        continue

                    index = self.indices[corpus_id][idx_type]
                    if node_ids is not None or metadata_filters is not None:
                        # Convert single node_id to list for consistency
                        node_ids_list = [node_ids] if isinstance(node_ids, str) else node_ids
                        index.delete_nodes(node_ids=node_ids_list, metadata_filters=metadata_filters)
                        logger.info(f"Deleted nodes from {idx_type} index for corpus {corpus_id}")
                    else:
                        # Delete entire index
                        index.clear()
                        del self.indices[corpus_id][idx_type]
                        del self.retrievers[corpus_id][idx_type]
                        logger.info(f"Deleted entire {idx_type} index for corpus {corpus_id}")

                # Clean up corpus if no indices remain
                if not self.indices[corpus_id]:
                    del self.indices[corpus_id]
                    del self.retrievers[corpus_id]
                    logger.info(f"Removed empty corpus {corpus_id}")

        except Exception as e:
            logger.error(f"Failed to delete from corpus {corpus_id}, index {index_type}: {str(e)}")
            raise

    def clear(self, corpus_id: Optional[str] = None) -> None:
        """Clear all indices for a specific corpus or all corpora.

        Args:
            corpus_id (Optional[str]): Specific corpus to clear. If None, clears all corpora.

        Raises:
            Exception: If clearing fails.
        """
        try:
            with self._registry_lock:
                target_corpora = [corpus_id] if corpus_id is not None else list(self.indices.keys())
                for cid in target_corpora:
                    if cid not in self.indices:
                        logger.warning(f"No indices found for corpus {cid}")
                        continue

                    for idx_type in list(self.indices[cid].keys()):
                        index = self.indices[cid][idx_type]
                        index.clear()
                        del self.indices[cid][idx_type]
                        del self.retrievers[cid][idx_type]
                        logger.info(f"Cleared {idx_type} index for corpus {cid}")

                    # Clean up corpus if no indices remain
                    del self.indices[cid]
                    del self.retrievers[cid]
                    logger.info(f"Cleared corpus {cid}")

        except Exception as e:
            logger.error(f"Failed to clear indices for corpus {corpus_id or 'all'}: {str(e)}")
            raise

    def _index_snapshots(self, corpus_id, index_type, graph_exported):
        """Capture detached committed data for exactly the selected indices."""
        with self._registry_lock:
            selected = [
                (cid, kind, index)
                for cid, indices in self.indices.items()
                if corpus_id is None or cid == corpus_id
                for kind, index in indices.items()
                if index_type is None or kind == index_type
            ]
        for cid, kind, index in selected:
            if kind == IndexType.GRAPH:
                if not graph_exported:
                    continue
                index.build_kv_store()
            nodes = index.snapshot_nodes() if hasattr(index, "snapshot_nodes") else list(index.id_to_node.values())
            corpus = Corpus.from_llama_nodes(nodes)
            corpus.corpus_id = cid
            vector = self.storage_handler.storageConfig.vectorConfig
            graph = self.storage_handler.storageConfig.graphConfig
            metadata = IndexMetadata(
                corpus_id=cid,
                index_type=kind,
                dimension=self.embed_model.dimensions,
                vector_db_type=vector.vector_name if vector else None,
                graph_db_type=graph.graph_name if graph else None,
                embedding_model_name=self.config.embedding.model_name,
                collection_name=vector.qdrant_collection_name if vector else None,
                date=str(datetime.now()),
            )
            yield IndexSnapshot(metadata=metadata, corpus=corpus)

    def save(self, output_path: Optional[str] = None, corpus_id: Optional[str] = None,
             index_type: Optional[str] = None, table: Optional[str] = None,
             graph_exported: bool = False) -> None:
        """Save committed snapshots to portable files or the configured database.

        File metadata is published after immutable content, with a checksum and
        the original logical corpus identity. Database rows retain all saved
        index types in a corpus, including types not selected by this call.
        """
        for snapshot in self._index_snapshots(corpus_id, index_type, graph_exported):
            if output_path is not None:
                save_file_snapshot(snapshot, output_path)
            else:
                self.storage_handler.save_index(snapshot.to_record(), table=table or "indexing")

    def _validate_snapshot(self, snapshot: IndexSnapshot) -> None:
        """Reject incompatible data before modifying any live vector index."""
        metadata = snapshot.metadata
        IndexType(metadata.index_type)
        if metadata.index_type != IndexType.VECTOR:
            return
        vector = self.storage_handler.storageConfig.vectorConfig
        if vector is None:
            raise ValueError("Loading vector snapshots requires a vector backend")
        if metadata.dimension != vector.dimensions:
            raise ValueError(
                f"Embedding dimension {metadata.dimension} does not match configured dimension {vector.dimensions}"
            )
        if metadata.embedding_model_name != self.config.embedding.model_name:
            raise ValueError(
                f"Snapshot embedding model {metadata.embedding_model_name!r} does not match "
                f"configured model {self.config.embedding.model_name!r}; configure an engine for the saved model"
            )
        if not self.embed_model.validate_model(self.config.embedding.provider, metadata.embedding_model_name):
            raise ValueError(f"Snapshot embedding model {metadata.embedding_model_name!r} is not supported")
        if metadata.vector_db_type is not None and metadata.vector_db_type != vector.vector_name:
            raise ValueError("Snapshot vector backend does not match the configured backend")
        expected_image = self.config.modality == "multimodal"
        if any(isinstance(chunk, ImageChunk) != expected_image for chunk in snapshot.corpus.chunks):
            raise ValueError("Snapshot chunk modality does not match the configured engine")

    def load(self, source: Optional[str] = None, corpus_id: Optional[str] = None,
             index_type: Optional[str] = None, table: Optional[str] = None) -> Sequence[str]:
        """Replace selected vector indices with validated, isolated snapshots.

        Repeated loading is idempotent and removes nodes absent from the saved
        snapshot. All selected vector indices are prepared before publication;
        a parsing, compatibility, embedding or backend failure leaves the live
        registry untouched. Other corpora and unselected index types are kept.
        Legacy JSONL/metadata pairs and single-index DB rows remain readable.
        """
        if source is not None:
            snapshots = list(load_file_snapshots(source, corpus_id, index_type))
        else:
            snapshots = []
            records = self.storage_handler.load(tables=[table or "indexing"]).get(table or "indexing", [])
            for record in records:
                parsed = self.storage_handler.parse_result(record, IndexStore)
                if corpus_id is not None and parsed["corpus_id"] != corpus_id:
                    continue
                for single in expand_index_record(parsed):
                    if index_type is None or single["metadata"].get("index_type") == index_type:
                        snapshots.append(snapshot_from_record(single))
        seen = set()
        for snapshot in snapshots:
            self._validate_snapshot(snapshot)
            identity = (snapshot.metadata.corpus_id, snapshot.metadata.index_type)
            if identity in seen:
                raise ValueError(f"Multiple snapshots found for {identity}")
            seen.add(identity)
        prepared = []
        identifiers = []
        for snapshot in snapshots:
            cid, kind = snapshot.metadata.corpus_id, snapshot.metadata.index_type
            if kind == IndexType.VECTOR:
                index = self._new_index(kind)
                retriever = self._new_retriever(kind, index)
                owned = self._owned_nodes(snapshot.corpus, cid, kind)
                loaded = index.load(owned)
                if len(loaded) != len(owned):
                    raise RuntimeError(f"Incomplete snapshot ingestion for corpus {cid}")
                prepared.append((cid, kind, index, retriever))
                identifiers.extend(loaded)
            else:
                # Graph persistence has its own external backend lifecycle.
                # Retain that path without promising vector-style rollback.
                identifiers.extend(self._load_index(snapshot.corpus, cid, kind))
        with self._registry_lock:
            for cid, kind, index, retriever in prepared:
                self.indices.setdefault(cid, {})[kind] = index
                self.retrievers.setdefault(cid, {})[kind] = retriever
        return identifiers

    def _load_index(self, corpus: Corpus, corpus_id: str, index_type: str) -> Sequence[str]:
        """Load one index; vector snapshots replace rather than append data."""
        with self._registry_lock:
            index = self.indices.get(corpus_id, {}).get(index_type)
            if index_type == IndexType.VECTOR or index is None:
                index = self._new_index(index_type)
                retriever = self._new_retriever(index_type, index)
            else:
                retriever = self.retrievers[corpus_id][index_type]
            loaded = index.load(self._owned_nodes(corpus, corpus_id, index_type))
            self.indices.setdefault(corpus_id, {})[index_type] = index
            self.retrievers.setdefault(corpus_id, {})[index_type] = retriever
            return loaded

    async def aget(self, corpus_id: str, index_type: str, node_ids: List[str]) -> List[Union[TextChunk, ImageChunk]]:
        """Retrieve chunks by node_ids from the index."""
        try:
            chunks = await self.indices[corpus_id][index_type].get(node_ids=node_ids)
            logger.info(f"Retrieved {len(chunks)} chunks for node_ids: {node_ids}")
            return chunks
        except Exception as e:
            logger.error(f"Failed to get chunks: {str(e)}")
            return []

    async def query_async(self, query: Union[str, Query], corpus_id: Optional[str] = None,
                        query_transforms: Optional[List] = None) -> RagResult:
        """Execute a query across indices and return processed results asynchronously.

        Performs query preprocessing, asynchronous retrieval, and post-processing.

        Args:
            query (Union[str, Query]): Query string or Query object.
            corpus_id (Optional[str]): Specific corpus to query. If None, queries all corpora.
            query_transforms (Optional[List]): Query Transforms is used to augment query in pre-processing.

        Returns:
            RagResult: Retrieved chunks with scores and metadata.

        Raises:
            Exception: If query processing fails.
        """
        try:
            if isinstance(query, str):
                query = Query(query_str=query, top_k=self.config.retrieval.top_k)
            
            if not self.indices or (corpus_id and corpus_id not in self.indices):
                logger.warning(f"No indices found for corpus {corpus_id or 'any'}")
                return RagResult(corpus=Corpus(chunks=[]), scores=[], metadata={"query": query.query_str})

            # Pre-Processing
            if query_transforms and query_transforms is not None:
                for t in query_transforms:
                    query = t(query)

            results = []
            # Capture references before awaiting. A concurrent snapshot load can
            # then publish a new registry while these requests finish against
            # their original, internally consistent corpus-owned indices.
            with self._registry_lock:
                selected = [
                    (cid, kind, retriever)
                    for cid, retrievers in self.retrievers.items()
                    if corpus_id is None or cid == corpus_id
                    for kind, retriever in retrievers.items()
                ]
            tasks = []
            for cid, idx_type, retriever in selected:
                if query.metadata_filters and query.metadata_filters.get("index_type") and \
                        query.metadata_filters["index_type"] != idx_type:
                    continue
                task = retriever.aretrieve(
                    Query(
                        query_str=query.query_str,
                        top_k=query.top_k if query.top_k is not None else self.config.retrieval.top_k,
                        custom_embedding_strs=query.custom_embedding_strs,
                        similarity_cutoff=query.similarity_cutoff,
                        keyword_filters=query.keyword_filters,
                        metadata_filters=query.metadata_filters,
                    )
                )
                tasks.append((task, cid, idx_type))

            # Run all retrievals concurrently
            retrieval_tasks = [task for task, _, _ in tasks]
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

            # Process results
            for (_, cid, idx_type), result in zip(tasks, retrieval_results):
                if isinstance(result, Exception):
                    logger.error(f"Retrieval failed for {idx_type} in corpus {cid}: {str(result)}")
                else:
                    results.append(result)
                    logger.info(f"Retrieved {len(result.corpus.chunks)} chunks from {idx_type} retriever for corpus {cid}")

            if not results:
                return RagResult(corpus=Corpus(chunks=[]), scores=[], metadata={"query": query.query_str})

            # Check the 'similarity_cutoff' and 'keyword_filters' in query
            query.similarity_cutoff = self.config.retrieval.similarity_cutoff if query.similarity_cutoff is None else query.similarity_cutoff
            query.keyword_filters = self.config.retrieval.keyword_filters if query.keyword_filters is None else query.keyword_filters

            postprocessor = self.postprocessor_factory.create(
                self.config.retrieval.postprocessor_type,
                query=query
            )
            final_result = postprocessor.postprocess(query, results)

            if query.metadata_filters:
                final_result.corpus.chunks = [
                    chunk for chunk in final_result.corpus.chunks
                    if all(chunk.metadata.model_dump().get(k) == v for k, v in query.metadata_filters.items())
                ]
                final_result.scores = [chunk.metadata.similarity_score for chunk in final_result.corpus.chunks]
                logger.info(f"Applied metadata filters, retained {len(final_result.corpus.chunks)} chunks")

            logger.info(f"Query returned {len(final_result.corpus.chunks)} chunks after post-processing")
            return final_result
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise

    def query(self, query: Union[str, Query], corpus_id: Optional[str] = None,
            query_transforms: Optional[List] = None) -> RagResult:
        """Synchronous wrapper for the async query method."""
        return asyncio.run(self.query_async(query, corpus_id, query_transforms))