"""
FAISS Database Toolkit for EvoAgentX

This module provides a comprehensive toolkit for interacting with FAISS vector databases
through the existing RAG engine and storage infrastructure. It includes tools for querying,
inserting, deleting, and managing vector data with semantic search capabilities.

The toolkit wraps the existing RAGEngine and StorageHandler to provide a unified interface
for vector database operations that can be easily used by agents.

Key Features:
- Automatic database path validation and creation
- Support for existing database detection and reuse
- Robust error handling for invalid paths
- Default configuration with sensible defaults
- Seamless integration with the RAG pipeline
- Automatic file path detection and content processing
- Support for multiple file formats (PDF, text, Markdown, etc.)

Usage:
    # Using default configuration with automatic path handling
    toolkit = FaissToolkit(db_path="./my_database.db")
    
    # Using custom configuration
    toolkit = FaissToolkit(
        storage_config=custom_storage_config,
        rag_config=custom_rag_config
    )
    
    # Insert documents including file paths
    toolkit.get_tool("faiss_insert")(
        documents=[
            "This is some text content",
            "./documents/report.pdf",  # Will be automatically read and processed
            "./data/notes.txt"         # Will be automatically read and processed
        ]
    )
"""

import os
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional
from uuid import uuid4
from datetime import datetime
from pathlib import Path

from .tool import Tool, Toolkit
from .storage_handler import LocalStorageHandler
from ..core.module import BaseModule
from ..core.logging import logger
from ..rag.rag import RAGEngine
from ..rag.rag_config import RAGConfig
from ..rag.schema import Query, Document, Corpus, DocumentMetadata
from ..storages.base import StorageHandler
from ..storages.storages_config import StoreConfig
from .storage_handler import FileStorageHandler


def _ensure_database_path(db_path: str) -> str:
    """
    Ensure the database path exists and is properly configured.
    
    Args:
        db_path (str): The database file path
        
    Returns:
        str: The validated and prepared database path
        
    Raises:
        ValueError: If the path is invalid or cannot be created
    """
    if not db_path:
        raise ValueError("Database path cannot be empty")
    
    # Convert to Path object for easier manipulation and make it absolute
    path = Path(db_path).resolve()
    
    # Validate the path is not a directory
    if path.exists() and path.is_dir():
        raise ValueError(f"Database path points to a directory: {db_path}")
    
    # Ensure the parent directory exists
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Cannot create directory for database path {db_path}: {e}")
    
    # Check if database file exists
    if path.exists():
        logger.info(f"Found existing database at: {db_path}")
        # Validate it's a valid SQLite database by trying to connect
        try:
            import sqlite3
            conn = sqlite3.connect(str(path))
            conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            conn.close()
            logger.info("Database validation successful")
        except Exception as e:
            logger.warning(f"Database validation failed: {e}. Will create new database.")
            # If validation fails, remove the corrupted file
            try:
                path.unlink()
            except Exception as unlink_error:
                logger.error(f"Failed to remove corrupted database file: {unlink_error}")
                raise ValueError(f"Cannot remove corrupted database file: {unlink_error}")
    else:
        logger.info(f"Database not found at: {db_path}. Will create new database.")
    
    return str(path)


def _create_default_storage_config(db_path: Optional[str] = None) -> StoreConfig:
    """
    Create a default storage configuration with proper path handling.
    
    Args:
        db_path (str, optional): Custom database path
        
    Returns:
        StoreConfig: Configured storage configuration
    """
    from ..storages.storages_config import StoreConfig, DBConfig, VectorStoreConfig
    
    # Set default database path if not provided
    if db_path is None:
        db_path = "./faiss_db.sqlite"
    
    # Ensure the database path is properly set up
    validated_db_path = _ensure_database_path(db_path)
    logger.info(f"Using validated database path: {validated_db_path}")
    
    # Create index cache path (ensure it's absolute)
    index_cache_path = str(Path(validated_db_path).parent.resolve() / "index_cache")
    
    # Create storage configuration
    storage_config = StoreConfig(
        dbConfig=DBConfig(
            db_name="sqlite", 
            path=validated_db_path
        ),
        vectorConfig=VectorStoreConfig(
            vector_name="faiss",
            dimensions=1536,
            index_type="flat_l2"
        ),
        path=index_cache_path
    )
    
    # Ensure the index cache directory exists
    Path(index_cache_path).mkdir(parents=True, exist_ok=True)
    
    return storage_config


def _create_default_rag_config() -> RAGConfig:
    """
    Create a default RAG configuration.
    
    Returns:
        RAGConfig: Configured RAG configuration
    """
    from ..rag.rag_config import RAGConfig, EmbeddingConfig, ChunkerConfig
    
    return RAGConfig(
        embedding=EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-ada-002"
        ),
        chunker=ChunkerConfig(
            chunk_size=500,
            chunk_overlap=50
        )
    )


class FaissDatabase(BaseModule):
    """
    A high-level interface for FAISS vector database operations.
    
    This class wraps the RAGEngine and StorageHandler to provide a unified interface
    for vector database operations including document ingestion, semantic search,
    and corpus management.
    
    Attributes:
        rag_engine (RAGEngine): The RAG engine for document processing and retrieval
        storage_handler (StorageHandler): The storage handler for persistence
        default_corpus_id (str): Default corpus ID for operations
        default_index_type (str): Default index type for vector operations
    """
    
    def __init__(
        self,
        storage_config: StoreConfig,
        rag_config: RAGConfig,
        default_corpus_id: str = "default",
        default_index_type: str = "vector",
        storage_handler: StorageHandler = None,
        file_handler: FileStorageHandler = None,
        **kwargs
    ):
        """
        Initialize the FAISS database.
        
        Args:
            storage_config (StoreConfig): Configuration for storage backends
            rag_config (RAGConfig): Configuration for RAG pipeline
            default_corpus_id (str): Default corpus ID for operations
            default_index_type (str): Default index type for vector operations
            storage_handler (StorageHandler, optional): Storage handler for file operations
            **kwargs: Additional arguments for BaseModule
        """
        super().__init__(**kwargs)
        
        # Initialize storage handler for database operations
        self.storage_handler = StorageHandler(storageConfig=storage_config)
        
        # Initialize RAG engine
        self.rag_engine = RAGEngine(config=rag_config, storage_handler=self.storage_handler)
        
        # Initialize file storage handler for external file operations
        # Use LocalStorageHandler as default if none provided
        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path="./workplace/storage")
        self.file_storage_handler = storage_handler
        
        # Set defaults
        self.default_corpus_id = default_corpus_id
        self.default_index_type = default_index_type
        
        logger.info(f"Initialized FAISS database with corpus_id: {default_corpus_id}")
    
    def query(
        self,
        query: str,
        corpus_id: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector database with semantic search.
        
        Args:
            query (str): The query string to search for
            corpus_id (str, optional): Corpus ID to search in
            top_k (int): Number of top results to return
            similarity_threshold (float): Minimum similarity threshold
            metadata_filters (Dict[str, Any], optional): Metadata filters for search
            
        Returns:
            Dict[str, Any]: Search results with chunks and scores
        """
        try:
            # Check if we're already in an event loop
            try:
                asyncio.get_running_loop()
                # We're in an event loop, use thread executor to avoid asyncio.run() conflict
                logger.info("Detected running event loop, using thread executor for query")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._query_sync, query, corpus_id, top_k, similarity_threshold, metadata_filters)
                    return future.result()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run() in the RAG engine
                logger.info("No event loop detected, using direct query processing")
                return self._query_sync(query, corpus_id, top_k, similarity_threshold, metadata_filters)
                
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _query_sync(
        self,
        query: str,
        corpus_id: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous version of query that can be safely called from a thread.
        
        Args:
            query (str): The query string to search for
            corpus_id (str, optional): Corpus ID to search in
            top_k (int): Number of top results to return
            similarity_threshold (float): Minimum similarity threshold
            metadata_filters (Dict[str, Any], optional): Metadata filters for search
            
        Returns:
            Dict[str, Any]: Search results with chunks and scores
        """
        try:
            corpus_id = corpus_id or self.default_corpus_id
            
            # Check if corpus exists
            if corpus_id not in self.rag_engine.indices:
                logger.warning(f"Corpus {corpus_id} not found. Returning empty results.")
                return {"success": True, "data": {
                    "query": query,
                    "corpus_id": corpus_id,
                    "total_results": 0,
                    "results": []
                }}
            
            # Create query object
            query_obj = Query(
                query_str=query,
                top_k=top_k,
                similarity_cutoff=similarity_threshold,
                metadata_filters=metadata_filters
            )
            
            # Execute query
            results = self.rag_engine.query(query_obj, corpus_id=corpus_id)
            
            # Handle case where results or corpus might be None
            if not results or not results.corpus:
                logger.warning(f"Query returned no results for corpus {corpus_id}")
                return {"success": True, "data": {
                    "query": query,
                    "corpus_id": corpus_id,
                    "total_results": 0,
                    "results": []
                }}
            
            # Format results
            chunks = results.corpus.chunks if results.corpus.chunks else []
            formatted_results = {
                "query": query,
                "corpus_id": corpus_id,
                "total_results": len(chunks),
                "results": []
            }
            
            for i, chunk in enumerate(chunks):
                score = results.scores[i] if results.scores and i < len(results.scores) else 0.0
                formatted_results["results"].append({
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.text,
                    "score": score,
                    "metadata": chunk.metadata.model_dump() if chunk.metadata else {},
                    "doc_id": chunk.metadata.doc_id if chunk.metadata else None
                })
            
            logger.info(f"Query executed successfully. Found {len(formatted_results['results'])} results.")
            return {"success": True, "data": formatted_results}
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _is_file_path(self, text: str) -> bool:
        """
        Check if a string appears to be a file path.
        
        Args:
            text (str): The string to check
            
        Returns:
            bool: True if the string looks like a file path
        """
        # Check if it contains path separators or file extensions
        path_indicators = ['/', '\\', '.txt', '.pdf', '.md', '.doc', '.docx', '.csv', '.json', '.xml', '.html', '.htm']
        return any(indicator in text for indicator in path_indicators) and os.path.exists(text)
    
    def _process_file_path(self, file_path: str, doc_index: int, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Process a file path and return Document objects.
        
        Args:
            file_path (str): Path to the file
            doc_index (int): Index of the document in the batch
            metadata (Dict[str, Any], optional): Additional metadata
            
        Returns:
            List[Document]: List of Document objects created from the file
        """
        try:
            # Check if we're already in an event loop
            try:
                asyncio.get_running_loop()
                # We're in an event loop, use thread executor to avoid asyncio.run() conflict
                logger.info(f"Detected running event loop, using thread executor for {file_path}")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._process_file_path_sync, file_path, doc_index, metadata)
                    return future.result()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run() in the RAG engine
                logger.info(f"No event loop detected, using direct processing for {file_path}")
                return self._process_file_path_sync(file_path, doc_index, metadata)
                
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {str(e)}")
            # Return a single document with error information
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update({
                "doc_index": doc_index,
                "insertion_time": datetime.now().isoformat(),
                "source_file": file_path,
                "error": str(e)
            })
            
            document_metadata = DocumentMetadata(**doc_metadata)
            return [Document(
                text=f"Error reading file {file_path}: {str(e)}",
                metadata=document_metadata,
                doc_id=str(uuid4())
            )]

    def _process_file_path_sync(self, file_path: str, doc_index: int, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Synchronous version of file processing that can be safely called from a thread.
        
        Args:
            file_path (str): Path to the file
            doc_index (int): Index of the document in the batch
            metadata (Dict[str, Any], optional): Additional metadata
            
        Returns:
            List[Document]: List of Document objects created from the file
        """
        try:
            # Use StorageHandler to read the file if available
            if self.file_storage_handler:
                result = self.file_storage_handler.read(file_path)
                if result["success"]:
                    file_content = result["content"]
                else:
                    raise Exception(f"Failed to read file: {result.get('error', 'Unknown error')}")
            else:
                # Fallback to direct file reading
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            
            # Use RAG engine to process the content
            temp_corpus_id = f"temp_file_{uuid4().hex[:8]}"
            
            # Create a temporary document for processing
            temp_doc = Document(
                text=file_content,
                metadata=DocumentMetadata(
                    source_file=file_path,
                    doc_index=doc_index,
                    insertion_time=datetime.now().isoformat()
                ),
                doc_id=str(uuid4())
            )
            
            # Process the document through RAG engine
            corpus = self.rag_engine.process_documents([temp_doc], corpus_id=temp_corpus_id)
            
            # Convert chunks back to documents with proper metadata
            documents = []
            for chunk in corpus.chunks:
                doc_metadata = metadata.copy() if metadata else {}
                doc_metadata.update({
                    "doc_index": doc_index,
                    "insertion_time": datetime.now().isoformat(),
                    "source_file": file_path,
                    "original_chunk_id": chunk.chunk_id
                })
                
                # Create DocumentMetadata object
                document_metadata = DocumentMetadata(**doc_metadata)
                
                # Create Document object
                documents.append(Document(
                    text=chunk.text,
                    metadata=document_metadata,
                    doc_id=chunk.chunk_id
                ))
            
            # Clean up temporary corpus
            self.rag_engine.clear(corpus_id=temp_corpus_id)
            
            logger.info(f"Processed file {file_path} into {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path} in sync mode: {str(e)}")
            # Return a single document with error information
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update({
                "doc_index": doc_index,
                "insertion_time": datetime.now().isoformat(),
                "source_file": file_path,
                "error": str(e)
            })
            
            document_metadata = DocumentMetadata(**doc_metadata)
            return [Document(
                text=f"Error reading file {file_path}: {str(e)}",
                metadata=document_metadata,
                doc_id=str(uuid4())
            )]
    
    def insert(
        self,
        documents: list,
        corpus_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Insert documents into the vector database.
        
        Args:
            documents (Union[List[str], List[Dict[str, Any]]]): Documents to insert. 
                Strings can be either text content or file paths (if they look like paths and exist)
            corpus_id (str, optional): Corpus ID to insert into
            metadata (Dict[str, Any], optional): Additional metadata for all documents
            batch_size (int): Batch size for processing
            
        Returns:
            Dict[str, Any]: Insertion results
        """
        try:
            corpus_id = corpus_id or self.default_corpus_id
            
            # Process documents and create proper Document objects
            processed_docs = []
            file_paths_processed = []
            
            for i, doc in enumerate(documents):
                if isinstance(doc, str):
                    # Check if this string looks like a file path
                    if self._is_file_path(doc):
                        logger.info(f"Detected file path: {doc}")
                        file_docs = self._process_file_path(doc, i, metadata)
                        processed_docs.extend(file_docs)
                        file_paths_processed.append(doc)
                    else:
                        # Treat as regular text content
                        doc_metadata = metadata.copy() if metadata else {}
                        doc_metadata.update({
                            "doc_index": i,
                            "insertion_time": datetime.now().isoformat()
                        })
                        # Create DocumentMetadata object
                        document_metadata = DocumentMetadata(**doc_metadata)
                        # Create Document object
                        processed_docs.append(Document(
                            text=doc,
                            metadata=document_metadata,
                            doc_id=str(uuid4())
                        ))
                elif isinstance(doc, dict):
                    doc_metadata = metadata.copy() if metadata else {}
                    doc_metadata.update(doc.get("metadata", {}))
                    doc_metadata.update({
                        "doc_index": i,
                        "insertion_time": datetime.now().isoformat()
                    })
                    # Create DocumentMetadata object
                    document_metadata = DocumentMetadata(**doc_metadata)
                    # Create Document object
                    processed_docs.append(Document(
                        text=doc.get("text", ""),
                        metadata=document_metadata,
                        doc_id=doc.get("doc_id", str(uuid4()))
                    ))
            
            # Create corpus
            corpus = Corpus(corpus_id=corpus_id)
            
            # Process in batches
            total_processed = 0
            for i in range(0, len(processed_docs), batch_size):
                batch = processed_docs[i:i+batch_size]
                
                # Chunk the documents
                batch_corpus = self.rag_engine.chunker.chunk(batch)
                batch_corpus.corpus_id = corpus_id
                
                # Add to index
                self.rag_engine.add(self.default_index_type, batch_corpus, corpus_id=corpus_id)
                
                # Add chunks to corpus
                corpus.chunks.extend(batch_corpus.chunks)
                total_processed += len(batch)
                
                logger.info(f"Processed batch {i//batch_size + 1}, total processed: {total_processed}")
            
            # Save index
            self.rag_engine.save(corpus_id=corpus_id, index_type=self.default_index_type)
            
            result = {
                "corpus_id": corpus_id,
                "documents_inserted": len(documents),
                "chunks_created": len(corpus.chunks),
                "total_processed": total_processed,
                "file_paths_processed": file_paths_processed
            }
            
            logger.info(f"Successfully inserted {len(documents)} documents into corpus {corpus_id}")
            if file_paths_processed:
                logger.info(f"Processed {len(file_paths_processed)} file paths: {file_paths_processed}")
            return {"success": True, "data": result}
            
        except Exception as e:
            logger.error(f"Insert failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def delete(
        self,
        corpus_id: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        clear_all: bool = False
    ) -> Dict[str, Any]:
        """
        Delete documents or chunks from the vector database.
        
        Args:
            corpus_id (str, optional): Corpus ID to delete from
            doc_ids (List[str], optional): Document IDs to delete
            metadata_filters (Dict[str, Any], optional): Metadata filters for deletion
            clear_all (bool): Whether to clear the entire corpus
            
        Returns:
            Dict[str, Any]: Deletion results
        """
        try:
            corpus_id = corpus_id or self.default_corpus_id
            
            if clear_all:
                # Clear entire corpus
                self.rag_engine.clear(corpus_id=corpus_id)
                logger.info(f"Cleared entire corpus: {corpus_id}")
                return {"success": True, "data": {"operation": "clear_all", "corpus_id": corpus_id}}
            
            # Check if corpus exists before attempting deletion
            if corpus_id not in self.rag_engine.indices:
                logger.warning(f"Corpus {corpus_id} not found. Nothing to delete.")
                return {"success": True, "data": {"operation": "selective_delete", "corpus_id": corpus_id, "message": "Corpus not found, nothing to delete"}}
            
            # Only attempt deletion if there are specific criteria
            if doc_ids or metadata_filters:
                # Delete specific documents or by filters
                self.rag_engine.delete(
                    corpus_id=corpus_id,
                    index_type=self.default_index_type,
                    node_ids=doc_ids,
                    metadata_filters=metadata_filters
                )
                
                result = {
                    "corpus_id": corpus_id,
                    "operation": "selective_delete",
                    "doc_ids": doc_ids,
                    "metadata_filters": metadata_filters
                }
                
                logger.info(f"Successfully deleted from corpus {corpus_id}")
                return {"success": True, "data": result}
            else:
                # No deletion criteria provided
                logger.warning(f"No deletion criteria provided for corpus {corpus_id}")
                return {"success": True, "data": {"operation": "selective_delete", "corpus_id": corpus_id, "message": "No deletion criteria provided"}}
            
        except Exception as e:
            logger.error(f"Delete failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def list_corpora(self) -> Dict[str, Any]:
        """
        List all available corpora and their metadata.
        
        Returns:
            Dict[str, Any]: List of corpora with metadata
        """
        try:
            corpora = []
            
            # Get corpus information from indices
            for corpus_id, indices in self.rag_engine.indices.items():
                corpus_info = {
                    "corpus_id": corpus_id,
                    "index_types": list(indices.keys()),
                    "retrievers": list(self.rag_engine.retrievers.get(corpus_id, {}).keys())
                }
                corpora.append(corpus_info)
            
            return {"success": True, "data": {"corpora": corpora, "total": len(corpora)}}
            
        except Exception as e:
            logger.error(f"List corpora failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_stats(self, corpus_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about the database or a specific corpus.
        
        Args:
            corpus_id (str, optional): Corpus ID to get stats for
            
        Returns:
            Dict[str, Any]: Database statistics
        """
        try:
            if corpus_id:
                # Stats for specific corpus
                corpus_id = corpus_id or self.default_corpus_id
                
                stats = {
                    "corpus_id": corpus_id,
                    "exists": corpus_id in self.rag_engine.indices,
                    "index_types": list(self.rag_engine.indices.get(corpus_id, {}).keys()),
                    "retrievers": list(self.rag_engine.retrievers.get(corpus_id, {}).keys())
                }
                
                # Try to get vector store stats if available
                if corpus_id in self.rag_engine.indices:
                    vector_index = self.rag_engine.indices[corpus_id].get(self.default_index_type)
                    if vector_index and hasattr(vector_index, 'get_index'):
                        try:
                            index = vector_index.get_index()
                            if hasattr(index, 'vector_store'):
                                vector_store = index.vector_store
                                if hasattr(vector_store, 'faiss_index'):
                                    stats["vector_count"] = vector_store.faiss_index.ntotal
                                    stats["dimensions"] = vector_store.faiss_index.d
                        except Exception:
                            pass
                
                return {"success": True, "data": stats}
            else:
                # Global stats
                stats = {
                    "total_corpora": len(self.rag_engine.indices),
                    "corpora": list(self.rag_engine.indices.keys()),
                    "embedding_model": self.rag_engine.config.embedding.model_name,
                    "vector_store_type": self.rag_engine.storage_handler.storageConfig.vectorConfig.vector_name if self.rag_engine.storage_handler.storageConfig.vectorConfig else None
                }
                
                return {"success": True, "data": stats}
                
        except Exception as e:
            logger.error(f"Get stats failed: {str(e)}")
            return {"success": False, "error": str(e)}


class FaissQueryTool(Tool):
    """Tool for querying the FAISS vector database with semantic search."""
    
    name: str = "faiss_query"
    description: str = "Query the FAISS vector database using semantic search to find relevant documents and chunks"
    inputs: Dict[str, Dict[str, Any]] = {
        "query": {
            "type": "string",
            "description": "The search query text to find semantically similar content"
        },
        "corpus_id": {
            "type": "string",
            "description": "Optional corpus ID to search in. If not provided, uses default corpus"
        },
        "top_k": {
            "type": "integer",
            "description": "Number of top results to return (default: 5)",
            "default": 5
        },
        "similarity_threshold": {
            "type": "number",
            "description": "Minimum similarity threshold for results (default: 0.0)",
            "default": 0.0
        },
        "metadata_filters": {
            "type": "object",
            "description": "Optional metadata filters to apply to search results (e.g., {'source': 'file1.txt'})"
        }
    }
    required: Optional[List[str]] = ["query"]
    
    def __init__(self, faiss_database: FaissDatabase = None):
        super().__init__()
        self.faiss_database = faiss_database
    
    def __call__(
        self,
        query: str,
        corpus_id: str = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        metadata_filters: dict = None
    ) -> Dict[str, Any]:
        """Execute the query operation."""
        return self.faiss_database.query(
            query=query,
            corpus_id=corpus_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filters=metadata_filters
        )


class FaissInsertTool(Tool):
    """Tool for inserting documents into the FAISS vector database."""
    
    name: str = "faiss_insert"
    description: str = "Insert documents into the FAISS vector database with automatic chunking and embedding. Supports both text content and file paths - if a string looks like a file path and exists, it will automatically read and process the file content."
    inputs: Dict[str, Dict[str, Any]] = {
        "documents": {
            "type": "array",
            "description": "Array of documents to insert. Can be strings (text content or file paths), or objects with 'text', 'metadata', and 'doc_id' fields. If a string contains path separators or file extensions and the file exists, it will be treated as a file path and its content will be read and processed.",
            "items": {"anyOf": [{"type": "string"}, {"type": "object"}]}
        },
        "corpus_id": {
            "type": "string",
            "description": "Optional corpus ID to insert into. If not provided, uses default corpus"
        },
        "metadata": {
            "type": "object",
            "description": "Optional metadata to add to all documents (e.g., {'source': 'file1.txt', 'category': 'research'})"
        },
        "batch_size": {
            "type": "integer",
            "description": "Batch size for processing documents (default: 100)",
            "default": 100
        }
    }
    required: Optional[List[str]] = ["documents"]
    
    def __init__(self, faiss_database: FaissDatabase = None):
        super().__init__()
        self.faiss_database = faiss_database
    
    def __call__(
        self,
        documents: list,
        corpus_id: str = None,
        metadata: dict = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Execute the insert operation."""
        return self.faiss_database.insert(
            documents=documents,
            corpus_id=corpus_id,
            metadata=metadata,
            batch_size=batch_size
        )


class FaissDeleteTool(Tool):
    """Tool for deleting documents from the FAISS vector database."""
    
    name: str = "faiss_delete"
    description: str = "Delete documents or chunks from the FAISS vector database. You can delete specific documents by ID, filter by metadata, or clear the entire corpus."
    inputs: Dict[str, Dict[str, Any]] = {
        "corpus_id": {
            "type": "string",
            "description": "Optional corpus ID to delete from. If not provided, uses default corpus"
        },
        "doc_ids": {
            "type": "array",
            "description": "Optional list of document IDs to delete. Use this to delete specific documents",
            "items": {"type": "string"}
        },
        "metadata_filters": {
            "type": "object",
            "description": "Optional metadata filters to select documents for deletion (e.g., {'source': 'file1.txt'})"
        },
        "clear_all": {
            "type": "boolean",
            "description": "Set to true to clear the entire corpus. WARNING: This will delete all documents in the corpus",
            "default": False
        }
    }
    required: Optional[List[str]] = []
    
    def __init__(self, faiss_database: FaissDatabase = None):
        super().__init__()
        self.faiss_database = faiss_database
    
    def __call__(
        self,
        corpus_id: str = None,
        doc_ids: list = None,
        metadata_filters: dict = None,
        clear_all: bool = False
    ) -> Dict[str, Any]:
        """Execute the delete operation."""
        return self.faiss_database.delete(
            corpus_id=corpus_id,
            doc_ids=doc_ids,
            metadata_filters=metadata_filters,
            clear_all=clear_all
        )


class FaissListTool(Tool):
    """Tool for listing available corpora in the FAISS vector database."""
    
    name: str = "faiss_list"
    description: str = "List all available corpora and their metadata in the FAISS vector database. This tool takes no parameters."
    inputs: Dict[str, Dict[str, Any]] = {}
    required: Optional[List[str]] = []
    
    def __init__(self, faiss_database: FaissDatabase = None):
        super().__init__()
        self.faiss_database = faiss_database
    
    def __call__(self) -> Dict[str, Any]:
        """Execute the list operation."""
        return self.faiss_database.list_corpora()


class FaissStatsTool(Tool):
    """Tool for getting statistics about the FAISS vector database."""
    
    name: str = "faiss_stats"
    description: str = "Get statistics about the FAISS vector database or a specific corpus. Optionally provide a corpus_id to get stats for a specific corpus."
    inputs: Dict[str, Dict[str, Any]] = {
        "corpus_id": {
            "type": "string",
            "description": "Optional corpus ID to get statistics for. If not provided, returns global statistics"
        }
    }
    required: Optional[List[str]] = []
    
    def __init__(self, faiss_database: FaissDatabase = None):
        super().__init__()
        self.faiss_database = faiss_database
    
    def __call__(self, corpus_id: str = None) -> Dict[str, Any]:
        """Execute the stats operation."""
        return self.faiss_database.get_stats(corpus_id=corpus_id)


class FaissToolkit(Toolkit):
    """
    Toolkit for FAISS vector database operations.
    
    This toolkit provides a comprehensive set of tools for interacting with FAISS vector databases,
    including semantic search, document insertion, deletion, and database management operations.
    
    The toolkit integrates with the existing RAG engine and storage infrastructure to provide
    a unified interface for vector database operations that can be easily used by agents.
    """
    
    def __init__(
        self,
        name: str = "FaissToolkit",
        storage_config: Optional[StoreConfig] = None,
        rag_config: Optional[RAGConfig] = None,
        default_corpus_id: str = "default",
        default_index_type: str = "vector",
        db_path: Optional[str] = None,
        storage_handler: StorageHandler = None,
        file_handler: FileStorageHandler = None,
        **kwargs
    ):
        """
        Initialize the FAISS toolkit.
        
        Args:
            name (str): Name of the toolkit
            storage_config (StoreConfig, optional): Configuration for storage backends
            rag_config (RAGConfig, optional): Configuration for RAG pipeline
            default_corpus_id (str): Default corpus ID for operations
            default_index_type (str): Default index type for vector operations
            db_path (str, optional): Custom database path. If provided, will check for existing database or create new one
            storage_handler (StorageHandler, optional): Storage handler for file operations
            file_handler (FileStorageHandler, optional): File handler for file operations
            **kwargs: Additional arguments
        """
        # Use default configurations if not provided
        if storage_config is None:
            storage_config = _create_default_storage_config(db_path)
        
        if rag_config is None:
            rag_config = _create_default_rag_config()
        
        # Create the shared FAISS database instance
        faiss_database = FaissDatabase(
            storage_config=storage_config,
            rag_config=rag_config,
            default_corpus_id=default_corpus_id,
            default_index_type=default_index_type,
            storage_handler=storage_handler,
            file_handler=file_handler
        )
        
        # Create tools with the shared database instance
        tools = [
            FaissQueryTool(faiss_database),
            FaissInsertTool(faiss_database),
            FaissDeleteTool(faiss_database),
            FaissListTool(faiss_database),
            FaissStatsTool(faiss_database)
        ]
        
        super().__init__(name=name, tools=tools, **kwargs)
        
        # Set instance variables after super().__init__()
        self.faiss_database = faiss_database
        
        logger.info(f"Initialized {name} with {len(tools)} tools")
    
    def get_database(self) -> FaissDatabase:
        """
        Get the underlying FAISS database instance.
        
        Returns:
            FaissDatabase: The FAISS database instance
        """
        return self.faiss_database 