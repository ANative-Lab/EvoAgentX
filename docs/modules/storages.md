# StorageHandler Documentation

## Overview
The `StorageHandler` class is a critical component designed to manage multiple storage backends for storing and retrieving various types of data, such as agent configurations, workflows, memory entries, and index data. It provides a unified interface to interact with different storage systems, including relational databases (e.g., SQLite), vector databases (e.g., FAISS), and graph databases (e.g., Neo4j). The class leverages the Pydantic library for configuration validation and uses factory patterns to initialize storage backends.

The `StorageHandler` is tightly integrated with the `RAGEngine` class to support retrieval-augmented generation (RAG) functionality by managing the storage of indexed documents, embeddings, and associated metadata. It abstracts the complexity of interacting with different storage systems, ensuring seamless data operations for applications like long-term memory management and RAG pipelines.

## Class Structure
The `StorageHandler` class inherits from `BaseModule` and uses Pydantic's `Field` for configuration and type validation. It supports three types of storage backends:
- **Database Storage** (`storageDB`): Manages relational database operations, such as SQLite, for structured data storage.
- **Vector Storage** (`vector_store`): Handles vector embeddings for semantic search, supporting providers like FAISS.
- **Graph Storage** (`graph_store`): Manages graph-based data, such as Neo4j, for relational or networked data structures.

### Key Attributes
- `storageConfig: StoreConfig`: Configuration object for all storage backends, defined in `storages_config.py`. It includes settings for database, vector, and graph stores.
- `storageDB: Optional[Union[DBStoreBase, Any]]`: Instance of the database storage backend, initialized via `DBStoreFactory`.
- `vector_store: Optional[Union[VectorStoreBase, Any]]`: Instance of the vector storage backend, initialized via `VectorStoreFactory`.
- `graph_store: Optional[Union[GraphStoreBase, Any]]`: Instance of the graph storage backend, initialized via `GraphStoreFactory`.

### Dependencies
- **Pydantic**: For configuration validation and type checking.
- **Factory Patterns**: `DBStoreFactory`, `VectorStoreFactory`, and `GraphStoreFactory` for creating storage backend instances.
- **Configuration**: `StoreConfig`, `DBConfig`, `VectorStoreConfig`, and `GraphStoreConfig` from `storages_config.py` for defining storage settings.
- **Schema**: `TableType`, `AgentStore`, `WorkflowStore`, `MemoryStore`, `HistoryStore`, and `IndexStore` for data validation and structure.

## Key Methods

### Initialization
- **`init_module(self)`**
  - Initializes all storage backends based on the provided `storageConfig`.
  - Creates the storage directory if specified and initializes database, vector, and graph stores by calling their respective initialization methods.

- **`_init_db_store(self)`**
  - Initializes the database storage backend using `DBStoreFactory` with the `dbConfig` from `storageConfig`.
  - Sets the `storageDB` attribute.

- **`_init_vector_store(self)`**
  - Initializes the vector storage backend using `VectorStoreFactory` if `vectorConfig` is provided.
  - Sets the `vector_store` attribute.

- **`_init_graph_store(self)`**
  - Initializes the graph storage backend using `GraphStoreFactory` if `graphConfig` is provided.
  - Sets the `graph_store` attribute.

### Data Operations
- **`load(self, tables: Optional[List[str]] = None, *args, **kwargs) -> Dict[str, Any]`**
  - Loads data from the database storage for specified tables or all tables defined in `TableType`.
  - Returns a dictionary with table names as keys and lists of records as values.
  - Each record is a dictionary mapping column names to values, requiring manual parsing for JSON fields.

- **`save(self, data: Dict[str, Any], *args, **kwargs)`**
  - Saves data to the database storage.
  - Takes a dictionary with table names as keys and lists of records to save.
  - Validates table names against `TableType` and inserts records using `storageDB.insert`.

- **`parse_result(self, results: Dict[str, str], store: Union[AgentStore, WorkflowStore, MemoryStore, HistoryStore]) -> Dict[str, Any]`**
  - Parses raw database results, deserializing JSON strings into Python objects based on the provided Pydantic model (`store`).
  - Returns a dictionary with parsed results, handling non-string fields appropriately.

### Entity-Specific Operations
- **`load_memory(self, memory_id: str, table: Optional[str]=None, **kwargs) -> Dict[str, Any]`**
  - Placeholder method for loading a single long-term memory entry by `memory_id`.
  - Defaults to the `memory` table if no table is specified.

- **`save_memory(self, memory_data: Dict[str, Any], table: Optional[str]=None, **kwargs)`**
  - Placeholder method for saving or updating a single memory entry.
  - Defaults to the `memory` table if no table is specified.

- **`load_agent(self, agent_name: str, table: Optional[str]=None, *args, **kwargs) -> Dict[str, Any]`**
  - Loads a single agent's data by `agent_name` from the database.
  - Defaults to the `agent` table if no table is specified.
  - Parses the result using `parse_result` with `AgentStore` for validation.
  - Returns `None` if the agent is not found.

- **`remove_agent(self, agent_name: str, table: Optional[str]=None, *args, **kwargs)`**
  - Deletes an agent by `agent_name` from the specified table (defaults to `agent`).
  - Raises a `ValueError` if the agent does not exist.

- **`save_agent(self, agent_data: Dict[str, Any], table: Optional[str]=None, *args, **kwargs)`**
  - Saves or updates an agent's data in the database.
  - Requires `agent_data` to include a `name` field.
  - Updates existing records or inserts new ones using `storageDB.update` or `storageDB.insert`.

- **`load_workflow(self, workflow_id: str, table: Optional[str]=None, *args, **kwargs) -> Dict[str, Any]`**
  - Loads a single workflow's data by `workflow_id` from the database.
  - Defaults to the `workflow` table if no table is specified.
  - Parses the result using `parse_result` with `WorkflowStore` for validation.
  - Returns `None` if the workflow is not found.

- **`save_workflow(self, workflow_data: Dict[str, Any], table: Optional[str]=None, *args, **kwargs)`**
  - Saves or updates a workflow's data in the database.
  - Requires `workflow_data` to include a `name` field.
  - Updates existing records or inserts new ones using `storageDB.update` or `storageDB.insert`.

- **`load_history(self, memory_id: str, table: Optional[str]=None, *args, **kwargs) -> Dict[str, Any]`**
  - Loads a single history entry by `memory_id` from the database.
  - Defaults to the `history` table if no table is specified.
  - Parses the result using `parse_result` with `HistoryStore` for validation.
  - Returns `None` if the history entry is not found.

- **`save_history(self, history_data: Dict[str, Any], table: Optional[str]=None, *args, **kwargs)`**
  - Saves or updates a single history entry in the database.
  - Requires `history_data` to include a `memory_id` field.
  - Updates existing records with `old_memory` preserved or inserts new ones.

- **`load_index(self, corpus_id: str, table: Optional[str]=None) -> Optional[Dict[str, Any]]`**
  - Loads index data by `corpus_id` from the database.
  - Parses the result using `parse_result` with `IndexStore` for validation.
  - Returns `None` if the index is not found.

- **`save_index(self, index_data: Dict[str, Any], table: Optional[str]=None)`**
  - Saves or updates index data in the database.
  - Requires `index_data` to include a `corpus_id` field.
  - Updates existing records or inserts new ones using `storageDB.update` or `storageDB.insert`.

## Integration with RAGEngine
The `StorageHandler` is tightly integrated with the `RAGEngine` class to support RAG functionality. It is used to:
- **Initialize Vector Storage**: The `RAGEngine` constructor checks the vector store's dimensions against the embedding model's dimensions and reinitializes the vector store if necessary.
- **Save Indices**: The `save` method in `RAGEngine` uses `StorageHandler.save_index` to persist index data (e.g., corpus chunks and metadata) to the database when no file output path is specified.
- **Load Indices**: The `load` method in `RAGEngine` uses `StorageHandler.load` and `StorageHandler.parse_result` to reconstruct indices from database records, ensuring compatibility with embedding models and dimensions.

## Configuration
The `StorageHandler` relies on the `StoreConfig` class (defined in `storages_config.py`) to configure its backends:
- **`DBConfig`**: Configures relational databases (e.g., SQLite) with settings like `db_name`, `path`, `ip`, and `port`.
- **`VectorStoreConfig`**: Configures vector databases (e.g., FAISS, Qdrant, Milvus) with settings like `vector_name`, `dimensions`, `index_type`, `qdrant_url`, `qdrant_collection_name`, `milvus_uri`, and `milvus_collection_name`.
- **`GraphStoreConfig`**: Configures graph databases (e.g., Neo4j) with settings上午 like `graph_name`, `uri`, `username`, `password`, and `database`.

The configuration is validated using Pydantic, ensuring robust type checking and default values.

## Usage Example
Below is an example of how to initialize and use `StorageHandler`:

```python
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import StoreConfig, DBConfig, VectorStoreConfig

# Define configuration
config = StoreConfig(
    dbConfig=DBConfig(db_name="sqlite", path="data/storage.db"),
    vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1536),
    path="data/index_cache"
)

# Initialize StorageHandler
storage_handler = StorageHandler(storageConfig=config)
storage_handler.init_module()

# Save agent data
agent_data = {"name": "agent1", "content": {"role": "analyst", "tasks": ["data analysis"]}}
storage_handler.save_agent(agent_data)

# Load agent data
agent = storage_handler.load_agent("agent1")
print(agent)  # {'name': 'agent1', 'content': {'role': 'analyst', 'tasks': ['data analysis']}}

# Save index data (used in RAGEngine)
index_data = {
    "corpus_id": "corpus1",
    "content": {"chunks": [{"chunk_id": "c1", "text": "Sample text", "metadata": {}}]},
    "metadata": {"index_type": "VECTOR", "dimension": 1536}
}
storage_handler.save_index(index_data)

# Load index data
index = storage_handler.load_index("corpus1")
print(index)  # {'corpus_id': 'corpus1', 'content': {...}, 'metadata': {...}}
```

To use Milvus instead of FAISS, install the optional dependency and set `vector_name="milvus"`:

```bash
pip install "evoagentx[milvus]"
```

```python
config = StoreConfig(
    dbConfig=DBConfig(db_name="sqlite", path="data/storage.db"),
    vectorConfig=VectorStoreConfig(
        vector_name="milvus",
        dimensions=1536,
        milvus_uri="./data/evoagentx_milvus.db",
        milvus_collection_name="evoagentx_vectors",
        milvus_metric_type="IP",
    ),
    path="data/index_cache"
)
```

`milvus_uri` can point to a Milvus Lite `.db` file, a self-hosted Milvus endpoint such as `http://localhost:19530`, or a Zilliz Cloud endpoint when paired with `milvus_token`.

## Notes
- The `load_memory` and `save_memory` methods are not yet fully implemented and will be developed alongside `LongTermMemory`.
- The `StorageHandler` assumes the database schema is managed by `DBStoreBase` and its factory, ensuring compatibility with `TableType` enums.
- When used with `RAGEngine`, ensure the vector store's dimensions match the embedding model's dimensions to avoid reinitialization issues.
- Error handling is implemented throughout, with logs generated via the `evoagentx.core.logging.logger` module.

## Conclusion
The `StorageHandler` class provides a flexible and extensible interface for managing multiple storage backends in a unified manner. Its integration with `RAGEngine` makes it a key component for RAG pipelines, enabling efficient storage and retrieval of indexed data. By leveraging factory patterns and Pydantic validation, it ensures robustness and scalability for applications requiring complex data management.
