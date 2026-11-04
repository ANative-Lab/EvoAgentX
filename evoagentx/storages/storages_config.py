from typing import Optional

from pydantic import Field

from evoagentx.core.base_config import BaseConfig


class DBConfig(BaseConfig):
    """
    Defines settings for connecting to a database, such as SQLite or PostgreSQL.
    """
    db_name: str = Field(default="sqlite", description="Name of the database provider (e.g., 'sqlite', 'posgre_sql')")
    path: Optional[str] = Field(default="", description="File path for file-based databases (e.g., SQLite)")
    ip: Optional[str] = Field(default="", description="IP address for network-based databases")
    port: Optional[str] = Field(default="", description="Port for network-based databases")


class VectorStoreConfig(BaseConfig):
    """
    Configuration for vector databases, supporting FAISS, Qdrant, and Milvus.
    """
    vector_name: str = Field(default="faiss", description="Name of the vector database provider (e.g., 'faiss', 'qdrant', 'milvus')")
    dimensions: Optional[int] = Field(default=1536, description="Dimension of the embedding vectors")
    index_type: Optional[str] = Field(default="flat_l2", description="Index type for FAISS (e.g., 'flat_l2', 'ivf_flat')")
    qdrant_url: Optional[str] = Field(default=None, description="URL for Qdrant server (e.g., 'http://localhost:6333')")
    qdrant_api_key: Optional[str] = Field(default=None, description="API key for Qdrant authentication")
    qdrant_collection_name: Optional[str] = Field(default="default_collection", description="Name of the Qdrant collection")
    milvus_uri: Optional[str] = Field(default="./milvus.db", description="Milvus URI. Use a local .db path for Milvus Lite or an HTTP endpoint for Milvus server/Zilliz Cloud.")
    milvus_token: Optional[str] = Field(default=None, description="Token for Milvus server or Zilliz Cloud authentication")
    milvus_db_name: Optional[str] = Field(default=None, description="Milvus database name")
    milvus_collection_name: Optional[str] = Field(default="evoagentx_vectors", description="Name of the Milvus collection")
    milvus_metric_type: Optional[str] = Field(default="IP", description="Milvus vector metric type, such as IP, COSINE, or L2")
    milvus_consistency_level: Optional[str] = Field(default="Session", description="Milvus consistency level")
    milvus_overwrite: Optional[bool] = Field(default=False, description="Drop and recreate the Milvus collection during initialization")


class GraphStoreConfig(BaseConfig):
    """
    Configuration for graph databases, supporting Neo4j.
    """
    graph_name: str = Field(default="neo4j", description="Name of the graph database provider (e.g., 'neo4j')")
    uri: Optional[str] = Field(default=None, description="URI for Neo4j database (e.g., 'bolt://localhost:7687')")
    username: Optional[str] = Field(default=None, description="Username for Neo4j authentication")
    password: Optional[str] = Field(default=None, description="Password for Neo4j authentication")
    database: Optional[str] = Field(default="neo4j", description="Name of the Neo4j database")
    max_retries: Optional[int] = Field(default=3, description="Maximum number of connection retries")
    timeout: Optional[float] = Field(default=30.0, description="Connection timeout in seconds")


class StoreConfig(BaseConfig):
    """
    Aggregates database, vector, file, and graph store configurations.
    """
    dbConfig: DBConfig = Field(..., description="Configuration for the database store")
    vectorConfig: Optional[VectorStoreConfig] = Field(None, description="Configuration for the vector store")
    graphConfig: Optional[GraphStoreConfig] = Field(None, description="Optional configuration for the graph store")
    # For file storage
    path: Optional[str] = Field(default="/index_cache", description="directory path for storing the index")
