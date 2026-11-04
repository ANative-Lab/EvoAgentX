from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class TableType(str, Enum):
    """
    Enum representing the default table types for the database.
    Each value corresponds to a specific table name used for storing different types of data.
    """
    store_agent = "agent"      # Table for agent data
    store_workflow = "workflow" # Table for workflow data
    store_memory = "memory"    # Table for memory data
    store_history = "history"  # Table for history data
    store_indexing = "indexing"   # Table for indexing data


class MemoryStore(BaseModel):
    """
    Stores memory-related metadata with optional fields for keywords, entities, and embeddings.
    """
    memory_id: str = Field(..., description="Unique identifier for the memory")      
    content: str = Field(..., description="Main content of the memory")
    date: str = Field(..., description="Date associated with the memory")
    key_words: Optional[List] = Field([], description="Optional list of keywords associated with the memory")
    entity_content: Optional[Dict[str, Any]] = Field({}, description="Optional dictionary of entity-related content")
    embedding: Optional[List] = Field([], description="Optional list of embedding vectors")

# Pydantic model for workflow data storage
class WorkflowStore(BaseModel):
    """
    Stores workflow metadata with a unique name and content dictionary.
    """
    name: str = Field(..., description="Unique workflow identifier")
    content: Dict[str, Any] = Field(..., description="Dictionary containing workflow details") 
    date: Optional[str] = Field("", description="Date associated with the workflow") 

# Pydantic model for agent data storage
class AgentStore(BaseModel):
    """
    Stores agent metadata with a unique name and content dictionary.
    """
    name: str = Field(..., description="Unique agent identifier")
    content: Dict[str, Any] = Field(..., description="Dictionary containing agent details")
    date: Optional[str] = Field("", description="Date associated with the agent")

# Pydantic model for history data storage
class HistoryStore(BaseModel):
    """
    Stores changes to memory with event details and timestamps.
    """
    memory_id: str = Field(..., description="Identifier of the memory being modified")
    old_memory: str = Field(..., description="Original memory content before change")
    new_memory: str = Field(..., description="Updated memory content after change")
    event: str = Field(..., description="Description of the event causing the change")
    created_at: Optional[str] = Field("", description="Optional timestamp for creation")
    updated_at: Optional[str] = Field("", description="Optional timestamp for last update")

# Pydantic model for index data storage
class IndexStore(BaseModel):
    """
    Stores indexing metadata with a unique id and basic chunk/node attribute.
    """
    corpus_id: str = Field(..., description="Identifier for the associated corpus")
    content: Dict[str, Any] = Field(..., description="Serialized index content (e.g., LlamaIndex JSON)")
    date: Optional[str] = Field(default="", description="Creation or last update date")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata, e.g., vector store collection name")


def expand_index_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand an aggregate DB row, retaining support for old single-index rows."""
    if record.get("metadata", {}).get("snapshot_format") != 2:
        return [record]
    indices = record.get("content", {}).get("indices")
    if not isinstance(indices, dict):
        raise ValueError("Aggregate snapshot must contain an indices object")
    records = []
    for index_type, payload in indices.items():
        if not isinstance(payload, dict):
            raise ValueError("Aggregate index payload must be an object")
        single = dict(payload, corpus_id=record["corpus_id"])
        if single.get("metadata", {}).get("index_type") != index_type:
            raise ValueError("Aggregate index key disagrees with its metadata")
        if single["metadata"].get("corpus_id", record["corpus_id"]) != record["corpus_id"]:
            raise ValueError("Aggregate index belongs to a different corpus")
        records.append(single)
    return records
