import os
import json
from copy import deepcopy
from threading import RLock
from typing import List, Dict, Any, Optional, Union

from pydantic import Field, PrivateAttr

from ..core.module import BaseModule
from .storages_config import StoreConfig
from .db_stores import DBStoreBase, DBStoreFactory
from .graph_stores import GraphStoreFactory, GraphStoreBase
from .vectore_stores import VectorStoreFactory, VectorStoreBase
from .schema import TableType, AgentStore, WorkflowStore, MemoryStore, HistoryStore, IndexStore, expand_index_record


class StorageHandler(BaseModule):
    """
    Implementation of a storage handler for managing various storage backends.
    
    StorageHandler provides an abstraction for reading and writing data (e.g., memory, agents, workflows).
    It supports multiple storage types, including database, vector, and graph storage, initialized via factories.
    """
    storageConfig: StoreConfig = Field(..., description="Configuration for all storage backends")
    storageDB: Optional[Union[DBStoreBase, Any]] = Field(None, description="Database storage backend")
    vector_store: Optional[Union[VectorStoreBase, Any]] = Field(None, description="Single vector storage backend")
    graph_store: Optional[Union[GraphStoreBase, Any]] = Field(None, description="Optional graph storage backend")

    _index_lock: Any = PrivateAttr(default_factory=RLock)

    def init_module(self):
        """
        Initialize all storage backends based on the provided configuration.
        Calls individual initialization methods for database, vector, and graph stores.
        """
        # Create the path
        if (self.storageConfig.path is not None) or (self.storageConfig.path != ":memory:") \
            or (not self.storageConfig.path):
            os.makedirs(os.path.dirname(self.storageConfig.path), exist_ok=True)
        
        self._init_db_store()
        self._init_vector_store()
        self._init_graph_store()
    
    def _init_db_store(self):
        """
        Initialize the database storage backend using the DBStoreFactory.
        Sets the storageDB attribute with the created instance.
        """
        db_config = self.storageConfig.dbConfig
        self.storageDB = DBStoreFactory.create(db_config.db_name, db_config)
    
    def _init_vector_store(self):
        """
        Initialize the vector storage backend using the VectorStoreFactory.
        Sets the storageVector attribute if the configuration is provided.
        """
        vector_config = self.storageConfig.vectorConfig
        if vector_config is not None:
            if self.vector_store is not None:
                del self.vector_store

            vector_config_dict = vector_config.model_dump()
            self.vector_store = VectorStoreFactory().create(
                store_type=vector_config.vector_name,
                store_config=vector_config_dict
            )
    
    def _init_graph_store(self):
        """
        Initialize the graph storage backend using the GraphStoreFactory.
        Sets the storageGraph attribute if the configuration is provided.
        """
        graph_config = self.storageConfig.graphConfig
        if graph_config is not None:
            self.graph_store = GraphStoreFactory().create(
                store_type=graph_config.graph_name,
                store_config=graph_config.model_dump()
            )

    def load(self, tables: Optional[List[str]] = None, *args, **kwargs) -> Dict[str, Any]:
        """
        Load all data from the database storage.

        Attributes:
            tables (Optional[List[str]]): List of table names to load; if None, loads all tables.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary with table names as keys and lists of records as values. You should parse the values by yourself.
        """
        result = {}
        table_info = self.storageDB.col_info()
        
        if tables is None:
            tables_to_load = [t.value for t in TableType]
        else:
            tables_to_load = tables

        # Load data for each table
        for table_name in tables_to_load:
            table_data = []
            # Check if the table exists
            if any(t["table_name"] == table_name for t in table_info):
                cursor = self.storageDB.connection.cursor()
                cursor.execute(f"SELECT * FROM {table_name}")
                # Get column names from the columns dictionary
                columns = next(t["columns"].keys() for t in table_info if t["table_name"] == table_name)
                rows = cursor.fetchall()
                table_data = [dict(zip(columns, row)) for row in rows]
            result[table_name] = table_data
        
        return result

    def save(self, data: Dict[str, Any], *args, **kwargs):
        """
        Save all provided data to the database storage.

        Attributes:
            data (Dict[str, Any]): Dictionary with table names as keys and lists of records to save.

        Raises:
            ValueError: If an unknown table name is provided.
        """
        for table_name, records in data.items():
            store_type = None
            # Map table name to store_type
            for st in TableType:
                if st.value == table_name:
                    store_type = st
                    break
            if store_type is None:
                raise ValueError(f"Unknown table: {table_name}")
            # Insert each record
            for record in records:
                self.storageDB.insert(metadata=record, store_type=store_type, table=table_name)

    def parse_result(self, results: Dict[str, str], 
                     store: Union[AgentStore, WorkflowStore, MemoryStore, HistoryStore]) -> Dict[str, Any]:
        """
        Parse database results, converting JSON strings to Python objects where applicable.

        Attributes:
            results (Dict[str, str]): Raw database results with column names as keys.
            store (Union[AgentStore, WorkflowStore, MemoryStore, HistoryStore]): Pydantic model for validation.

        Returns:
            Dict[str, Any]: Parsed results with JSON strings deserialized to Python objects.
        """
        for k, v in store.model_fields.items():
            if v.annotation not in [Optional[str], str]:
                try:
                    results[k] = json.loads(results[k])
                except (json.JSONDecodeError, KeyError, TypeError):
                    results[k] = results.get(k)
        return results

    def load_memory(self, memory_id: str, table: Optional[str]=None, **kwargs) -> Dict[str, Any]:
        """
        Load a single long-term memory data.

        Attributes:
            memory_id (str): The ID of the long-term memory.
            table (Optional[str]): The table name; defaults to 'memory' if None.

        Returns:
            Dict[str, Any]: The data that can be used to create a LongTermMemory instance.
        """
        table = table or TableType.store_memory.value
        result = self.storageDB.get_by_id(memory_id, store_type="memory", table=table)
        if result is not None:
            result = self.parse_result(result, MemoryStore)
        return result

    def save_memory(self, memory_data: Dict[str, Any], table: Optional[str]=None, **kwargs):
        """
        Save or update a single memory.

        Attributes:
            memory_data (Dict[str, Any]): The long-term memory's data.
            table (Optional[str]): The table name; defaults to 'memory' if None.

        """
        table = table or TableType.store_memory.value
        memory_id = memory_data.get("memory_id")
        if not memory_id:
            raise ValueError("Memory data must include a 'memory_id' field")
        existing = self.storageDB.get_by_id(memory_id, store_type="memory", table=table)
        if existing:
            self.storageDB.update(memory_id, new_metadata=memory_data, store_type="memory", table=table)
        else:
            self.storageDB.insert(metadata=memory_data, store_type="memory", table=table)

    def load_agent(self, agent_name: str, table: Optional[str]=None, *args, **kwargs) -> Dict[str, Any]:
        """
        Load a single agent's data.

        Attributes:
            agent_name (str): The unique name of the agent to retrieve.
            table (Optional[str]): The table name; defaults to 'agent' if None.

        Returns:
            Dict[str, Any]: The data that can be used to create an Agent instance, or None if not found.
        """
        table = table or TableType.store_agent.value
        result = self.storageDB.get_by_id(agent_name, store_type="agent", table=table)
        # Parse the result to convert JSON strings to Python objects
        if result is not None:
            result = self.parse_result(result, AgentStore)
        return result

    def remove_agent(self, agent_name: str, table: Optional[str]=None, *args, **kwargs):
        """
        Remove an agent from storage if the agent exists.

        Attributes:
            agent_name (str): The name of the agent to be deleted.
            table (Optional[str]): The table name; defaults to 'agent' if None.

        Raises:
            ValueError: If the agent does not exist in the specified table.
        """
        table = table or TableType.store_agent.value
        success = self.storageDB.delete(agent_name, store_type="agent", table=table)
        if not success:
            raise ValueError(f"Agent with name {agent_name} not found in table {table}")

    def save_agent(self, agent_data: Dict[str, Any], table: Optional[str]=None, *args, **kwargs):
        """
        Save or update a single agent's data.

        Attributes:
            agent_data (Dict[str, Any]): The agent's data, must include 'name' and 'content' keys.
            table (Optional[str]): The table name; defaults to 'agent' if None.

        Raises:
            ValueError: If 'name' field is missing or if Pydantic validation fails.
        """
        table = table or TableType.store_agent.value
        agent_name = agent_data.get("name")
        if not agent_name:
            raise ValueError("Agent data must include a 'name' field")
        
        existing = self.storageDB.get_by_id(agent_name, store_type="agent", table=table)
        if existing:
            self.storageDB.update(agent_name, new_metadata=agent_data, store_type="agent", table=table)
        else:
            self.storageDB.insert(metadata=agent_data, store_type="agent", table=table)

    def load_workflow(self, workflow_id: str, table: Optional[str] = None, *args, **kwargs) -> Dict[str, Any]:
        """
        Load a single workflow's data.

        Attributes:
            workflow_id (str): The ID of the workflow.
            table (Optional[str]): The table name; defaults to 'workflow' if None.

        Returns:
            Dict[str, Any]: The data that can be used to create a WorkFlow instance, or None if not found.
        """
        table = table or TableType.store_workflow.value
        result = self.storageDB.get_by_id(workflow_id, store_type="workflow", table=table)
        # Parse the result to convert JSON strings to Python objects
        if result is not None:
            result = self.parse_result(result, WorkflowStore)
        return result

    def save_workflow(self, workflow_data: Dict[str, Any], table: Optional[str] = None, *args, **kwargs):
        """
        Save or update a workflow's data.

        Attributes:
            workflow_data (Dict[str, Any]): The workflow's data, must include 'name' field.
            table (Optional[str]): The table name; defaults to 'workflow' if None.

        Raises:
            ValueError: If 'name' field is missing or if Pydantic validation fails.
        """
        table = table or TableType.store_workflow.value
        workflow_id = workflow_data.get("name")
        if not workflow_id:
            raise ValueError("Workflow data must include a 'name' field")
        # Check if workflow exists to decide between insert or update
        existing = self.storageDB.get_by_id(workflow_id, store_type="workflow", table=table)
        if existing:

            self.storageDB.update(workflow_id, new_metadata=workflow_data, store_type="workflow", table=table)
        else:
            self.storageDB.insert(metadata=workflow_data, store_type="workflow", table=table)

    def load_history(self, memory_id: str, table: Optional[str] = None, *args, **kwargs) -> Dict[str, Any]:
        """
        Load a single history entry.

        Attributes:
            memory_id (str): The ID of the memory associated with the history entry.
            table (Optional[str]): The table name; defaults to 'history' if None.

        Returns:
            Dict[str, Any]: The history data, or None if not found.
        """
        table = table or TableType.store_history.value
        result = self.storageDB.get_by_id(memory_id, store_type="history", table=table)
        # Parse the result to convert JSON strings to Python objects (if any)
        if result is not None:
            result = self.parse_result(result, HistoryStore)
        return result

    def save_history(self, history_data: Dict[str, Any], table: Optional[str] = None, *args, **kwargs):
        """
        Save or update a single history entry.

        Attributes:
            history_data (Dict[str, Any]): The history data, must include 'memory_id' field.
            table (Optional[str]): The table name; defaults to 'history' if None.

        Raises:
            ValueError: If 'memory_id' field is missing or if Pydantic validation fails.
        """
        table = table or TableType.store_history.value
        memory_id = history_data.get("memory_id")
        if not memory_id:
            raise ValueError("History data must include a 'memory_id' field")
        # Check if history entry exists to decide between insert or update
        existing = self.storageDB.get_by_id(memory_id, store_type="history", table=table)
        if existing:
            # parse the history, then change the old_hisotry
            result = HistoryStore.model_validate(self.parse_result(existing, HistoryStore))
            history_data["old_memory"] = result.old_memory
            self.storageDB.update(memory_id, new_metadata=history_data, store_type="history", table=table)
        else:
            self.storageDB.insert(metadata=history_data, store_type="history", table=table)

    def load_index(
        self, corpus_id: str, table: Optional[str] = None,
        index_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load a corpus row, optionally selecting one index from its snapshots.

        Existing single-index rows remain readable without migration. An
        unfiltered call returns the stored row, including an aggregate when
        several index types share the corpus primary key.
        """
        table = table or TableType.store_indexing.value
        with self._index_lock:
            result = self.storageDB.get_by_id(corpus_id, store_type="indexing", table=table)
            if result is None:
                return None
            result = self.parse_result(result, IndexStore)
            if index_type is None:
                return result
            return next(
                (record for record in expand_index_record(result)
                 if record.get("metadata", {}).get("index_type") == index_type),
                None,
            )

    def save_index(self, index_data: Dict[str, Any], table: Optional[str] = None):
        """Upsert one index without overwriting other types in the same corpus.

        A single index retains the legacy row format. When a second type is
        saved, both payloads are stored in a versioned content envelope using
        the existing corpus_id primary key and columns. Updating one type
        preserves the other snapshots, including those not loaded by an engine.
        """
        table = table or TableType.store_indexing.value
        data = IndexStore.model_validate(deepcopy(index_data)).model_dump()
        corpus_id = data["corpus_id"]
        if not corpus_id:
            raise ValueError("Index data must include a nonempty corpus_id")
        index_type = data["metadata"].get("index_type")
        with self._index_lock:
            existing = self.load_index(corpus_id, table=table)
            if existing is not None and index_type is not None:
                records = {
                    record["metadata"].get("index_type"): record
                    for record in expand_index_record(existing)
                }
                records[index_type] = data
                if len(records) > 1:
                    data = {
                        "corpus_id": corpus_id,
                        "content": {"indices": records},
                        "metadata": {"snapshot_format": 2},
                        "date": data["date"],
                    }
            elif existing is not None and existing.get("metadata", {}).get("snapshot_format") == 2:
                raise ValueError("index_type is required to update an aggregate snapshot")
            if existing is None:
                self.storageDB.insert(metadata=data, store_type="indexing", table=table)
            else:
                self.storageDB.update(corpus_id, new_metadata=data, store_type="indexing", table=table)
