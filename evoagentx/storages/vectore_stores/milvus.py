import json
import re
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import node_to_metadata_dict

from .base import VectorStoreBase
from evoagentx.core.logging import logger


DEFAULT_MILVUS_URI = "./milvus.db"
DEFAULT_COLLECTION_NAME = "evoagentx_vectors"
DEFAULT_METRIC_TYPE = "IP"
DEFAULT_CONSISTENCY_LEVEL = "Session"
DEFAULT_TEXT_ID_FIELD = "id"
DEFAULT_DOC_ID_FIELD = "doc_id"
DEFAULT_EMBEDDING_FIELD = "embedding"
MAX_VARCHAR_LENGTH = 65535

_VALID_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_FIELDS = {
    DEFAULT_TEXT_ID_FIELD,
    DEFAULT_DOC_ID_FIELD,
    DEFAULT_EMBEDDING_FIELD,
}


class MilvusVectorStore(BasePydanticVectorStore):
    """LlamaIndex-compatible Milvus vector store backed by MilvusClient."""

    stores_text: bool = False
    uri: str = Field(default=DEFAULT_MILVUS_URI)
    token: Optional[str] = Field(default=None)
    db_name: Optional[str] = Field(default=None)
    collection_name: str = Field(default=DEFAULT_COLLECTION_NAME)
    dimensions: int = Field(default=1536)
    metric_type: str = Field(default=DEFAULT_METRIC_TYPE)
    consistency_level: str = Field(default=DEFAULT_CONSISTENCY_LEVEL)
    overwrite: bool = Field(default=False)
    text_id_field: str = Field(default=DEFAULT_TEXT_ID_FIELD)
    doc_id_field: str = Field(default=DEFAULT_DOC_ID_FIELD)
    embedding_field: str = Field(default=DEFAULT_EMBEDDING_FIELD)

    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.metric_type = self.metric_type.upper()
        self._client = self._create_client()
        self._ensure_collection()

    @classmethod
    def class_name(cls) -> str:
        return "MilvusVectorStore"

    @property
    def client(self) -> Any:
        return self._client

    def _create_client(self) -> Any:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise ImportError(
                "Milvus vector store requires the optional dependency "
                "`pymilvus[milvus-lite]`. Install it with `pip install evoagentx[milvus]`."
            ) from exc

        return MilvusClient(
            uri=self.uri,
            token=self.token or "",
            db_name=self.db_name or "",
        )

    def _ensure_collection(self) -> None:
        if self.overwrite and self._client.has_collection(self.collection_name):
            self._client.drop_collection(self.collection_name)

        if self._client.has_collection(self.collection_name):
            self._validate_existing_collection()
            return

        from pymilvus import DataType

        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(
            field_name=self.text_id_field,
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=MAX_VARCHAR_LENGTH,
        )
        schema.add_field(
            field_name=self.embedding_field,
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dimensions,
        )
        schema.add_field(
            field_name=self.doc_id_field,
            datatype=DataType.VARCHAR,
            max_length=MAX_VARCHAR_LENGTH,
        )

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=self.embedding_field,
            index_type="AUTOINDEX",
            metric_type=self.metric_type,
        )

        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level=self.consistency_level,
        )

    def _validate_existing_collection(self) -> None:
        description = self._client.describe_collection(self.collection_name)
        fields = {field["name"]: field for field in description["fields"]}

        if self.text_id_field not in fields:
            raise ValueError(
                f"Milvus collection '{self.collection_name}' is missing primary field "
                f"'{self.text_id_field}'."
            )
        if self.embedding_field not in fields:
            raise ValueError(
                f"Milvus collection '{self.collection_name}' is missing vector field "
                f"'{self.embedding_field}'."
            )

        vector_dim = int(fields[self.embedding_field]["params"].get("dim", 0))
        if vector_dim != self.dimensions:
            raise ValueError(
                f"Milvus collection '{self.collection_name}' has dimension {vector_dim}, "
                f"but the configured dimension is {self.dimensions}."
            )

    def add(
        self,
        nodes: Sequence[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        rows = [self._node_to_row(node) for node in nodes]
        if rows:
            self._client.upsert(
                collection_name=self.collection_name,
                data=rows,
            )
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._client.delete(
            collection_name=self.collection_name,
            filter=f"{self.doc_id_field} == {self._format_value(ref_doc_id)}",
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        expressions = []
        if node_ids:
            ids = ", ".join(self._format_value(node_id) for node_id in node_ids)
            expressions.append(f"{self.text_id_field} in [{ids}]")
        if filters is not None:
            expressions.append(self._to_filter_expression(filters))

        if not expressions:
            return

        self._client.delete(
            collection_name=self.collection_name,
            filter=" and ".join(f"({expr})" for expr in expressions if expr),
        )

    def clear(self) -> None:
        if self._client.has_collection(self.collection_name):
            self._client.drop_collection(self.collection_name)
        self._ensure_collection()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError(f"Milvus vector store does not support query mode: {query.mode}")
        if query.query_embedding is None:
            raise ValueError("Milvus vector store requires a query embedding.")

        filters = []
        if query.node_ids:
            ids = ", ".join(self._format_value(node_id) for node_id in query.node_ids)
            filters.append(f"{self.text_id_field} in [{ids}]")
        if query.filters is not None:
            filters.append(self._to_filter_expression(query.filters))

        result = self._client.search(
            collection_name=self.collection_name,
            data=[query.query_embedding],
            anns_field=self.embedding_field,
            limit=query.similarity_top_k,
            filter=" and ".join(f"({expr})" for expr in filters if expr),
            output_fields=[self.text_id_field],
        )

        hits = result[0] if result else []
        ids = [str(hit.get("id", hit.get(self.text_id_field))) for hit in hits]
        similarities = [float(hit["distance"]) for hit in hits]
        return VectorStoreQueryResult(ids=ids, similarities=similarities)

    def _node_to_row(self, node: BaseNode) -> Dict[str, Any]:
        metadata = node_to_metadata_dict(node, remove_text=True, flat_metadata=False)
        metadata.pop("_node_content", None)

        row = {
            self.text_id_field: node.node_id,
            self.embedding_field: node.get_embedding(),
            self.doc_id_field: node.ref_doc_id or "None",
        }
        row.update(self._metadata_to_dynamic_fields(metadata))
        return row

    def _metadata_to_dynamic_fields(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        fields = {}
        for key, value in metadata.items():
            if key in _RESERVED_FIELDS or not _VALID_FIELD_NAME.match(key):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                fields[key] = value
        return fields

    def _to_filter_expression(self, filters: MetadataFilters) -> str:
        condition = filters.condition or FilterCondition.AND
        joiner = " and " if condition == FilterCondition.AND else " or "
        expressions = []

        for filter_item in filters.filters:
            if isinstance(filter_item, MetadataFilters):
                expressions.append(f"({self._to_filter_expression(filter_item)})")
                continue

            field = filter_item.key
            if not _VALID_FIELD_NAME.match(field) or field in _RESERVED_FIELDS:
                raise ValueError(f"Unsupported metadata filter field: {field}")
            expressions.append(
                self._metadata_filter_to_expression(
                    field=field,
                    operator=filter_item.operator,
                    value=filter_item.value,
                )
            )

        return joiner.join(expressions)

    def _metadata_filter_to_expression(
        self,
        field: str,
        operator: FilterOperator,
        value: Any,
    ) -> str:
        if operator == FilterOperator.EQ:
            return f"{field} == {self._format_value(value)}"
        if operator == FilterOperator.NE:
            return f"{field} != {self._format_value(value)}"
        if operator == FilterOperator.GT:
            return f"{field} > {self._format_value(value)}"
        if operator == FilterOperator.GTE:
            return f"{field} >= {self._format_value(value)}"
        if operator == FilterOperator.LT:
            return f"{field} < {self._format_value(value)}"
        if operator == FilterOperator.LTE:
            return f"{field} <= {self._format_value(value)}"
        if operator == FilterOperator.IN:
            return f"{field} in {self._format_value_list(value)}"
        if operator == FilterOperator.NIN:
            return f"{field} not in {self._format_value_list(value)}"

        raise ValueError(f"Unsupported metadata filter operator for Milvus: {operator}")

    def _format_value_list(self, value: Any) -> str:
        if not isinstance(value, list):
            raise ValueError("Milvus 'in' filters require a list value.")
        return "[" + ", ".join(self._format_value(item) for item in value) + "]"

    def _format_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if value is None:
            return "null"
        return json.dumps(str(value))


class MilvusVectorStoreWrapper(VectorStoreBase):
    """Wrapper for Milvus vector store."""

    def __init__(
        self,
        dimensions: int = 1536,
        milvus_uri: str = DEFAULT_MILVUS_URI,
        milvus_token: Optional[str] = None,
        milvus_db_name: Optional[str] = None,
        milvus_collection_name: str = DEFAULT_COLLECTION_NAME,
        milvus_metric_type: str = DEFAULT_METRIC_TYPE,
        milvus_consistency_level: str = DEFAULT_CONSISTENCY_LEVEL,
        milvus_overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        self.vector_store = MilvusVectorStore(
            uri=milvus_uri or DEFAULT_MILVUS_URI,
            token=milvus_token,
            db_name=milvus_db_name,
            collection_name=milvus_collection_name or DEFAULT_COLLECTION_NAME,
            dimensions=dimensions,
            metric_type=milvus_metric_type or DEFAULT_METRIC_TYPE,
            consistency_level=milvus_consistency_level or DEFAULT_CONSISTENCY_LEVEL,
            overwrite=bool(milvus_overwrite),
        )

    def get_vector_store(self) -> MilvusVectorStore:
        return self.vector_store

    async def aload(self, node: BaseNode) -> None:
        self.vector_store.add([node])
        logger.info(f"Inserted node with ID {node.node_id} into Milvus vector store.")
