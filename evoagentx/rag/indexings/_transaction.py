"""A delta journal for the in-memory catalogs used by vector indexing."""
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Iterator, Optional, Tuple

from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore
from llama_index.core.storage.kvstore.types import DEFAULT_COLLECTION


class JournaledKVStore(SimpleKVStore):
    """Undo writes to node, reference-document and index catalog records.

    Only changed keys are retained. In particular, adding a node does not copy
    the text/metadata of every existing node. Serialized index records contain
    immutable JSON strings, so journaling them does not duplicate that payload.
    Reads are detached because LlamaIndex may update nested reference-node lists
    while preparing a subsequent put; a shallow read would change live state
    before the journal had a chance to capture it.
    """

    def __init__(self):
        super().__init__()
        self._journal: Optional[Dict[Tuple[str, str], Any]] = None

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Restore every changed record if the enclosing vector write fails."""
        if self._journal is not None:
            yield
            return
        self._journal = {}
        try:
            yield
        except BaseException:
            for (collection, key), previous in reversed(list(self._journal.items())):
                if previous is None:
                    super().delete(key, collection=collection)
                else:
                    super().put(key, previous, collection=collection)
            raise
        finally:
            self._journal = None

    def _remember(self, key: str, collection: str) -> None:
        """Capture a key's original value once per transaction."""
        identity = (collection, key)
        if self._journal is not None and identity not in self._journal:
            self._journal[identity] = deepcopy(super().get(key, collection=collection))

    def get(self, key: str, collection: str = DEFAULT_COLLECTION):
        """Return detached data so nested caller changes cannot bypass journaling."""
        return deepcopy(super().get(key, collection=collection))

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        """Record the prior value before publishing a catalog record."""
        self._remember(key, collection)
        super().put(key, deepcopy(val), collection=collection)

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Record a deletion before removing the catalog record."""
        self._remember(key, collection)
        return super().delete(key, collection=collection)
