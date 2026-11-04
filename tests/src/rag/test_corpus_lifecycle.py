"""Offline integration regressions for the vector corpus lifecycle."""
import asyncio
from unittest.mock import patch

import pytest
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode

from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.rag_config import RAGConfig, EmbeddingConfig, RetrievalConfig
from evoagentx.rag.schema import Corpus, Query, TextChunk
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import DBConfig, StoreConfig, VectorStoreConfig
from evoagentx.storages.vectore_stores.faiss import FaissVectorStoreWrapper


class LocalEmbedding(BaseEmbedding):
    """Deterministic embedding with an explicit failure trigger."""
    def _get_query_embedding(self, query):
        return [0.0, 0.0]

    async def _aget_query_embedding(self, query):
        await asyncio.sleep(0)
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text):
        if "FAIL_EMBED" in text:
            raise RuntimeError("embedding unavailable")
        return [5.0, 0.0]


class LocalEmbeddingWrapper:
    dimensions = 2

    def __init__(self):
        self.model = LocalEmbedding(model_name="text-embedding-ada-002")

    def get_embedding_model(self):
        return self.model

    def validate_model(self, provider, model_name):
        return model_name == "text-embedding-ada-002"


@pytest.fixture
def engine(tmp_path):
    handler = StorageHandler(storageConfig=StoreConfig(
        path=str(tmp_path),
        dbConfig=DBConfig(path=str(tmp_path / "corpora.sqlite")),
        vectorConfig=VectorStoreConfig(dimensions=2),
    ))
    config = RAGConfig(
        embedding=EmbeddingConfig(api_key="offline", dimensions=2),
        retrieval=RetrievalConfig(top_k=3, similarity_cutoff=None),
    )
    with patch("evoagentx.rag.rag.EmbeddingFactory.create", return_value=LocalEmbeddingWrapper()):
        rag = RAGEngine(config=config, storage_handler=handler)
    yield rag
    handler.storageDB.connection.close()


def node(identifier, text, distance=1.0, **metadata):
    return TextNode(id_=identifier, text=text, embedding=[distance, 0.0], metadata=metadata)


def contents(engine, corpus, top_k=10):
    result = engine.query(Query(query_str="lookup", top_k=top_k), corpus_id=corpus)
    return [chunk.text for chunk in result.corpus.chunks]


def test_top_k_is_applied_within_corpus(engine):
    engine.add("vector", [node("far", "target", 10)], corpus_id="target")
    engine.add("vector", [node("near", "unrelated", 0)], corpus_id="other")
    assert contents(engine, "target", 1) == ["target"]


def test_same_chunk_id_in_different_corpora_has_independent_lifecycle(engine):
    engine.add("vector", [node("shared", "alpha", 1)], corpus_id="a")
    engine.add("vector", [node("shared", "beta", 2)], corpus_id="b")
    engine.delete("a", "vector", node_ids="shared")
    assert contents(engine, "b") == ["beta"]
    assert contents(engine, "a") == []


def test_failed_embedding_preserves_previous_node(engine):
    engine.add("vector", [node("shared", "original")], corpus_id="a")
    replacement = TextNode(id_="shared", text="FAIL_EMBED")
    assert engine.add("vector", [replacement], corpus_id="a") == []
    assert contents(engine, "a") == ["original"]
    chunks = asyncio.run(engine.aget("a", "vector", ["shared"]))
    assert [chunk.text for chunk in chunks] == ["original"]


def test_delete_then_add_does_not_reuse_live_faiss_id(engine):
    engine.add("vector", [node("a", "first", 1), node("b", "second", 2)], corpus_id="a")
    engine.delete("a", "vector", node_ids="a")
    engine.add("vector", [node("c", "third", 3)], corpus_id="a")
    assert contents(engine, "a") == ["second", "third"]


def test_duplicate_ids_in_one_batch_have_one_last_value(engine):
    ids = engine.add("vector", [node("a", "old", 0), node("a", "new", 2)], corpus_id="a")
    assert ids == ["a"]
    assert contents(engine, "a") == ["new"]


def test_add_preserves_caller_nodes_and_uses_corpus_identity(engine):
    original = node("a", "text", 1, corpus_id="original")
    engine.add("vector", [original], corpus_id="new")
    assert original.metadata["corpus_id"] == "original"
    corpus = Corpus(chunks=[TextChunk(chunk_id="b", text="corpus text")], corpus_id="chosen")
    assert engine.add("vector", corpus) == ["b"]
    assert contents(engine, "chosen") == ["corpus text"]


@pytest.mark.parametrize("database", [False, True])
def test_repeated_load_replaces_snapshot_without_duplicate_vectors(engine, tmp_path, database):
    engine.add("vector", [node("same", "alpha", 1)], corpus_id="a")
    engine.add("vector", [node("same", "beta", 2)], corpus_id="b")
    destination = None if database else str(tmp_path / "snapshots")
    engine.save(output_path=destination)
    engine.add("vector", [node("stale", "not in snapshot", 3)], corpus_id="a")
    for _ in range(2):
        assert sorted(engine.load(source=destination)) == ["same", "same"]
        assert contents(engine, "a") == ["alpha"]
        assert contents(engine, "b") == ["beta"]


def test_snapshot_identity_does_not_come_from_sanitized_filename(engine, tmp_path):
    for corpus, text in [("a/b", "slash"), ("a_b", "underscore")]:
        engine.add("vector", [node("id", text)], corpus_id=corpus)
    engine.save(output_path=str(tmp_path / "snapshots"))
    engine.clear()
    engine.load(source=str(tmp_path / "snapshots"))
    assert contents(engine, "a/b") == ["slash"]
    assert contents(engine, "a_b") == ["underscore"]


def test_clear_one_corpus_leaves_other_corpus_retrievable(engine):
    engine.add("vector", [node("same", "alpha")], corpus_id="a")
    engine.add("vector", [node("same", "beta")], corpus_id="b")
    engine.clear("a")
    assert contents(engine, "b") == ["beta"]


def test_vector_backend_failed_batch_has_no_partial_native_vectors():
    store = FaissVectorStoreWrapper(dimensions=2).get_vector_store()
    store.add([node("old", "old")])
    with pytest.raises(ValueError):
        store.add([node("new", "new"), TextNode(id_="bad", text="bad", embedding=[0.0])])
    from llama_index.core.vector_stores.types import VectorStoreQuery
    result = store.query(VectorStoreQuery(query_embedding=[0.0, 0.0], similarity_top_k=10))
    assert result.ids == ["old"]


def test_failed_new_corpus_is_not_registered(engine):
    assert engine.add("vector", [TextNode(id_="bad", text="FAIL_EMBED")], corpus_id="failed") == []
    assert "failed" not in engine.indices
    assert "failed" not in engine.retrievers


def test_backend_failure_rolls_back_vectors_docstore_and_cache(engine):
    engine.add("vector", [node("a", "original", 1), node("b", "untouched", 2)], corpus_id="a")
    index = engine.indices["a"]["vector"]
    original_insert = index.index.insert_nodes

    def fail_after_insert(nodes):
        original_insert(nodes)
        raise RuntimeError("write failed after native insertion")

    with patch.object(index.index, "insert_nodes", side_effect=fail_after_insert):
        assert engine.add("vector", [node("a", "replacement", 0), node("c", "new", 3)], corpus_id="a") == []
    assert contents(engine, "a") == ["original", "untouched"]
    assert [chunk.text for chunk in asyncio.run(index.get(["a", "b", "c"]))] == ["original", "untouched"]
    engine.add("vector", [node("c", "after rollback", 3)], corpus_id="a")
    assert contents(engine, "a") == ["original", "untouched", "after rollback"]


def test_invalid_file_snapshot_does_not_change_live_corpus(engine, tmp_path):
    import json
    engine.add("vector", [node("a", "saved")], corpus_id="a")
    directory = tmp_path / "snapshot"
    engine.save(output_path=str(directory))
    engine.add("vector", [node("b", "live")], corpus_id="a")
    metadata_path = next(directory.glob("*_metadata.json"))
    metadata = json.loads(metadata_path.read_text())
    metadata["dimension"] = 3
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="dimension"):
        engine.load(source=str(directory))
    assert contents(engine, "a") == ["saved", "live"]


def test_wrong_saved_model_is_rejected_before_mutation(engine, tmp_path):
    import json
    engine.add("vector", [node("a", "live")], corpus_id="a")
    directory = tmp_path / "snapshot"
    engine.save(output_path=str(directory))
    metadata_path = next(directory.glob("*_metadata.json"))
    metadata = json.loads(metadata_path.read_text())
    metadata["embedding_model_name"] = "other-model"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="model"):
        engine.load(source=str(directory))
    assert contents(engine, "a") == ["live"]


def test_unknown_save_index_selector_does_not_save_other_types(engine, tmp_path):
    engine.add("vector", [node("a", "live")], corpus_id="a")
    directory = tmp_path / "snapshot"
    engine.save(output_path=str(directory), index_type="graph")
    assert not directory.exists() or not list(directory.iterdir())


def test_database_preserves_multiple_index_snapshots(engine):
    # Exercise real storage with graph payloads, without starting external Neo4j.
    handler = engine.storage_handler
    handler.save_index({"corpus_id": "a", "content": {"chunks": [{"text": "vector"}]},
                        "metadata": {"index_type": "vector"}})
    handler.save_index({"corpus_id": "a", "content": {"chunks": [{"text": "graph"}]},
                        "metadata": {"index_type": "graph"}})
    vector = handler.load_index("a", index_type="vector")
    graph = handler.load_index("a", index_type="graph")
    assert vector["content"]["chunks"] == [{"text": "vector"}]
    assert graph["content"]["chunks"] == [{"text": "graph"}]


def test_reranking_duplicate_ids_keeps_both_corpus_contents(engine):
    engine.config.retrieval.similarity_cutoff = 0.1
    engine.add("vector", [node("shared", "alpha", 1)], corpus_id="a")
    engine.add("vector", [node("shared", "beta", 2)], corpus_id="b")
    assert contents(engine, None) == ["alpha", "beta"]


def test_concurrent_top_k_requests_are_independent(engine):
    engine.add("vector", [node(str(i), str(i), i) for i in range(5)], corpus_id="a")
    retriever = engine.retrievers["a"]["vector"]

    async def run():
        return await asyncio.gather(*[
            retriever.aretrieve(Query(query_str="lookup", top_k=k)) for k in [1, 4, 2, 5]
        ])

    results = asyncio.run(run())
    assert [len(result.corpus.chunks) for result in results] == [1, 4, 2, 5]


def test_query_started_before_failed_write_sees_restored_catalog(engine):
    engine.add("vector", [node("a", "original")], corpus_id="a")
    index = engine.indices["a"]["vector"]
    original_insert = index.index.insert_nodes
    model = engine.embed_model.model

    def fail_after_insert(nodes):
        original_insert(nodes)
        raise RuntimeError("write failed")

    async def run():
        started, resume = asyncio.Event(), asyncio.Event()

        async def delayed_embedding(query):
            started.set()
            await resume.wait()
            return [0.0, 0.0]

        with patch.object(model, "_aget_query_embedding", side_effect=delayed_embedding):
            task = asyncio.create_task(engine.query_async(Query(query_str="lookup", top_k=1), "a"))
            await started.wait()
            with patch.object(index.index, "insert_nodes", side_effect=fail_after_insert):
                assert engine.add("vector", [node("a", "uncommitted")], corpus_id="a") == []
            resume.set()
            return await task

    result = asyncio.run(run())
    assert [chunk.text for chunk in result.corpus.chunks] == ["original"]


def test_native_add_failure_after_mutation_is_undone(engine):
    engine.add("vector", [node("a", "original")], corpus_id="a")
    backend = engine.indices["a"]["vector"].vector_store.get_vector_store()
    native = backend.client
    original = native.add_with_ids
    attempted = False

    def fail_once(vectors, ids):
        nonlocal attempted
        result = original(vectors, ids)
        if not attempted:
            attempted = True
            raise RuntimeError("native add failed after mutation")
        return result

    with patch.object(native, "add_with_ids", side_effect=fail_once):
        assert engine.add("vector", [node("a", "replacement"), node("b", "new")], corpus_id="a") == []
    assert contents(engine, "a") == ["original"]


def test_native_delete_failure_after_mutation_is_undone(engine):
    engine.add("vector", [node("a", "original"), node("b", "untouched", 2)], corpus_id="a")
    native = engine.indices["a"]["vector"].vector_store.get_vector_store().client
    original = native.remove_ids
    attempted = False

    def fail_once(ids):
        nonlocal attempted
        result = original(ids)
        if not attempted:
            attempted = True
            raise RuntimeError("native deletion failed after mutation")
        return result

    with patch.object(native, "remove_ids", side_effect=fail_once):
        with pytest.raises(RuntimeError, match="native deletion"):
            engine.delete("a", "vector", node_ids="a")
    assert contents(engine, "a") == ["original", "untouched"]


@pytest.mark.parametrize("database", [False, True])
def test_image_snapshot_preserves_path_embedding_and_relationships(engine, tmp_path, database):
    from evoagentx.rag.schema import ImageChunk
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
    engine.config.modality = "multimodal"
    engine.chunk_class = ImageChunk
    image = ImageChunk(
        chunk_id="image", image_path=str(tmp_path / "image.png"),
        image_mimetype="image/png", embedding=[1.0, 0.0],
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="source")},
        text_template="{content}",
    )
    assert engine.add("vector", [image], corpus_id="images") == ["image"]
    destination = None if database else str(tmp_path / "images")
    engine.save(output_path=destination)
    engine.clear()
    assert engine.load(source=destination) == ["image"]
    loaded = asyncio.run(engine.aget("images", "vector", ["image"]))[0]
    assert isinstance(loaded, ImageChunk)
    assert loaded.image_path == str(tmp_path / "image.png")
    assert loaded.embedding == [1.0, 0.0]
    assert loaded.relationships[NodeRelationship.SOURCE].node_id == "source"
    assert loaded.text_template == "{content}"


def test_legacy_file_snapshot_uses_metadata_identity(engine, tmp_path):
    import json
    directory = tmp_path / "legacy"
    directory.mkdir()
    chunk = TextChunk(chunk_id="old", text="legacy", embedding=[1.0, 0.0], text_template="{content}")
    (directory / "sanitized_vector_nodes.jsonl").write_text(chunk.to_json(indent=None) + "\n")
    (directory / "sanitized_vector_metadata.json").write_text(json.dumps({
        "corpus_id": "original/identity", "index_type": "vector", "dimension": 2,
        "embedding_model_name": "text-embedding-ada-002", "vector_db_type": "faiss",
    }))
    assert engine.load(source=str(directory), corpus_id="original/identity") == ["old"]
    assert contents(engine, "original/identity") == ["legacy"]
    loaded = asyncio.run(engine.aget("original/identity", "vector", ["old"]))[0]
    assert loaded.text_template == "{content}"


def test_failed_docstore_update_restores_reference_document_membership(engine):
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
    first, second = node("a", "first"), node("b", "second", 2)
    for value in (first, second):
        value.relationships = {NodeRelationship.SOURCE: RelatedNodeInfo(node_id="source")}
    engine.add("vector", [first, second], corpus_id="a")
    index = engine.indices["a"]["vector"]
    original = index.index.insert_nodes

    def fail_after_insert(nodes):
        original(nodes)
        raise RuntimeError("catalog failed")

    with patch.object(index.index, "insert_nodes", side_effect=fail_after_insert):
        assert engine.add("vector", [node("a", "replacement")], corpus_id="a") == []
    assert contents(engine, "a") == ["first", "second"]
    assert sorted(index.index.docstore.get_ref_doc_info("source").node_ids) == ["a", "b"]


def test_trained_ivf_template_isolated_corpora_keep_training_and_stable_ids(engine):
    import numpy as np
    template = FaissVectorStoreWrapper(dimensions=2, metrics="ivf_flat")
    template.get_vector_store().client.train(
        np.random.default_rng(123).normal(size=(4000, 2)).astype("float32")
    )
    engine.storage_handler.vector_store = template
    assert engine.add("vector", [node("a", "first", 1), node("b", "second", 2)], corpus_id="a") == ["a", "b"]
    assert engine.add("vector", [node("a", "other", 0)], corpus_id="b") == ["a"]
    for corpus in ["a", "b"]:
        backend = engine.indices[corpus]["vector"].vector_store.get_vector_store()
        import faiss
        faiss.extract_index_ivf(backend.client).nprobe = 100
    engine.delete("a", "vector", node_ids="a")
    engine.add("vector", [node("c", "third", 3)], corpus_id="a")
    assert contents(engine, "a") == ["second", "third"]
    assert contents(engine, "b") == ["other"]


@pytest.mark.parametrize("selection", [{"node_ids": []}, {"metadata_filters": {}}, {"index_type": ""}])
def test_empty_delete_selection_does_not_clear_the_corpus(engine, selection):
    engine.add("vector", [node("a", "keep")], corpus_id="a")
    engine.delete("a", **selection)
    assert contents(engine, "a") == ["keep"]


def test_empty_corpus_selector_does_not_clear_every_corpus(engine):
    engine.add("vector", [node("a", "keep")], corpus_id="a")
    engine.clear("")
    assert contents(engine, "a") == ["keep"]


def test_saving_loaded_legacy_directory_can_be_loaded_again(engine, tmp_path):
    import json
    directory = tmp_path / "legacy"
    directory.mkdir()
    chunk = TextChunk(chunk_id="a", text="legacy", embedding=[1.0, 0.0])
    (directory / "a_vector_nodes.jsonl").write_text(chunk.to_json(indent=None) + "\n")
    (directory / "a_vector_metadata.json").write_text(json.dumps({
        "corpus_id": "a", "index_type": "vector", "dimension": 2,
        "embedding_model_name": "text-embedding-ada-002", "vector_db_type": "faiss",
    }))
    engine.load(source=str(directory))
    engine.add("vector", [node("b", "new", 2)], corpus_id="a")
    engine.save(output_path=str(directory))
    engine.clear()
    assert engine.load(source=str(directory)) == ["a", "b"]
    assert contents(engine, "a") == ["legacy", "new"]


def test_failed_file_publication_keeps_previous_snapshot(engine, tmp_path):
    from evoagentx.rag import snapshots
    directory = tmp_path / "snapshot"
    engine.add("vector", [node("a", "saved")], corpus_id="a")
    engine.save(output_path=str(directory))
    engine.add("vector", [node("b", "unpublished", 2)], corpus_id="a")
    original_write = snapshots._atomic_write

    def fail_metadata(path, payload):
        if path.name.endswith("_metadata.json"):
            raise OSError("metadata publication failed")
        original_write(path, payload)

    with patch.object(snapshots, "_atomic_write", side_effect=fail_metadata):
        with pytest.raises(OSError, match="publication failed"):
            engine.save(output_path=str(directory))
    assert engine.load(source=str(directory)) == ["a"]
    assert contents(engine, "a") == ["saved"]


def test_unrelated_async_query_and_heartbeat_finish_before_ingestion_is_released(engine):
    from threading import Event, Thread, Timer
    engine.add("vector", [node("old", "query me")], corpus_id="existing")
    started, release = Event(), Event()
    original_embedding = engine.embed_model.model._get_text_embedding
    inserted = []

    def delayed_embedding(text):
        started.set()
        if not release.wait(5):
            raise RuntimeError("embedding watchdog expired")
        return original_embedding(text)

    def ingest():
        inserted.extend(engine.add("vector", [TextNode(id_="new", text="slow embedding")], corpus_id="other"))

    async def proof():
        async def heartbeat():
            await asyncio.sleep(0.01)
            return not release.is_set()

        heartbeat_task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        result = await engine.query_async("lookup", corpus_id="existing")
        query_finished_before_release = not release.is_set()
        heartbeat_finished_before_release = await heartbeat_task
        return result, query_finished_before_release, heartbeat_finished_before_release

    with patch.object(engine.embed_model.model, "_get_text_embedding", side_effect=delayed_embedding):
        worker = Thread(target=ingest)
        worker.start()
        watchdog = Timer(3, release.set)
        watchdog.start()
        try:
            assert started.wait(2)
            result, query_progressed, heartbeat_progressed = asyncio.run(proof())
        finally:
            release.set()
            watchdog.cancel()
            worker.join(5)
            watchdog.join(5)
    assert not worker.is_alive()
    assert [chunk.text for chunk in result.corpus.chunks] == ["query me"]
    assert query_progressed, "Query could finish only after the blocked embedding was released"
    assert heartbeat_progressed, "The query blocked an independent event-loop heartbeat"
    assert inserted == ["new"]


@pytest.mark.parametrize("operation", ["add", "delete", "clear", "load"])
def test_lifecycle_writers_serialize_with_an_inflight_add(engine, operation):
    from threading import Event, Thread
    engine.add("vector", [node("old", "saved", 1)], corpus_id="a")
    engine.save()
    started, release, following_started = Event(), Event(), Event()
    original_embedding = engine.embed_model.model._get_text_embedding
    errors = []

    def delayed_embedding(text):
        started.set()
        if not release.wait(5):
            raise RuntimeError("embedding watchdog expired")
        return original_embedding(text)

    def ingest():
        try:
            engine.add("vector", [TextNode(id_="pending", text="slow embedding")], corpus_id="a")
        except BaseException as error:
            errors.append(error)

    def following():
        following_started.set()
        try:
            if operation == "add":
                engine.add("vector", [node("second", "second", 2)], corpus_id="a")
            elif operation == "delete":
                engine.delete("a", "vector", node_ids="pending")
            elif operation == "clear":
                engine.clear()
            else:
                engine.load()
        except BaseException as error:
            errors.append(error)

    with patch.object(engine.embed_model.model, "_get_text_embedding", side_effect=delayed_embedding):
        first, second = Thread(target=ingest), Thread(target=following)
        first.start()
        try:
            assert started.wait(2)
            second.start()
            assert following_started.wait(2)
        finally:
            release.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    if operation == "add":
        assert contents(engine, "a") == ["saved", "second", "slow embedding"]
    elif operation in ("delete", "load"):
        assert contents(engine, "a") == ["saved"]
    else:
        assert engine.indices == {} and engine.retrievers == {}
