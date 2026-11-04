"""Portable corpus snapshots independent of vector-backend process state."""
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from .schema import Corpus, ImageChunk, IndexMetadata, TextChunk


@dataclass
class IndexSnapshot:
    """A validated logical index payload, rather than a live backend handle."""

    metadata: IndexMetadata
    corpus: Corpus

    def to_record(self) -> Dict[str, Any]:
        """Build the existing database record format without redundant indexes."""
        return {
            "corpus_id": self.metadata.corpus_id,
            "content": {
                "corpus_id": self.metadata.corpus_id,
                "chunks": [chunk.model_dump(mode="json") for chunk in self.corpus.chunks],
            },
            "metadata": self.metadata.model_dump(mode="json", exclude_none=True),
            "date": self.metadata.date or "",
        }


def chunk_from_record(data: Dict[str, Any]):
    """Reconstruct text or image chunks without losing formatting/relationships."""
    if not isinstance(data, dict):
        raise ValueError("Snapshot chunks must be JSON objects")
    if not isinstance(data.get("chunk_id"), str) or not data["chunk_id"]:
        raise ValueError("Snapshot chunks must contain a nonempty chunk_id")
    common = {
        key: data[key]
        for key in (
            "chunk_id", "metadata", "embedding", "excluded_embed_metadata_keys",
            "excluded_llm_metadata_keys", "text_template", "relationships",
        )
        if key in data
    }
    if "image_path" in data:
        return ImageChunk(
            image_path=data["image_path"],
            image_mimetype=data.get("image_mimetype"),
            **common,
        )
    if "text" not in data:
        raise ValueError("Snapshot chunks must contain text or image_path")
    return TextChunk(
        text=data["text"],
        start_char_idx=data.get("start_char_idx"),
        end_char_idx=data.get("end_char_idx"),
        **common,
    )


def snapshot_from_record(record: Dict[str, Any]) -> IndexSnapshot:
    """Read a legacy or current single-index database record."""
    metadata = IndexMetadata.model_validate(record["metadata"])
    if metadata.corpus_id != record["corpus_id"]:
        raise ValueError("Snapshot corpus_id disagrees with its metadata")
    content = record["content"]
    if not isinstance(content, dict) or not isinstance(content.get("chunks"), list):
        raise ValueError("Snapshot content must contain a chunks list")
    if content.get("corpus_id", metadata.corpus_id) != metadata.corpus_id:
        raise ValueError("Snapshot content belongs to a different corpus")
    chunks = [chunk_from_record(data) for data in content["chunks"]]
    identifiers = [chunk.chunk_id for chunk in chunks]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Snapshot contains duplicate chunk IDs")
    return IndexSnapshot(
        metadata=metadata,
        corpus=Corpus(chunks=chunks, corpus_id=metadata.corpus_id, metadata=metadata),
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    """Publish one complete file and remove temporary files after failures."""
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".rag-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_file_snapshot(snapshot: IndexSnapshot, directory: str) -> None:
    """Publish a collision-free metadata pointer to immutable JSONL content.

    Write content first and metadata last. If metadata publication fails, a
    previous snapshot still points to its intact content. Unreferenced content
    is removed on failure; older committed generations remain valid for readers
    that already opened the previous metadata file.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    identity = json.dumps(
        [snapshot.metadata.corpus_id, snapshot.metadata.index_type], ensure_ascii=False
    ).encode("utf-8")
    stem = hashlib.sha256(identity).hexdigest()
    metadata_file = destination / f"{stem}_metadata.json"
    nodes_file = destination / f"{stem}_{uuid4().hex}_nodes.jsonl"
    nodes = "".join(
        json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n"
        for chunk in snapshot.corpus.chunks
    ).encode("utf-8")
    metadata = snapshot.metadata.model_dump(mode="json", exclude_none=True)
    metadata.update(
        snapshot_version=2,
        nodes_file=nodes_file.name,
        nodes_sha256=hashlib.sha256(nodes).hexdigest(),
    )
    _atomic_write(nodes_file, nodes)
    try:
        _atomic_write(
            metadata_file,
            json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8"),
        )
    except BaseException:
        nodes_file.unlink(missing_ok=True)
        raise


def load_file_snapshots(
    directory: str,
    corpus_id: Optional[str] = None,
    index_type: Optional[str] = None,
) -> Iterable[IndexSnapshot]:
    """Read both legacy filename pairs and versioned metadata pointers.

    Logical identity always comes from metadata, including for legacy files
    whose names contain sanitized corpus IDs. A pointer cannot escape the
    selected directory. Selected corrupt/incomplete snapshots fail explicitly.
    """
    source = Path(directory)
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory {directory} does not exist")
    selected = {}
    for metadata_file in sorted(source.glob("*_metadata.json")):
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if (corpus_id is not None and metadata.get("corpus_id") != corpus_id) or (
            index_type is not None and metadata.get("index_type") != index_type
        ):
            continue
        version = metadata.get("snapshot_version", 1)
        if version not in (1, 2):
            raise ValueError(f"Unsupported snapshot version: {version}")
        identity = (metadata.get("corpus_id"), metadata.get("index_type"))
        previous = selected.get(identity)
        if previous is not None:
            old_version = previous[1].get("snapshot_version", 1)
            if old_version == version:
                raise ValueError(f"Multiple snapshots found for {identity}")
            # Saving into a legacy directory publishes a canonical versioned
            # pointer. Retain old files for readers already using them, but
            # prefer the new pointer when both formats describe the same index.
            if old_version > version:
                continue
        selected[identity] = (metadata_file, metadata)
    for metadata_file, metadata in selected.values():
        version = metadata.get("snapshot_version", 1)
        nodes_name = metadata.get("nodes_file")
        if nodes_name is None and version == 1:
            nodes_name = metadata_file.name[:-len("_metadata.json")] + "_nodes.jsonl"
        if not isinstance(nodes_name, str) or Path(nodes_name).name != nodes_name:
            raise ValueError("Snapshot nodes_file must be a filename in the source directory")
        nodes_file = source / nodes_name
        if nodes_file.resolve().parent != source.resolve():
            raise ValueError("Snapshot nodes_file must remain in the source directory")
        nodes = nodes_file.read_bytes()
        checksum = metadata.get("nodes_sha256")
        if version == 2 and (not checksum or hashlib.sha256(nodes).hexdigest() != checksum):
            raise ValueError("Snapshot nodes checksum does not match metadata")
        chunks = [json.loads(line) for line in nodes.decode("utf-8").splitlines() if line.strip()]
        yield snapshot_from_record({
            "corpus_id": metadata["corpus_id"],
            "metadata": metadata,
            "content": {"chunks": chunks},
        })
