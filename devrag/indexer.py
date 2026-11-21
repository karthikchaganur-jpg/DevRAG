from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .chunking import split_chunks
from .config import EMBEDDING_MODEL, MANIFEST_NAMES, MAX_FILE_BYTES, SKIP_DIRS, TEXT_EXTS
from .embeddings import EmbeddingsUnavailable, LocalEmbedder
from .store import IndexStore


def _chunk_id(repo_id: str, chunk: dict[str, Any], ordinal: int) -> str:
    value = f"{repo_id}\0{chunk['path']}\0{ordinal}\0{chunk['start']}\0{chunk['text']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _embedding_text(chunk: dict[str, Any]) -> str:
    symbol = f" symbol: {chunk['symbol']}" if chunk.get("symbol") else ""
    return f"{chunk['path']}{symbol}\n{chunk['text']}"


def _vectors_or_none(embedder: LocalEmbedder, chunks: list[dict[str, Any]]) -> tuple[list[list[float]] | None, str | None]:
    if not chunks:
        return [], None
    try:
        vectors = []
        texts = [_embedding_text(chunk) for chunk in chunks]
        for start in range(0, len(texts), 64):
            vectors.extend(embedder.embed_passages(texts[start:start + 64]))
        return vectors, None
    except EmbeddingsUnavailable as exc:
        return None, str(exc)


def index_repository(root: Path, repo_id: str, name: str, store: IndexStore,
                     embedder: LocalEmbedder) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[float]]]:
    root = root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(400, f"Folder not found: {root}")

    store.ensure_repository(repo_id, name, str(root), None)
    existing = store.get_files(repo_id)
    seen: set[str] = set()
    changed: dict[str, tuple[os.stat_result, str, int, list[dict[str, Any]]]] = {}
    unchanged_paths: list[str] = []
    file_count = 0
    embedding_error: str | None = None
    scan_errors: list[str] = []

    def record_walk_error(error: OSError) -> None:
        scan_errors.append(str(error))

    for folder, dirs, names in os.walk(root, onerror=record_walk_error):
        dirs[:] = [directory for directory in dirs if directory.lower() not in SKIP_DIRS and not directory.startswith(".")]
        for filename in names:
            path = Path(folder) / filename
            if path.suffix.lower() not in TEXT_EXTS and filename.lower() not in MANIFEST_NAMES:
                continue
            try:
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    continue
                stat = path.stat()
                if stat.st_size > MAX_FILE_BYTES:
                    continue
                relative = path.relative_to(root).as_posix()
                seen.add(relative)
                file_count += 1
                cached = existing.get(relative)
                if cached and cached["mtime_ns"] == stat.st_mtime_ns and cached["size"] == stat.st_size:
                    unchanged_paths.append(relative)
                    continue
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                scan_errors.append(str(path))
                continue
            if not content.strip():
                # Store an empty file record so it is not needlessly retried every scan.
                changed[relative] = (stat, hashlib.sha256(b"").hexdigest(), 0, [])
                continue
            file_chunks = split_chunks(Path(relative), content)
            for ordinal, chunk in enumerate(file_chunks):
                chunk["id"] = _chunk_id(repo_id, chunk, ordinal)
                chunk["ordinal"] = ordinal
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            changed[relative] = (stat, digest, len(content.splitlines()), file_chunks)

    # A partial scan must never make unreadable cached files look deleted.
    removed_files = 0 if scan_errors else store.remove_files_except(repo_id, seen)
    # Include old, unembedded chunks so installing the model later upgrades an
    # existing index without rereading or rechunking every unchanged file.
    pending_chunks: list[dict[str, Any]] = []
    pending_file_paths: list[str] = []
    for relative, (_, _, _, file_chunks) in changed.items():
        pending_chunks.extend(file_chunks)
        pending_file_paths.extend([relative] * len(file_chunks))
    for relative in unchanged_paths:
        if not store.embeddings_ready(repo_id, relative, EMBEDDING_MODEL):
            cached_chunks = store.get_file_chunks(repo_id, relative)
            for chunk in cached_chunks:
                pending_chunks.append(chunk)
                pending_file_paths.append(relative)

    vectors, embedding_error = _vectors_or_none(embedder, pending_chunks)
    if vectors is not None:
        for relative in unchanged_paths:
            if not store.embeddings_ready(repo_id, relative, EMBEDDING_MODEL):
                indexes = [i for i, pending_path in enumerate(pending_file_paths) if pending_path == relative]
                store.set_embeddings(repo_id, relative, [vectors[i] for i in indexes], EMBEDDING_MODEL)

    for relative, (stat, digest, line_count, file_chunks) in changed.items():
        indexes = [i for i, pending_path in enumerate(pending_file_paths) if pending_path == relative]
        file_vectors = [vectors[i] for i in indexes] if vectors is not None else None
        store.replace_file(repo_id, relative, stat.st_mtime_ns, stat.st_size, digest, line_count,
                           file_chunks, file_vectors, EMBEDDING_MODEL if file_vectors is not None else None)

    store.update_model(repo_id, EMBEDDING_MODEL if store.repository_has_embeddings(repo_id, EMBEDDING_MODEL) else None)

    files, chunks = store.load_repository(repo_id)
    chunk_count = len(chunks)
    vectors_by_id = {
        chunk["id"]: chunk["embedding"] for chunk in chunks
        if chunk.get("embedding") and chunk.get("embedding_model") == EMBEDDING_MODEL
    }
    for chunk in chunks:
        chunk.pop("embedding", None)
        chunk.pop("embedding_model", None)
    info = {
        "files": file_count,
        "chunks": chunk_count,
        "root": str(root),
        "added_or_changed": len(changed),
        "unchanged": len(unchanged_paths),
        "removed": removed_files,
        "semantic_search": bool(vectors_by_id),
        "embedding_model": EMBEDDING_MODEL if vectors_by_id else None,
        "embedding_error": embedding_error,
        "scan_incomplete": bool(scan_errors),
        "scan_warning": f"Could not read {len(scan_errors)} file or folder location(s); existing index entries were kept." if scan_errors else None,
    }
    return info, files, chunks, vectors_by_id
