"""
FAISS vector store + JSON metadata sidecar.

Stores:
  - FAISS index on disk  (data/faiss_index/index.faiss)
  - Chunk metadata list   (data/faiss_index/metadata.json)
  - Document manifest      (data/documents.json)
"""

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

import config

log = logging.getLogger(__name__)

INDEX_FILE = config.FAISS_INDEX_DIR / "index.faiss"
META_FILE = config.FAISS_INDEX_DIR / "metadata.json"


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass
class ChunkMeta:
    doc_id: str
    filename: str
    page: int
    chunk_id: int
    text: str
    image_path: str  # relative path to page PNG
    # ── Phase 1: structured metadata for legal-PDF retrieval ──
    page_start: int = 0
    page_end: int = 0
    chunk_type: str = "generic"        # article|clause|heading|footnote|amendment|generic
    article_id: str = ""               # e.g. "21", "21A"
    schedule_id: str = ""              # e.g. "VII"
    list_id: str = ""                  # e.g. "I", "II", "III"
    entry_id: str = ""                 # e.g. "1", "11A"
    section_id: str = ""               # for IPC/CrPC sections
    source_extraction: str = ""        # sarvam_ocr | pymupdf


@dataclass
class DocRecord:
    doc_id: str
    filename: str
    file_hash: str
    pages: int
    chunk_count: int
    ingestion_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── In-memory state ──────────────────────────────────────────────────────

_index: faiss.IndexFlatIP | None = None
_metadata: list[dict] = []
_docs: list[dict] = []


def _ensure_loaded():
    """Load index + metadata from disk if not yet loaded."""
    global _index, _metadata, _docs

    if _index is not None:
        return

    if INDEX_FILE.exists():
        log.info("Loading FAISS index from %s", INDEX_FILE)
        _index = faiss.read_index(str(INDEX_FILE))
        _metadata = json.loads(META_FILE.read_text(encoding="utf-8-sig"))

        # ── Consistency guard ──────────────────────────────────────────
        if _index.ntotal != len(_metadata):
            log.warning(
                "FAISS/metadata MISMATCH: index has %d vectors but metadata has %d entries. "
                "Truncating to the smaller count to prevent IndexError.",
                _index.ntotal, len(_metadata),
            )
            safe_count = min(_index.ntotal, len(_metadata))
            _metadata = _metadata[:safe_count]
            if _index.ntotal > safe_count:
                # Rebuild index from scratch with only the valid vectors
                # This is rare — only happens after a crash mid-save
                log.warning("Rebuilding FAISS index to match metadata (%d entries)", safe_count)
                new_index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
                if safe_count > 0:
                    vecs = faiss.rev_swig_ptr(_index.get_xb(), _index.ntotal * _index.d)
                    vecs = np.reshape(vecs, (_index.ntotal, _index.d))[:safe_count]
                    new_index.add(np.ascontiguousarray(vecs))
                _index = new_index
    else:
        log.info("Creating new FAISS index (dim=%d)", config.EMBEDDING_DIMENSION)
        _index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
        _metadata = []

    if config.DOCUMENTS_JSON.exists():
        _docs = json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8-sig"))
    else:
        _docs = []

    log.info("Loaded: %d vectors, %d metadata entries, %d doc records",
             _index.ntotal, len(_metadata), len(_docs))


def _atomic_json_write(path: Path, data):
    """Write JSON atomically: write to temp file then rename over target."""
    content = json.dumps(data, ensure_ascii=False, indent=2)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem
    )
    try:
        with open(tmp_fd, "w", encoding="utf-8") as f:
            f.write(content)
        Path(tmp_path).replace(path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def save():
    """Persist index + metadata to disk."""
    _ensure_loaded()
    if _index.ntotal != len(_metadata):
        log.error(
            "REFUSING to save: FAISS has %d vectors but metadata has %d entries",
            _index.ntotal, len(_metadata),
        )
        raise RuntimeError(
            f"FAISS/metadata length mismatch: {_index.ntotal} vs {len(_metadata)}"
        )
    faiss.write_index(_index, str(INDEX_FILE))
    _atomic_json_write(META_FILE, _metadata)
    _atomic_json_write(config.DOCUMENTS_JSON, _docs)
    log.info("Saved FAISS index (%d vectors) and %d metadata entries", _index.ntotal, len(_metadata))


# ── Public API ────────────────────────────────────────────────────────────


def add_chunks(vectors: np.ndarray, metas: list[ChunkMeta]):
    """Add chunk vectors + metadata to the store (call save() after)."""
    _ensure_loaded()
    assert vectors.shape[0] == len(metas), "vectors and metas length mismatch"
    _index.add(vectors)
    _metadata.extend([asdict(m) for m in metas])
    log.info("Added %d chunks — total now %d", len(metas), _index.ntotal)


def add_doc_record(rec: DocRecord):
    """Append a document record to the manifest."""
    _ensure_loaded()
    _docs.append(asdict(rec))


def search(query_vec: np.ndarray, top_k: int = config.TOP_K) -> list[tuple[dict, float]]:
    """
    Search FAISS index. Returns list of (metadata_dict, score).
    query_vec shape: (1, dim)
    """
    _ensure_loaded()
    if _index.ntotal == 0:
        return []

    k = min(top_k, _index.ntotal)
    scores, indices = _index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append((_metadata[idx], float(score)))
    return results


def doc_already_ingested(file_hash: str) -> bool:
    """Check if a document with this hash was already ingested WITH chunks > 0."""
    _ensure_loaded()
    return any(
        d.get("file_hash") == file_hash and d.get("chunk_count", 0) > 0
        for d in _docs
    )


def get_total_chunks() -> int:
    _ensure_loaded()
    return _index.ntotal


def get_all_metadata() -> list[dict]:
    _ensure_loaded()
    return list(_metadata)


def remove_doc_record(file_hash: str):
    """Remove any doc records matching this hash (e.g. failed ingestions with 0 chunks)."""
    _ensure_loaded()
    global _docs
    before = len(_docs)
    _docs = [d for d in _docs if d.get("file_hash") != file_hash]
    removed = before - len(_docs)
    if removed:
        log.info("Removed %d stale doc record(s) for hash %s", removed, file_hash)


def get_doc_records() -> list[dict]:
    _ensure_loaded()
    return list(_docs)
