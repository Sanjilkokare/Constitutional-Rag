"""
Text chunker - splits OCR-extracted text into chunks.

Two strategies:
  1. Legal-structure-aware: splits on Article/Part boundaries first,
     preserving legal identifiers. Used when the text contains
     recognizable Article headings.
  2. Generic paragraph-based: splits on paragraph boundaries with
     configurable size and overlap. Fallback for non-legal PDFs.
"""

import logging
import re
from typing import Optional

import config
import legal_index as li

log = logging.getLogger(__name__)

_FOOTNOTE_PATTERNS = re.compile(
    r"(?:"
    r"Subs\.\s+by|Ins\.\s+by|Omitted\s+by|Added\s+by|"
    r"w\.e\.f\.\s*\d|with\s+effect\s+from|"
    r"Amendment\s+Act,?\s*\d{4}|"
    r"^\d+\.\s+The\s+Constitution\s+\(|"
    r"^\d+\.\s+(?:Subs\.|Ins\.|Omitted|Added|Rep\.)"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

_SECTION_IPC = re.compile(
    r"(?:Section|Sec\.?|IPC|CrPC|Cr\.P\.C\.)\s+(\d+[A-Z]?)",
    re.IGNORECASE,
)

_CONTINUATION_BLOCKERS = re.compile(
    r"^(?:PART\s+[IVXLC]+(?:-[A-Z])?\b|CHAPTER\b|"
    r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|"
    r"ELEVENTH|TWELFTH)\s+SCHEDULE\b|SCHEDULE\b|"
    r"LIST\s+[IVXLC]+\b|UNION\s+LIST\b|STATE\s+LIST\b|CONCURRENT\s+LIST\b)",
    re.IGNORECASE,
)

_EDITORIAL_LINE = re.compile(
    r"^\s*(?:"
    r"\d+\.\s+(?:Subs\.|Ins\.|Omitted|Added|Rep\.|The\s+Constitution\s+\()|"
    r"(?:Subs\.|Ins\.|Omitted|Added|Rep\.)\s+by|"
    r"w\.e\.f\."
    r")",
    re.IGNORECASE,
)


def classify_chunk_type(text: str, segment_type: str) -> str:
    """
    Classify a chunk into: article, clause, heading, footnote, amendment, generic.

    Uses the segment_type from legal_index.segment_articles() as a base,
    then refines with pattern detection.
    """
    if segment_type in ("part_heading", "schedule_heading", "list_heading"):
        return "heading"

    if segment_type == "article":
        return "article"

    matches = list(_FOOTNOTE_PATTERNS.finditer(text))
    if matches:
        text_len = len(text)
        match_density = len(matches) / max(text.count("\n") + 1, 1)
        if match_density > 0.3 or (len(matches) >= 3 and text_len < 600):
            return "amendment"
        if len(matches) >= 1 and text_len < 300:
            return "footnote"

    if segment_type == "text_block":
        return "generic"

    return "generic"


def detect_section_id(text: str) -> str:
    """Extract IPC/CrPC section number if present."""
    m = _SECTION_IPC.search(text)
    return m.group(1) if m else ""


def chunk_page_legal(
    text: str,
    page_num: int,
    chunk_size: int = config.CHUNK_SIZE,
    min_size: int = config.CHUNK_MIN_SIZE,
    carry_article: Optional[str] = None,
    carry_heading: str = "",
    carry_schedule: str = "",
    carry_list: str = "",
    carry_entry: str = "",
) -> list[dict]:
    """
    Chunk a single page's text using legal structure.

    Returns list of dicts:
      {"text": ..., "article": ..., "page": page_num, "type": ...}

    If no article headings found, falls back to generic chunking
    and returns dicts with article=None.
    """
    segments = li.segment_schedule_entries(
        text,
        page_num,
        carry_schedule=carry_schedule,
        carry_list=carry_list,
        carry_entry=carry_entry,
    )
    if not segments:
        segments = li.segment_articles(text, page_num)

    if not segments:
        return []

    if carry_article and segments and not any(seg.get("schedule_id") or seg.get("list_id") for seg in segments):
        first = segments[0]
        if first["type"] == "text_block" and _looks_like_article_continuation(first["text"], min_size):
            first["article"] = carry_article
            first["heading"] = carry_heading
            first["type"] = "article"

    results = []
    for seg in segments:
        seg_text = seg["text"].strip()
        if (
            len(seg_text) < min_size
            and seg["type"] not in ("part_heading", "schedule_heading", "list_heading")
            and not _FOOTNOTE_PATTERNS.search(seg_text)
        ):
            continue

        pieces = [(
            seg_text,
            seg["type"],
            seg["heading"],
            seg.get("article"),
            seg.get("schedule_id", ""),
            seg.get("list_id", ""),
            seg.get("entry_id", ""),
        )]
        if seg.get("article") or seg.get("entry_id"):
            body_text, note_text = _split_editorial_tail(seg_text)
            pieces = []
            if body_text:
                pieces.append((
                    body_text,
                    seg["type"],
                    seg["heading"],
                    seg.get("article"),
                    seg.get("schedule_id", ""),
                    seg.get("list_id", ""),
                    seg.get("entry_id", ""),
                ))
            if note_text:
                pieces.append((
                    note_text,
                    "text_block",
                    "",
                    seg.get("article"),
                    seg.get("schedule_id", ""),
                    seg.get("list_id", ""),
                    "",
                ))

        for piece_text, piece_type, piece_heading, piece_article, piece_schedule, piece_list, piece_entry in pieces:
            piece_text = piece_text.strip()
            if (
                len(piece_text) < min_size
                and piece_type not in ("part_heading", "schedule_heading", "list_heading")
                and not _FOOTNOTE_PATTERNS.search(piece_text)
            ):
                continue

            ctype = classify_chunk_type(piece_text, piece_type)
            sec_id = detect_section_id(piece_text)

            if len(piece_text) <= chunk_size:
                prefixed = _add_article_prefix(
                    piece_text,
                    piece_article if piece_type == "article" else None,
                    piece_heading if piece_type == "article" else "",
                )
                results.append({
                    "text": prefixed,
                    "article": piece_article,
                    "page": page_num,
                    "type": piece_type,
                    "heading": piece_heading,
                    "chunk_type": ctype,
                    "schedule_id": piece_schedule,
                    "list_id": piece_list,
                    "entry_id": piece_entry,
                    "section_id": sec_id,
                })
            else:
                sub_chunks = _subchunk_article(
                    piece_text,
                    piece_article,
                    piece_heading,
                    chunk_size,
                    min_size,
                )
                for i, sc in enumerate(sub_chunks):
                    results.append({
                        "text": sc,
                        "article": piece_article,
                        "page": page_num,
                        "type": piece_type,
                        "heading": piece_heading,
                        "chunk_type": "clause" if (i > 0 and ctype == "article") else ctype,
                        "schedule_id": piece_schedule,
                        "list_id": piece_list,
                        "entry_id": piece_entry,
                        "section_id": sec_id,
                    })

    return results


def _looks_like_article_continuation(text: str, min_size: int) -> bool:
    stripped = text.strip()
    if not stripped or _CONTINUATION_BLOCKERS.match(stripped):
        return False
    if len(re.findall(r"^\d+[A-Z]?\.\s+[A-Z]", stripped, flags=re.MULTILINE)) >= 2:
        return False
    if re.match(r"^\d+[A-Z]?\.\s+[A-Z]", stripped) and not _FOOTNOTE_PATTERNS.search(stripped):
        return False
    if len(stripped) >= max(min_size, 120):
        return True
    if re.match(r"^(?:\(\d+\)|\([a-z]\)|No\b|The\b|There\b|Nothing\b|Any\b|Every\b|Provided\b)", stripped):
        return True
    if _FOOTNOTE_PATTERNS.search(stripped):
        return True
    return False


def _split_editorial_tail(text: str) -> tuple[str, str]:
    """Split trailing amendment or footnote lines away from article body."""
    lines = text.splitlines()
    note_start = None
    for i, line in enumerate(lines):
        if _EDITORIAL_LINE.search(line):
            note_start = i
            break

    if note_start is None:
        return text.strip(), ""

    body = "\n".join(lines[:note_start]).strip()
    notes = "\n".join(lines[note_start:]).strip()
    if not body:
        return "", notes
    return body, notes


def _add_article_prefix(text: str, article: Optional[str], heading: str) -> str:
    """Prepend a clear identifier to help embedding and retrieval."""
    if article:
        prefix = f"[Article {article}]"
        if heading:
            prefix += f" {heading}"
        if f"Article {article}" not in text[:80]:
            return f"{prefix}\n{text}"
    return text


def _subchunk_article(
    text: str,
    article: Optional[str],
    heading: str,
    chunk_size: int,
    min_size: int,
) -> list[str]:
    """
    Split a long article into sub-chunks, preferring clause boundaries.
    Each sub-chunk gets the article prefix for retrieval context.
    """
    prefix = ""
    if article:
        prefix = f"[Article {article}] (continued) "

    clause_parts = re.split(r"(?=\(\d+\)\s|\([a-z]\)\s)", text)
    clause_parts = [p.strip() for p in clause_parts if p.strip()]

    if len(clause_parts) <= 1:
        return [prefix + c for c in _generic_split(text, chunk_size, min_size)]

    chunks = []
    current = ""
    for part in clause_parts:
        candidate = f"{current}\n{part}".strip() if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current and len(current) >= min_size:
                chunks.append(prefix + current if prefix and len(chunks) > 0 else current)
            current = part
    if current and len(current) >= min_size:
        chunks.append(prefix + current if prefix and len(chunks) > 0 else current)

    final = []
    for c in chunks:
        if len(c) <= chunk_size:
            final.append(c)
        else:
            final.extend(prefix + s for s in _generic_split(c, chunk_size, min_size))

    return final if final else [text[:chunk_size]]


def _split_on_boundaries(text: str) -> list[str]:
    """Split text on paragraph / heading boundaries first."""
    parts = re.split(r"\n{2,}|(?=^#{1,4}\s)", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    min_size: int = config.CHUNK_MIN_SIZE,
) -> list[str]:
    """
    Generic fallback chunker. Splits on paragraph boundaries with overlap.
    Used for non-legal PDFs or text without recognizable Article structure.
    """
    paragraphs = _split_on_boundaries(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current and len(current) >= min_size:
                chunks.append(current)
            if len(para) > chunk_size:
                sub_chunks = _hard_split(para, chunk_size, overlap)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = para

    if current and len(current) >= min_size:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap)

    log.info("Chunked %d chars into %d chunks (generic)", len(text), len(chunks))
    return chunks


def _generic_split(text: str, chunk_size: int, min_size: int) -> list[str]:
    """Split by paragraphs then hard-split, no overlap."""
    paras = _split_on_boundaries(text)
    if not paras:
        return [text[:chunk_size]] if text.strip() else []

    chunks = []
    current = ""
    for para in paras:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current and len(current) >= min_size:
                chunks.append(current)
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size):
                    chunks.append(para[i:i + chunk_size])
                current = ""
            else:
                current = para
    if current and len(current) >= min_size:
        chunks.append(current)

    return chunks if chunks else [text[:chunk_size]]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Split oversized text by sentence boundaries, then by raw chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    step = max(size - overlap, 1)
    final: list[str] = []
    for c in chunks:
        if len(c) <= size:
            final.append(c)
        else:
            for i in range(0, len(c), step):
                final.append(c[i : i + size])

    return final


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prepend the last *overlap* chars of the previous chunk."""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        merged = f"{tail} {chunks[i]}".strip()
        result.append(merged)
    return result
