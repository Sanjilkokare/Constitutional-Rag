"""
PDF ingestion pipeline.

For each uploaded PDF:
  1. Compute file hash (skip if already ingested).
  2. Render each page to PNG via PyMuPDF.
  3. Send PDF to Sarvam Document Intelligence for OCR.
  4. Parse OCR output, chunk text, embed, store in FAISS.
"""

import asyncio
import hashlib
import json
import logging
import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

import chunker
import config
import embeddings
import legal_index as li
import sarvam_client
import storage
from storage import ChunkMeta, DocRecord

log = logging.getLogger(__name__)


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


_DOCINTEL_SKIP_LAYOUTS = {
    "header",
    "image",
    "image-caption",
    "page-number",
    "table",
}

_STRUCTURAL_BLOCK = re.compile(
    r"^(?:\*{0,2})?(?:PART\s+[IVXLC]+(?:-[A-Z])?\b|"
    r"(?:Article|Art\.?)\s+\d+[A-Z]?\b|"
    r"\d+[A-Z]?\.\s+[A-Z])",
    re.IGNORECASE,
)

_INLINE_ARTICLE_BODY = re.compile(
    r"^(\d+[A-Z]?\.\s+[^\n]{3,}?\.)\s*[-\u2013\u2014]\s*(\(.+)$",
    re.MULTILINE,
)

_MOJIBAKE_REPLACEMENTS = (
    ("\u00e2\u20ac\u201d", "-"),
    ("\u00e2\u20ac\u201c", "-"),
    ("\u00e2\u20ac\u02dc", "'"),
    ("\u00e2\u20ac\u2122", "'"),
    ("\u00e2\u20ac\u0153", "\""),
    ("\u00e2\u20ac\u009d", "\""),
    ("\u00e2\u20ac\u00a6", "..."),
    ("\u00c2", ""),
    ("\xa0", " "),
)


# Text cleaning for Sarvam DocIntel structured JSON output

# Repeated page headers to strip (case-insensitive, after whitespace norm)
_REPEATED_HEADERS = re.compile(
    r"^(?:THE\s+CONSTITUTION\s+OF\s+INDIA)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_docintel_payload(content: str):
    """Parse DocIntel JSON, tolerating BOMs and fenced/wrapped payloads."""
    if not isinstance(content, str):
        return None

    cleaned = content.lstrip("\ufeff").strip()
    if not cleaned:
        return None

    candidates = [cleaned]

    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            candidates.append(cleaned[start : end + 1])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    return None


def _find_docintel_pages(payload) -> list[dict]:
    """Return page payloads that contain DocIntel blocks."""
    if isinstance(payload, dict):
        blocks = payload.get("blocks")
        if isinstance(blocks, list):
            return [payload]

        for value in payload.values():
            nested = _find_docintel_pages(value)
            if nested:
                return nested

    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) and "blocks" in item for item in payload):
            return [item for item in payload if isinstance(item.get("blocks"), list)]
        if payload and all(isinstance(item, dict) and "text" in item for item in payload):
            return [{"blocks": payload}]

        for item in payload:
            nested = _find_docintel_pages(item)
            if nested:
                return nested

    return []


def _block_sort_key(block: dict) -> tuple[float, float, float, str]:
    """Use reading order when present, otherwise fall back to coordinates."""
    order = block.get("reading_order")
    try:
        order_val = float(order)
    except (TypeError, ValueError):
        order_val = float("inf")

    coords = block.get("coordinates") or {}
    try:
        y_val = float(coords.get("y1", float("inf")))
    except (TypeError, ValueError):
        y_val = float("inf")
    try:
        x_val = float(coords.get("x1", float("inf")))
    except (TypeError, ValueError):
        x_val = float("inf")

    return (order_val, y_val, x_val, str(block.get("block_id", "")))


def _render_docintel_page(page_payload: dict) -> str:
    """Convert one DocIntel page payload into readable text lines."""
    blocks = page_payload.get("blocks")
    if not isinstance(blocks, list):
        return ""

    lines: list[str] = []
    prev_layout = ""

    for block in sorted(blocks, key=_block_sort_key):
        if not isinstance(block, dict):
            continue

        layout = str(block.get("layout_tag") or "").strip().lower()
        if layout in _DOCINTEL_SKIP_LAYOUTS:
            continue

        text = str(block.get("text") or "").replace("\r", "").strip()
        if not text:
            continue

        if lines and (
            layout in {"section-title", "title", "headline"}
            or _STRUCTURAL_BLOCK.match(text)
            or (prev_layout == "footnote" and layout != "footnote")
        ):
            lines.append("")

        lines.append(text)
        prev_layout = layout

    return "\n".join(lines).strip()


def extract_readable_text_from_docintel(content: str) -> str | None:
    """Extract only readable text from DocIntel JSON payloads."""
    payload = _parse_docintel_payload(content)
    if payload is None:
        return None

    pages = _find_docintel_pages(payload)
    if len(pages) != 1:
        return None

    text = _render_docintel_page(pages[0])
    return text or None


def normalize_page_text(raw: str) -> str:
    """Normalize OCR page text for chunking and legal structure detection."""
    text = raw

    for bad, good in _MOJIBAKE_REPLACEMENTS:
        text = text.replace(bad, good)

    text = re.sub(r"\$\^\{\d+\}\[", "", text)
    text = re.sub(r"\\mathrm\{([A-Z])\.\}", r"\1.", text)
    text = text.replace("$", "")
    text = text.replace("**", "")
    text = _INLINE_ARTICLE_BODY.sub(r"\1\n\2", text)

    text = _REPEATED_HEADERS.sub("", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def render_pages(pdf_bytes: bytes, doc_id: str) -> list[Path]:
    """Render every page of a PDF to PNG. Returns list of image paths."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    image_paths: list[Path] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=config.PAGE_IMAGE_DPI)
        img_name = f"{doc_id}_page_{page_num + 1}.png"
        img_path = config.PAGE_IMAGES_DIR / img_name
        pix.save(str(img_path))
        image_paths.append(img_path)
        log.debug("Rendered page %d -> %s", page_num + 1, img_path)

    doc.close()
    log.info("Rendered %d pages for doc %s", len(image_paths), doc_id)
    return image_paths


def _page_number_from_name(fname: str, default: int) -> int:
    """Extract a page number from an OCR result filename."""
    match = re.search(r"(\d+)", Path(fname).stem)
    if match:
        return int(match.group(1))
    return default


def _extract_combined_pages(content: str) -> list[tuple[int, str]]:
    """
    Extract explicit page blocks from a single combined OCR file.

    Any preamble before the first page marker is ignored so it cannot become
    a phantom extra page during batch merge.
    """
    marker_re = re.compile(r"---\s*page\s*(\d+)\s*---", re.IGNORECASE)
    matches = list(marker_re.finditer(content))
    if matches:
        pages: list[tuple[int, str]] = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            cleaned = normalize_page_text(content[start:end])
            if cleaned:
                pages.append((int(match.group(1)), cleaned))
        return pages

    if "\f" in content:
        return [
            (idx + 1, normalize_page_text(part))
            for idx, part in enumerate(content.split("\f"))
            if part.strip()
        ]

    return []


def _validate_page_map(
    page_texts: dict[int, str],
    expected_start_page: int,
    expected_end_page: int,
    context: str,
) -> dict[int, str]:
    """Clamp to the expected page window and log missing/extra pages."""
    expected_pages = list(range(expected_start_page, expected_end_page + 1))
    expected_set = set(expected_pages)
    actual_set = set(page_texts)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)

    log.info(
        "%s expected %d page(s), actual %d page(s)",
        context,
        len(expected_pages),
        len(page_texts),
    )
    if missing:
        log.warning("%s missing pages: %s", context, missing)
    if extra:
        log.warning("%s extra pages: %s", context, extra)

    normalized = {page_num: page_texts.get(page_num, "") for page_num in expected_pages}
    if normalized:
        keys = list(normalized)
        log.info("%s final sorted page keys range: %d-%d", context, keys[0], keys[-1])
    return normalized


def _parse_ocr_pages(
    ocr_results: dict[str, str],
    page_number_offset: int = 0,
) -> dict[int, str]:
    """Parse OCR output files into {page_number: cleaned_text}."""
    pages: dict[int, str] = {}
    docintel_detected = False

    for fname, content in sorted(
        ocr_results.items(),
        key=lambda item: _page_number_from_name(item[0], len(ocr_results) + 1),
    ):
        fallback_page_num = _page_number_from_name(fname, len(pages) + 1)

        payload = _parse_docintel_payload(content)
        docintel_pages = _find_docintel_pages(payload) if payload is not None else []
        if docintel_pages:
            docintel_detected = True
            for offset, page_payload in enumerate(docintel_pages):
                page_num = page_payload.get("page_num")
                try:
                    page_num = int(page_num)
                except (TypeError, ValueError):
                    page_num = fallback_page_num + offset

                cleaned = normalize_page_text(_render_docintel_page(page_payload))
                if cleaned:
                    pages[page_num + page_number_offset] = cleaned
            continue

        cleaned = extract_readable_text_from_docintel(content)
        if cleaned is not None:
            docintel_detected = True
            pages[fallback_page_num + page_number_offset] = normalize_page_text(cleaned)
        else:
            pages[fallback_page_num + page_number_offset] = normalize_page_text(
                content.strip()
            )

    if docintel_detected:
        log.info("Detected Sarvam DocIntel JSON format - extracted readable text from blocks")

    if len(pages) == 1 and len(ocr_results) == 1:
        combined = list(pages.values())[0]
        combined_pages = _extract_combined_pages(combined)
        if combined_pages:
            pages = {
                page_number_offset + page_num: page_text
                for page_num, page_text in combined_pages
            }

    return pages


def _fallback_text_extract(
    pdf_bytes: bytes,
    page_number_offset: int = 0,
) -> dict[str, str]:
    """Extract text from PDF using PyMuPDF as OCR fallback."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results: dict[str, str] = {}
    for i in range(len(doc)):
        text = doc[i].get_text()
        results[f"page_{page_number_offset + i + 1}.md"] = text
    doc.close()
    return results


def _ocr_small_pdf(
    pdf_bytes: bytes,
    filename: str,
    num_pages: int,
) -> tuple[dict[int, str], dict[int, str]]:
    """Use the existing single-request OCR flow for PDFs within Sarvam's page limit."""
    try:
        ocr_results = sarvam_client.ocr_pdf(pdf_bytes, filename)
        page_texts = _parse_ocr_pages(ocr_results)
        page_texts = _validate_page_map(page_texts, 1, num_pages, f"Small PDF OCR {filename}")
        if any(text.strip() for text in page_texts.values()):
            return page_texts, {page_num: "sarvam_ocr" for page_num in page_texts}

        log.warning("OCR returned no usable text for %s - falling back to PyMuPDF", filename)
    except Exception as e:
        log.error("OCR failed for %s: %s", filename, e)

    fallback_results = _fallback_text_extract(pdf_bytes)
    page_texts = _parse_ocr_pages(fallback_results)
    page_texts = _validate_page_map(
        page_texts,
        1,
        num_pages,
        f"Small PDF PyMuPDF fallback {filename}",
    )
    return page_texts, {page_num: "pymupdf" for page_num in page_texts}


def _write_batch_pdf(
    source_doc: fitz.Document,
    start_page: int,
    end_page: int,
    output_path: Path,
) -> bytes:
    """Write an inclusive page range to a temporary mini-PDF and return its bytes."""
    batch_doc = fitz.open()
    try:
        batch_doc.insert_pdf(source_doc, from_page=start_page - 1, to_page=end_page - 1)
        batch_doc.save(str(output_path))
    finally:
        batch_doc.close()
    return output_path.read_bytes()


def _ocr_pdf_in_batches(
    pdf_bytes: bytes,
    filename: str,
    num_pages: int,
) -> tuple[dict[int, str], dict[int, str]]:
    """
    OCR a PDF, splitting larger files into sequential mini-PDF batches.

    Returns
    -------
    tuple[dict[int, str], dict[int, str]]
        Page text keyed by absolute page number, plus per-page extraction source.
    """
    batch_size = config.SARVAM_OCR_MAX_PAGES_PER_PDF
    if num_pages <= batch_size:
        return _ocr_small_pdf(pdf_bytes, filename, num_pages)

    total_batches = (num_pages + batch_size - 1) // batch_size
    log.info("PDF has %d pages, using %d OCR batches", num_pages, total_batches)

    merged_page_texts: dict[int, str] = {}
    page_sources: dict[int, str] = {}

    source_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for batch_idx, start_idx in enumerate(range(0, num_pages, batch_size), start=1):
            start_page = start_idx + 1
            end_page = min(start_idx + batch_size, num_pages)
            batch_path = config.DATA_DIR / (
                f".sarvam_ocr_batch_{uuid.uuid4().hex}_{start_page}_{end_page}.pdf"
            )

            try:
                batch_bytes = _write_batch_pdf(source_doc, start_page, end_page, batch_path)
                log.info(
                    "OCR batch %d/%d pages %d-%d",
                    batch_idx,
                    total_batches,
                    start_page,
                    end_page,
                )

                try:
                    ocr_results = sarvam_client.ocr_pdf(batch_bytes, batch_path.name)
                    batch_page_texts = _parse_ocr_pages(
                        ocr_results,
                        page_number_offset=start_idx,
                    )
                    if any(text.strip() for text in batch_page_texts.values()):
                        batch_source = "sarvam_ocr"
                    else:
                        raise RuntimeError("OCR returned no usable text")
                except Exception as e:
                    log.warning(
                        "Batch %d failed, falling back to PyMuPDF for pages %d-%d: %s",
                        batch_idx,
                        start_page,
                        end_page,
                        e,
                    )
                    fallback_results = _fallback_text_extract(
                        batch_bytes,
                        page_number_offset=start_idx,
                    )
                    batch_page_texts = _parse_ocr_pages(fallback_results)
                    batch_source = "pymupdf"

                batch_page_texts = _validate_page_map(
                    batch_page_texts,
                    start_page,
                    end_page,
                    f"OCR batch {batch_idx}/{total_batches} pages {start_page}-{end_page}",
                )

                merged_page_texts.update(batch_page_texts)
                page_sources.update(
                    {page_num: batch_source for page_num in batch_page_texts}
                )
            finally:
                batch_path.unlink(missing_ok=True)
    finally:
        source_doc.close()

    merged_page_texts = _validate_page_map(
        merged_page_texts,
        1,
        num_pages,
        f"Merged OCR text for {filename}",
    )
    page_sources = {page_num: page_sources.get(page_num, "sarvam_ocr") for page_num in merged_page_texts}
    log.info("Merged OCR text for %d pages", num_pages)
    return merged_page_texts, page_sources

async def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    progress_callback=None,
) -> DocRecord:
    """
    Full ingestion pipeline for one PDF.

    Parameters
    ----------
    pdf_bytes : raw bytes
    filename : display name
    progress_callback : async callable(message: str) for UI updates

    Returns
    -------
    DocRecord describing what was ingested.
    """
    fhash = file_hash(pdf_bytes)

    if storage.doc_already_ingested(fhash):
        msg = f"⏭ {filename} already ingested (hash={fhash}), skipping."
        log.info(msg)
        if progress_callback:
            await progress_callback(msg)
        # Return existing record
        for d in storage.get_doc_records():
            if d["file_hash"] == fhash and d.get("chunk_count", 0) > 0:
                return DocRecord(**d)
        # Shouldn't happen but fallback
        return DocRecord(doc_id=fhash, filename=filename, file_hash=fhash,
                         pages=0, chunk_count=0)

    # Remove any stale 0-chunk records for this hash before re-ingestion
    storage.remove_doc_record(fhash)

    doc_id = uuid.uuid4().hex[:12]

    # ── Step 1: Render pages as images (offloaded to thread) ─────────
    if progress_callback:
        await progress_callback(f"Rendering pages for {filename}...")
    image_paths = await asyncio.to_thread(render_pages, pdf_bytes, doc_id)
    num_pages = len(image_paths)
    if progress_callback:
        await progress_callback(f"Rendered {num_pages} page(s) as images.")

    # ── Step 2: OCR via Sarvam (offloaded to thread — polls with sleep) ──
    if progress_callback:
        if num_pages <= config.SARVAM_OCR_MAX_PAGES_PER_PDF:
            await progress_callback(
                f"Sending {filename} to Sarvam OCR (this may take a minute)..."
            )
        else:
            total_batches = (
                num_pages + config.SARVAM_OCR_MAX_PAGES_PER_PDF - 1
            ) // config.SARVAM_OCR_MAX_PAGES_PER_PDF
            await progress_callback(
                f"Sending {filename} to Sarvam OCR in {total_batches} batches "
                "(this may take a minute)..."
            )

    page_texts, page_sources = await asyncio.to_thread(
        _ocr_pdf_in_batches,
        pdf_bytes,
        filename,
        num_pages,
    )

    log.info("Parsed %d page_texts, pages with text: %d",
             len(page_texts),
             sum(1 for t in page_texts.values() if t.strip()))


    if progress_callback:
        await progress_callback(
            f"Text extraction complete — got text from {len(page_texts)} page(s)."
        )

    # ── Step 3: Chunk + Embed + Store ─────────────────────────────────
    all_chunks: list[ChunkMeta] = []
    all_texts: list[str] = []
    chunk_id = 0
    # Load existing legal index (preserves other docs) instead of reset
    legal_idx = li.get_legal_index()
    # Global offset so legal index chunk_ids match FAISS positions
    global_offset = storage.get_total_chunks()
    articles_found = 0
    carry_article = ""
    carry_heading = ""
    carry_schedule = ""
    carry_list = ""
    carry_entry = ""

    for page_num in sorted(page_texts.keys()):
        text = page_texts[page_num]
        if not text.strip():
            continue

        # Determine which image corresponds to this page
        img_idx = page_num - 1
        if 0 <= img_idx < len(image_paths):
            img_rel = str(image_paths[img_idx].relative_to(config.BASE_DIR))
        else:
            img_rel = ""

        # Try legal-structure-aware chunking first
        legal_chunks = chunker.chunk_page_legal(
            text,
            page_num,
            carry_article=carry_article or None,
            carry_heading=carry_heading,
            carry_schedule=carry_schedule,
            carry_list=carry_list,
            carry_entry=carry_entry,
        )

        if legal_chunks:
            last_article = ""
            last_heading = ""
            last_schedule = ""
            last_list = ""
            last_entry = ""
            for lc in legal_chunks:
                page_source = page_sources.get(page_num, "sarvam_ocr")
                meta = ChunkMeta(
                    doc_id=doc_id,
                    filename=filename,
                    page=page_num,
                    chunk_id=chunk_id,
                    text=lc["text"],
                    image_path=img_rel,
                    page_start=page_num,
                    page_end=page_num,
                    chunk_type=lc.get("chunk_type", "generic"),
                    article_id=lc.get("article") or "",
                    schedule_id=lc.get("schedule_id") or "",
                    list_id=lc.get("list_id") or "",
                    entry_id=lc.get("entry_id") or "",
                    section_id=lc.get("section_id") or "",
                    source_extraction=page_source,
                )
                all_chunks.append(meta)
                all_texts.append(lc["text"])
                # Register in legal index with global FAISS position
                legal_idx.register_chunk(
                    global_offset + chunk_id, lc["text"],
                    article=lc.get("article"),
                    schedule=lc.get("schedule_id") or "",
                    list_id=lc.get("list_id") or "",
                    entry_id=lc.get("entry_id") or "",
                )
                if lc.get("article"):
                    articles_found += 1
                    last_article = lc.get("article") or last_article
                    last_heading = lc.get("heading") or last_heading
                if lc.get("schedule_id"):
                    last_schedule = lc.get("schedule_id") or last_schedule
                if lc.get("list_id"):
                    last_list = lc.get("list_id") or last_list
                if lc.get("entry_id"):
                    last_entry = lc.get("entry_id") or last_entry
                elif lc.get("list_id") and lc.get("chunk_type") == "heading":
                    last_entry = ""
                chunk_id += 1
            carry_article = last_article
            carry_heading = last_heading
            carry_schedule = last_schedule
            carry_list = last_list
            carry_entry = last_entry
        else:
            carry_article = ""
            carry_heading = ""
            carry_schedule = ""
            carry_list = ""
            carry_entry = ""
            # Fallback to generic chunking
            chunks = chunker.chunk_text(text)
            for chunk_str in chunks:
                page_source = page_sources.get(page_num, "sarvam_ocr")
                meta = ChunkMeta(
                    doc_id=doc_id,
                    filename=filename,
                    page=page_num,
                    chunk_id=chunk_id,
                    text=chunk_str,
                    image_path=img_rel,
                    page_start=page_num,
                    page_end=page_num,
                    chunk_type="generic",
                    article_id="",
                    schedule_id="",
                    list_id="",
                    entry_id="",
                    section_id="",
                    source_extraction=page_source,
                )
                all_chunks.append(meta)
                all_texts.append(chunk_str)
                legal_idx.register_chunk(global_offset + chunk_id, chunk_str)
                chunk_id += 1

    log.info("Chunking complete: %d chunks, %d article-tagged chunks", len(all_chunks), articles_found)

    if all_texts:
        # Embed in batches with progress for large PDFs
        embed_batch = 256
        total = len(all_texts)
        all_vectors = []
        for batch_start in range(0, total, embed_batch):
            batch_end = min(batch_start + embed_batch, total)
            if progress_callback:
                await progress_callback(
                    f"Embedding chunks {batch_start + 1}–{batch_end} of {total}..."
                )
            batch_vecs = await asyncio.to_thread(
                embeddings.embed_texts, all_texts[batch_start:batch_end]
            )
            all_vectors.append(batch_vecs)

        vectors = np.concatenate(all_vectors, axis=0)
        storage.add_chunks(vectors, all_chunks)

    # ── Step 4: Record + Save ─────────────────────────────────────────
    if not all_chunks:
        # Do NOT record the doc — extraction produced 0 chunks.
        # Remove any stale record for this hash so it can be retried.
        storage.remove_doc_record(fhash)
        storage.save()
        msg = (
            f"Ingestion produced 0 chunks for {filename}. "
            "The document was NOT marked as ingested so you can retry."
        )
        log.warning(msg)
        if progress_callback:
            await progress_callback(msg)
        return DocRecord(
            doc_id=doc_id, filename=filename, file_hash=fhash,
            pages=num_pages, chunk_count=0,
        )

    rec = DocRecord(
        doc_id=doc_id,
        filename=filename,
        file_hash=fhash,
        pages=num_pages,
        chunk_count=len(all_chunks),
    )
    storage.add_doc_record(rec)
    storage.save()
    legal_idx.save()

    if progress_callback:
        await progress_callback(
            f"Ingestion complete for {filename}: "
            f"{num_pages} pages, {len(all_chunks)} chunks indexed, "
            f"{len(legal_idx.articles)} articles in legal index."
        )

    log.info(
        "Ingested %s — doc_id=%s, pages=%d, chunks=%d",
        filename, doc_id, num_pages, len(all_chunks),
    )
    return rec
