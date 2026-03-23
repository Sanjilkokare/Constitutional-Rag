"""
Chainlit application — PDF upload, ingestion status, RAG chat.
Entry point:  chainlit run app.py
"""

import asyncio
import logging
import sys
from pathlib import Path

import chainlit as cl

import config
import ingest
import retriever
import sarvam_client
import storage

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("app")


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle hooks
# ═══════════════════════════════════════════════════════════════════════════


@cl.on_chat_start
async def on_start():
    """Welcome message + show current index stats."""
    total = storage.get_total_chunks()
    all_docs = storage.get_doc_records()
    indexed_docs = [d for d in all_docs if d.get("chunk_count", 0) > 0]
    failed_docs = [d for d in all_docs if d.get("chunk_count", 0) == 0]

    welcome = (
        "## Welcome to RAG-Sarvam\n\n"
        "Upload one or more **PDF files** to index them, "
        "then ask questions about their content.\n\n"
    )
    if indexed_docs:
        doc_list = "\n".join(
            f"- **{d['filename']}** ({d['pages']} pages, {d['chunk_count']} chunks)"
            for d in indexed_docs
        )
        welcome += (
            f"**Indexed documents ({len(indexed_docs)}):**\n{doc_list}\n\n"
            f"Total chunks in vector store: **{total}**\n\n"
            "You can start asking questions or upload more PDFs."
        )
    else:
        welcome += (
            "No documents indexed yet. "
            "Attach a PDF to get started!"
        )

    if failed_docs:
        failed_list = ", ".join(f"**{d['filename']}**" for d in failed_docs)
        welcome += (
            f"\n\nDocuments that failed to index (re-upload to retry): {failed_list}"
        )

    await cl.Message(content=welcome).send()


# ═══════════════════════════════════════════════════════════════════════════
# Message handler
# ═══════════════════════════════════════════════════════════════════════════


@cl.on_message
async def on_message(message: cl.Message):
    """Handle both file uploads and chat queries."""

    # ── Check for file attachments ────────────────────────────────────
    # Chainlit delivers user-uploaded files via message.elements
    pdf_files = []
    for el in (message.elements or []):
        name = getattr(el, "name", "") or ""
        if name.lower().endswith(".pdf") and getattr(el, "path", None):
            pdf_files.append(el)

    if pdf_files:
        await _handle_uploads(pdf_files)
        # If there's also a text query alongside the upload, answer it
        if message.content.strip():
            await _handle_query(message.content.strip())
        return

    # ── Chat query ────────────────────────────────────────────────────
    if message.content.strip():
        await _handle_query(message.content.strip())
    else:
        await cl.Message(
            content="Please upload a PDF or type a question."
        ).send()


# ═══════════════════════════════════════════════════════════════════════════
# Upload processing
# ═══════════════════════════════════════════════════════════════════════════


async def _handle_uploads(pdf_files: list):
    """Ingest each uploaded PDF with progress updates."""
    for f in pdf_files:
        status_msg = cl.Message(content=f"**Processing:** {f.name}...")
        await status_msg.send()

        async def progress_cb(text: str):
            await cl.Message(content=text).send()

        try:
            pdf_bytes = Path(f.path).read_bytes()
            rec = await ingest.ingest_pdf(pdf_bytes, f.name, progress_cb)

            summary = (
                f"**{rec.filename}** ingested successfully.\n"
                f"- Pages: {rec.pages}\n"
                f"- Chunks: {rec.chunk_count}\n"
                f"- Doc ID: `{rec.doc_id}`"
            )
            await cl.Message(content=summary).send()

        except Exception as e:
            log.exception("Ingestion failed for %s", f.name)
            await cl.Message(
                content=f"Ingestion failed for **{f.name}**: {e}"
            ).send()


# ═══════════════════════════════════════════════════════════════════════════
# Query handling
# ═══════════════════════════════════════════════════════════════════════════


async def _handle_query(query: str):
    """Retrieve context, call Sarvam LLM, display answer with sources."""

    if storage.get_total_chunks() == 0:
        await cl.Message(
            content="No documents have been indexed yet. Please upload a PDF first."
        ).send()
        return

    # ── Retrieve (offloaded to thread) ────────────────────────────────
    thinking_msg = cl.Message(content="Searching documents...")
    await thinking_msg.send()

    chunks = await asyncio.to_thread(retriever.retrieve, query)

    if not chunks:
        await cl.Message(
            content="No relevant chunks found. Try rephrasing your question."
        ).send()
        return

    # ── Build prompt & call LLM (offloaded to thread) ─────────────────
    await cl.Message(content="Generating answer with Sarvam AI...").send()

    messages = retriever.build_rag_prompt(query, chunks)

    try:
        answer = await asyncio.to_thread(sarvam_client.chat_answer, messages)
    except Exception as e:
        log.exception("Chat completion failed")
        await cl.Message(content=f"LLM error: {e}").send()
        return

    # ── Format answer ─────────────────────────────────────────────────
    answer = retriever.clean_rag_answer(answer, chunks, query=query)
    await cl.Message(content=answer).send()

    # ── Source citations ──────────────────────────────────────────────
    # Collect unique cited pages
    seen_pages: set[str] = set()
    source_elements: list[cl.Element] = []

    for c in chunks:
        if c.get("synthetic"):
            continue
        page_key = f"{c['filename']}:p{c['page']}"
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)

        img_path = Path(config.BASE_DIR) / c.get("image_path", "")
        if img_path.exists() and img_path.is_file():
            source_elements.append(
                cl.Image(
                    name=page_key,
                    path=str(img_path),
                    display="inline",
                    size="medium",
                )
            )

    # ── Retrieved context (expandable) ────────────────────────────────
    context_lines: list[str] = []
    retrieval_notes: list[str] = []
    for i, c in enumerate(chunks, 1):
        if c.get("synthetic"):
            article = c.get("article_id") or c.get("article") or "Unknown"
            reason = c.get("synthetic_reason", "synthetic")
            retrieval_notes.append(
                f"**Retrieval Note {i}** - Article {article} [{reason}] "
                f"(score: {c.get('score', 'N/A')})\n"
                f"```\n{c['text'][:500]}{'...' if len(c['text']) > 500 else ''}\n```"
            )
            continue
        art_tag = f" Art.{c['article_id']}" if c.get("article_id") else ""
        ctype_tag = f" [{c['chunk_type']}]" if c.get("chunk_type", "generic") != "generic" else ""
        page_display = c.get("page_start") or c.get("page", "?")
        context_lines.append(
            f"**Chunk {i}** — {c['filename']} p.{page_display}{art_tag}{ctype_tag} "
            f"(score: {c.get('score', 'N/A')})\n"
            f"```\n{c['text'][:500]}{'...' if len(c['text']) > 500 else ''}\n```"
        )

    sections: list[str] = []
    if context_lines:
        sections.append("### Source Pages\n\n" + "\n\n".join(context_lines))
    if retrieval_notes:
        sections.append("### Retrieval Notes\n\n" + "\n\n".join(retrieval_notes))
    sources_msg = cl.Message(
        content="\n\n".join(sections),
        elements=source_elements,
    )
    await sources_msg.send()
