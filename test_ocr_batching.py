import json
import unittest
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import fitz
import numpy as np

import config
import embeddings
import ingest
import legal_index as li
import sarvam_client
import storage


def _make_pdf(page_texts: list[str]) -> bytes:
    doc = fitz.open()
    try:
        for text in page_texts:
            page = doc.new_page()
            page.insert_textbox(fitz.Rect(72, 72, 540, 780), text, fontsize=12)
        return doc.tobytes()
    finally:
        doc.close()


class OcrBatchingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._reset_runtime_state()

    def tearDown(self):
        self._reset_runtime_state()

    def _reset_runtime_state(self):
        storage._index = None
        storage._metadata = []
        storage._docs = []
        li._legal_index = None

    def _runtime_patches(self) -> ExitStack:
        unique = uuid.uuid4().hex
        data_dir = Path(config.DATA_DIR)
        faiss_dir = Path(config.FAISS_INDEX_DIR)
        documents_json = data_dir / f"test_documents_{unique}.json"
        index_file = faiss_dir / f"test_index_{unique}.faiss"
        meta_file = faiss_dir / f"test_metadata_{unique}.json"
        legal_index_file = faiss_dir / f"test_legal_index_{unique}.json"

        stack = ExitStack()
        stack.enter_context(patch.object(config, "DOCUMENTS_JSON", documents_json))
        stack.enter_context(patch.object(storage, "INDEX_FILE", index_file))
        stack.enter_context(patch.object(storage, "META_FILE", meta_file))
        stack.enter_context(patch.object(li, "LEGAL_INDEX_FILE", legal_index_file))
        stack.callback(documents_json.unlink, missing_ok=True)
        stack.callback(index_file.unlink, missing_ok=True)
        stack.callback(meta_file.unlink, missing_ok=True)
        stack.callback(legal_index_file.unlink, missing_ok=True)
        return stack

    def test_small_pdf_uses_single_ocr_flow(self):
        pdf_bytes = _make_pdf(["Alpha page", "Beta page"])
        ocr_output = {
            "page_1.md": "Alpha OCR text",
            "page_2.md": "Beta OCR text",
        }

        with patch.object(sarvam_client, "ocr_pdf", return_value=ocr_output) as mock_ocr:
            page_texts, page_sources = ingest._ocr_pdf_in_batches(pdf_bytes, "small.pdf", 2)

        self.assertEqual(page_texts, {1: "Alpha OCR text", 2: "Beta OCR text"})
        self.assertEqual(page_sources, {1: "sarvam_ocr", 2: "sarvam_ocr"})
        mock_ocr.assert_called_once()
        self.assertEqual(mock_ocr.call_args[0][1], "small.pdf")

    def test_large_pdf_batches_preserve_absolute_page_order(self):
        pdf_bytes = _make_pdf([f"Original page {i}" for i in range(1, 13)])

        def fake_ocr(_batch_bytes: bytes, batch_filename: str):
            if batch_filename.endswith("_1_10.pdf"):
                combined = "OCR preamble\n" + "".join(
                    f"--- page {page_num} ---\nOCR page {page_num}\n"
                    for page_num in range(1, 11)
                )
                return {"combined.md": combined}
            if batch_filename.endswith("_11_12.pdf"):
                return {
                    "page_1.md": "OCR page 11",
                    "page_2.md": "OCR page 12",
                }
            raise AssertionError(f"Unexpected batch filename: {batch_filename}")

        with patch.object(sarvam_client, "ocr_pdf", side_effect=fake_ocr) as mock_ocr:
            with self.assertLogs("ingest", level="INFO") as captured:
                page_texts, page_sources = ingest._ocr_pdf_in_batches(
                    pdf_bytes,
                    "large.pdf",
                    12,
                )

        self.assertEqual(list(page_texts), list(range(1, 13)))
        self.assertEqual(len(page_texts), 12)
        self.assertNotIn(13, page_texts)
        self.assertEqual(page_texts[1], "OCR page 1")
        self.assertEqual(page_texts[10], "OCR page 10")
        self.assertEqual(page_texts[11], "OCR page 11")
        self.assertEqual(page_texts[12], "OCR page 12")
        self.assertTrue(all(source == "sarvam_ocr" for source in page_sources.values()))
        self.assertEqual(mock_ocr.call_count, 2)

        log_text = "\n".join(captured.output)
        self.assertIn("PDF has 12 pages, using 2 OCR batches", log_text)
        self.assertIn("OCR batch 1/2 pages 1-10", log_text)
        self.assertIn("OCR batch 2/2 pages 11-12", log_text)
        self.assertIn("Merged OCR text for large.pdf expected 12 page(s), actual 12 page(s)", log_text)
        self.assertIn("Merged OCR text for 12 pages", log_text)

    def test_large_pdf_mixed_ocr_and_fallback_preserves_exact_count(self):
        pdf_bytes = _make_pdf([f"Original page {i}" for i in range(1, 13)])

        def fake_ocr(_batch_bytes: bytes, batch_filename: str):
            if batch_filename.endswith("_1_10.pdf"):
                combined = "OCR preamble\n" + "".join(
                    f"--- page {page_num} ---\nOCR page {page_num}\n"
                    for page_num in range(1, 11)
                )
                return {"combined.md": combined}
            if batch_filename.endswith("_11_12.pdf"):
                raise RuntimeError("Sarvam rejected this batch")
            raise AssertionError(f"Unexpected batch filename: {batch_filename}")

        with patch.object(sarvam_client, "ocr_pdf", side_effect=fake_ocr):
            with self.assertLogs("ingest", level="INFO") as captured:
                page_texts, page_sources = ingest._ocr_pdf_in_batches(
                    pdf_bytes,
                    "mixed.pdf",
                    12,
                )

        self.assertEqual(list(page_texts), list(range(1, 13)))
        self.assertEqual(len(page_texts), 12)
        self.assertNotIn(13, page_texts)
        self.assertEqual(page_texts[11], "Original page 11")
        self.assertEqual(page_texts[12], "Original page 12")
        self.assertEqual(page_sources[1], "sarvam_ocr")
        self.assertEqual(page_sources[11], "pymupdf")
        self.assertEqual(page_sources[12], "pymupdf")

        log_text = "\n".join(captured.output)
        self.assertIn("Batch 2 failed, falling back to PyMuPDF for pages 11-12", log_text)
        self.assertIn("Merged OCR text for mixed.pdf expected 12 page(s), actual 12 page(s)", log_text)

    async def test_large_pdf_batch_fallback_still_reaches_storage_and_legal_index(self):
        page_texts = [
            f"Article {page_num}. Page {page_num} heading\n"
            f"This is the body text for article {page_num}. "
            f"It includes enough repeated detail to exceed the minimum chunk size. "
            f"This is the body text for article {page_num}. "
            f"It includes enough repeated detail to exceed the minimum chunk size."
            for page_num in range(1, 13)
        ]
        pdf_bytes = _make_pdf(page_texts)

        def fake_ocr(_batch_bytes: bytes, batch_filename: str):
            if batch_filename.endswith("_1_10.pdf"):
                return {
                    f"page_{page_num}.md": page_texts[page_num - 1]
                    for page_num in range(1, 11)
                }
            if batch_filename.endswith("_11_12.pdf"):
                raise RuntimeError("Sarvam rejected this batch")
            raise AssertionError(f"Unexpected batch filename: {batch_filename}")

        def fake_embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
            del batch_size
            return np.ones((len(texts), config.EMBEDDING_DIMENSION), dtype=np.float32)

        def fake_render_pages(_pdf_bytes: bytes, doc_id: str) -> list[Path]:
            return [
                Path(config.PAGE_IMAGES_DIR) / f"{doc_id}_page_{page_num}.png"
                for page_num in range(1, 13)
            ]

        with self._runtime_patches():
            with patch.object(ingest, "render_pages", side_effect=fake_render_pages):
                with patch.object(sarvam_client, "ocr_pdf", side_effect=fake_ocr):
                    with patch.object(embeddings, "embed_texts", side_effect=fake_embed_texts):
                        with self.assertLogs("ingest", level="INFO") as captured:
                            rec = await ingest.ingest_pdf(pdf_bytes, "batched.pdf")

            self.assertEqual(rec.pages, 12)
            self.assertGreater(rec.chunk_count, 0)
            self.assertEqual(storage.get_total_chunks(), rec.chunk_count)
            self.assertTrue(storage.INDEX_FILE.exists())
            self.assertTrue(storage.META_FILE.exists())
            self.assertTrue(config.DOCUMENTS_JSON.exists())
            self.assertTrue(li.LEGAL_INDEX_FILE.exists())

            docs = json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]["pages"], 12)

            metadata = storage.get_all_metadata()
            self.assertTrue(any(chunk["source_extraction"] == "sarvam_ocr" for chunk in metadata))
            self.assertTrue(any(chunk["source_extraction"] == "pymupdf" for chunk in metadata))

            source_by_page: dict[int, str] = {}
            for chunk in metadata:
                source_by_page.setdefault(chunk["page"], chunk["source_extraction"])
            self.assertEqual(sorted(source_by_page), list(range(1, 13)))
            self.assertNotIn(13, source_by_page)
            self.assertEqual(source_by_page[1], "sarvam_ocr")
            self.assertEqual(source_by_page[11], "pymupdf")
            self.assertEqual(source_by_page[12], "pymupdf")

            legal_index_data = json.loads(li.LEGAL_INDEX_FILE.read_text(encoding="utf-8"))
            self.assertIn("11", legal_index_data["articles"])
            self.assertIn("12", legal_index_data["articles"])

            log_text = "\n".join(captured.output)
            self.assertIn("PDF has 12 pages, using 2 OCR batches", log_text)
            self.assertIn("OCR batch 1/2 pages 1-10", log_text)
            self.assertIn("OCR batch 2/2 pages 11-12", log_text)
            self.assertIn(
                "Batch 2 failed, falling back to PyMuPDF for pages 11-12",
                log_text,
            )
            self.assertIn("Merged OCR text for batched.pdf expected 12 page(s), actual 12 page(s)", log_text)
            self.assertIn("Merged OCR text for 12 pages", log_text)


if __name__ == "__main__":
    unittest.main()
