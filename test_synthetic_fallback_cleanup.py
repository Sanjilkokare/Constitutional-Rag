import unittest

import retriever


class SyntheticFallbackCleanupTests(unittest.TestCase):
    @staticmethod
    def _template(article: str) -> list[dict]:
        return [
            {
                "doc_id": "constitution",
                "filename": "constitution_of_india.pdf",
                "page": 42,
                "page_start": 42,
                "page_end": 42,
                "chunk_id": 999,
                "text": "",
                "image_path": "page_images/constitution/page_0042.png",
                "chunk_type": "article",
                "article_id": article,
                "source_extraction": "sarvam_ocr",
            }
        ]

    def test_article_21_fallback_is_marked_synthetic_and_uncited(self):
        query = "What was substituted in Article 21"
        chunk = retriever._amendment_not_found_chunk("21", query, self._template("21"))
        prompt = retriever.build_rag_prompt(query, [chunk])

        self.assertTrue(chunk["synthetic"])
        self.assertEqual(chunk["synthetic_reason"], "no_exact_amendment_evidence")
        self.assertFalse(chunk["display_citation"])
        self.assertIsNone(chunk["page"])
        self.assertIsNone(chunk["page_start"])
        self.assertIsNone(chunk["page_end"])
        self.assertEqual(chunk["image_path"], "")
        self.assertEqual(retriever._chunk_citation_header(chunk), "")
        self.assertIn("No substitution/amendment note for Article 21", chunk["text"])
        self.assertIn("synthetic retrieval note", prompt[1]["content"].lower())
        self.assertNotIn("p:", prompt[1]["content"])

    def test_article_21a_fallback_is_marked_synthetic_and_uncited(self):
        query = "What was inserted in Article 21A"
        chunk = retriever._amendment_not_found_chunk("21A", query, self._template("21A"))
        prompt = retriever.build_rag_prompt(query, [chunk])

        self.assertTrue(chunk["synthetic"])
        self.assertEqual(chunk["synthetic_reason"], "no_exact_amendment_evidence")
        self.assertFalse(chunk["display_citation"])
        self.assertIsNone(chunk["page"])
        self.assertIsNone(chunk["page_start"])
        self.assertIsNone(chunk["page_end"])
        self.assertEqual(chunk["image_path"], "")
        self.assertEqual(retriever._chunk_citation_header(chunk), "")
        self.assertIn("No insertion/amendment note for Article 21A", chunk["text"])
        self.assertIn("synthetic retrieval note", prompt[1]["content"].lower())
        self.assertNotIn("p:", prompt[1]["content"])


if __name__ == "__main__":
    unittest.main()
