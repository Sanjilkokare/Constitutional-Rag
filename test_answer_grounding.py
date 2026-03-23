import unittest

import retriever


def _sample_chunks() -> list[dict]:
    return [
        {
            "filename": "constitution of india.pdf",
            "article_id": "21",
            "page": 42,
            "chunk_type": "article",
            "text": (
                "Article 21. Protection of life and personal liberty. "
                "No person shall be deprived of his life or personal liberty "
                "except according to procedure established by law."
            ),
        },
        {
            "filename": "constitution of india.pdf",
            "article_id": "20",
            "page": 41,
            "chunk_type": "article",
            "text": (
                "Article 20. Protection in respect of conviction for offences. "
                "No person shall be convicted of any offence except for violation "
                "of a law in force at the time of the commission of the act."
            ),
        },
        {
            "filename": "constitution of india.pdf",
            "article_id": "22",
            "page": 43,
            "chunk_type": "article",
            "text": (
                "Article 22. Protection against arrest and detention in certain cases. "
                "No person who is arrested shall be detained in custody without being "
                "informed of the grounds for such arrest."
            ),
        },
    ]


def _sample_amendment_chunks() -> list[dict]:
    return [
        {
            "filename": "constitution of india.pdf",
            "article_id": "22",
            "page": 42,
            "page_start": 42,
            "page_end": 42,
            "chunk_type": "amendment",
            "text": (
                "1. Subs. by the Constitution (First Amendment) Act, 1951, s. 3, "
                "for certain words (w.e.f. 18-6-1951)."
            ),
        },
        {
            "filename": "constitution of india.pdf",
            "article_id": "360",
            "page": 248,
            "page_start": 248,
            "page_end": 248,
            "chunk_type": "amendment",
            "text": (
                "1. Ins. by the Constitution (Forty-fourth Amendment) Act, 1978, "
                "s. 40 (w.e.f. 20-6-1979)."
            ),
        },
    ]


def _synthetic_fallback_chunk(article: str, text: str) -> list[dict]:
    return [
        {
            "filename": "constitution of india.pdf",
            "article_id": article,
            "chunk_id": -1,
            "chunk_type": "amendment",
            "text": text,
            "synthetic": True,
            "synthetic_reason": "no_exact_amendment_evidence",
            "display_citation": False,
            "page": None,
            "page_start": None,
            "page_end": None,
        }
    ]


class AnswerGroundingTests(unittest.TestCase):
    def test_build_rag_prompt_requires_strict_grounding(self):
        messages = retriever.build_rag_prompt(
            "Explain Article 21 in detail. Include its text, scope, and how it differs from Article 20 and Article 22.",
            _sample_chunks(),
        )
        system_msg = messages[0]["content"]

        self.assertIn("Not found in provided context.", system_msg)
        self.assertIn("Do NOT add case law", system_msg)
        self.assertIn("Do not output <think>", system_msg)
        self.assertIn("Copy citations exactly", system_msg)
        self.assertIn("Do NOT mention privacy, dignity, health, pollution-free environment, Maneka Gandhi", system_msg)

    def test_build_rag_prompt_adds_constitution_only_constraints_when_requested(self):
        messages = retriever.build_rag_prompt(
            "Based only on the Constitution, what protections does Article 22 provide?",
            _sample_chunks(),
        )
        system_msg = messages[0]["content"]
        user_msg = messages[1]["content"]

        self.assertIn("Constitution-only mode is active.", system_msg)
        self.assertIn("Do NOT use IPC, CrPC, Evidence Act, other statutes, case law", system_msg)
        self.assertIn("Not found in retrieved constitutional context.", system_msg)
        self.assertIn("CONSTITUTION-ONLY MODE:", user_msg)

    def test_build_rag_prompt_uses_clean_compact_citations_for_real_chunks(self):
        messages = retriever.build_rag_prompt(
            "Give amendment history of Article 22",
            _sample_amendment_chunks()[:1],
        )
        user_msg = messages[1]["content"]

        self.assertIn("[constitution of india.pdf | article:22 | p:42 | amendment]", user_msg)
        self.assertNotIn("doc_id:", user_msg)
        self.assertNotIn("type:", user_msg)

    def test_clean_rag_answer_strips_think_and_normalizes_citations(self):
        raw_answer = (
            "<think>hidden reasoning</think>\n"
            "Article 20 protects against conviction for offences only under law in force "
            "[doc\n constitution of india.pdf | doc_id:4e285f984c48 | article:20 | p:41 | type:article ]."
        )

        cleaned = retriever.clean_rag_answer(raw_answer, _sample_chunks())

        self.assertNotIn("<think>", cleaned)
        self.assertNotIn("</think>", cleaned)
        self.assertIn("[constitution of india.pdf | article:20 | p:41]", cleaned)
        self.assertNotIn("doc_id:", cleaned)
        self.assertNotIn("type:", cleaned)

    def test_article_21_example_drops_unsupported_expansion(self):
        raw_answer = (
            "<think>chain of thought</think>\n"
            "Article 21 says that no person shall be deprived of life or personal liberty "
            "except according to procedure established by law. "
            "[doc constitution of india.pdf | article:21 | p:42 | type:article]\n"
            "It has been expanded to include privacy, dignity, health, pollution-free environment, "
            "and Maneka Gandhi broadened it.\n"
            "Article 20 concerns conviction for offences. "
            "[doc:constitution of india.pdf | article:20 | p:41]\n"
            "Article 22 concerns arrest and detention. "
            "[doc:constitution of india.pdf | article:22 | p:43]"
        )

        cleaned = retriever.clean_rag_answer(raw_answer, _sample_chunks())

        self.assertNotIn("<think>", cleaned)
        self.assertIn(
            "Article 21 says that no person shall be deprived of life or personal liberty",
            cleaned,
        )
        self.assertIn("[constitution of india.pdf | article:21 | p:42]", cleaned)
        self.assertIn("Article 20 concerns conviction for offences.", cleaned)
        self.assertIn("Article 22 concerns arrest and detention.", cleaned)
        self.assertNotIn("privacy", cleaned.lower())
        self.assertNotIn("dignity", cleaned.lower())
        self.assertNotIn("health", cleaned.lower())
        self.assertNotIn("pollution-free environment", cleaned.lower())
        self.assertNotIn("maneka gandhi", cleaned.lower())

    def test_clean_rag_answer_keeps_real_amendment_citation_compact(self):
        raw_answer = (
            "Article 22 has an amendment note "
            "[doc constitution of india.pdf | doc_id:4e285f984c48 | article:22 | p:42 | type:amendment]."
        )

        cleaned = retriever.clean_rag_answer(raw_answer, _sample_amendment_chunks()[:1])

        self.assertIn("[constitution of india.pdf | article:22 | p:42 | amendment]", cleaned)
        self.assertNotIn("doc_id:", cleaned)
        self.assertNotIn("type:", cleaned)

    def test_synthetic_only_fallback_answer_is_plain_and_uncited_for_article_21(self):
        chunks = _synthetic_fallback_chunk(
            "21",
            "No substitution/amendment note for Article 21 was found in the retrieved Constitution context.",
        )
        raw_answer = (
            'No substitution/amendment note for Article 21 was found. '
            '<citation type="synthetic_retrieval_note" chunk="1">'
            '(synthetic retrieval note for Article 21; reason: no_exact_amendment_evidence)'
            "</citation>"
        )

        cleaned = retriever.clean_rag_answer(raw_answer, chunks)

        self.assertEqual(
            cleaned,
            "No substitution/amendment note for Article 21 was found in the retrieved Constitution context.",
        )
        self.assertNotIn("<citation", cleaned)

    def test_synthetic_only_fallback_answer_is_plain_and_uncited_for_article_21a(self):
        chunks = _synthetic_fallback_chunk(
            "21A",
            "No insertion/amendment note for Article 21A was found in the retrieved Constitution context.",
        )
        raw_answer = (
            'No insertion/amendment note for Article 21A was found. '
            '<citation type="synthetic_retrieval_note" chunk="1">'
            '(synthetic retrieval note for Article 21A; reason: no_exact_amendment_evidence)'
            "</citation>"
        )

        cleaned = retriever.clean_rag_answer(raw_answer, chunks)

        self.assertEqual(
            cleaned,
            "No insertion/amendment note for Article 21A was found in the retrieved Constitution context.",
        )
        self.assertNotIn("<citation", cleaned)

    def test_constitution_only_cleanup_drops_external_law_and_case_law_lines(self):
        raw_answer = (
            "Article 22 says a person arrested must be informed of the grounds for arrest. "
            "[constitution of india.pdf | article:22 | p:43]\n"
            "Under the CrPC, the police must also follow arrest procedure.\n"
            "Case law has expanded these safeguards."
        )

        cleaned = retriever.clean_rag_answer(
            raw_answer,
            _sample_chunks(),
            query="Constitution only: can police arrest without informing grounds?",
        )

        self.assertIn("Article 22 says a person arrested must be informed of the grounds for arrest.", cleaned)
        self.assertNotIn("CrPC", cleaned)
        self.assertNotIn("Case law", cleaned)

    def test_constitution_only_cleanup_returns_constitutional_fallback_when_only_external_law_remains(self):
        cleaned = retriever.clean_rag_answer(
            "Under the CrPC, an arrested person must be produced before a magistrate.",
            _sample_chunks(),
            query="Based only on the Constitution, what protections does Article 22 provide?",
        )

        self.assertEqual(cleaned, "Not found in retrieved constitutional context.")

    def test_non_constitution_only_query_keeps_existing_behavior(self):
        raw_answer = "Under the CrPC, an arrested person must be produced before a magistrate."

        cleaned = retriever.clean_rag_answer(
            raw_answer,
            _sample_chunks(),
            query="Can police arrest without informing grounds?",
        )

        self.assertEqual(cleaned, raw_answer)


if __name__ == "__main__":
    unittest.main()
