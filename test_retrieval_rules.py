import unittest

import retriever


class RetrievalRuleTests(unittest.TestCase):
    def setUp(self):
        self._orig_embed_query = retriever.embeddings.embed_query
        retriever.embeddings.embed_query = self._fail_on_hybrid

    def tearDown(self):
        retriever.embeddings.embed_query = self._orig_embed_query

    @staticmethod
    def _fail_on_hybrid(query: str):
        raise AssertionError(f"Unexpected hybrid fallback for query: {query}")

    @staticmethod
    def _unique_articles(results: list[dict]) -> list[str]:
        articles: list[str] = []
        for result in results:
            article = result.get("article_id") or result.get("article")
            if article and article not in articles:
                articles.append(article)
        return articles

    @staticmethod
    def _chunk_ids(results: list[dict]) -> list[int]:
        return [result["chunk_id"] for result in results]

    def test_article_22_full_bundle_excludes_amendments(self):
        results = retriever.retrieve("What does Article 22 say?", top_k=8)

        self.assertEqual(self._unique_articles(results), ["22"])
        self.assertTrue(all(r.get("chunk_type") != "amendment" for r in results))
        self.assertIn(44, [r.get("page_start") for r in results])

        combined = " ".join(r["text"] for r in results).lower()
        self.assertNotIn("stand substituted", combined)

    def test_article_360_bundle_repairs_continuation(self):
        results = retriever.retrieve(
            "Explain Article 360 and how it affects State salaries and judiciary",
            top_k=8,
        )

        self.assertEqual(self._unique_articles(results), ["360"])
        self.assertTrue(all(r.get("chunk_type") != "amendment" for r in results))
        self.assertIn(249, [r.get("page_start") for r in results])

        combined = " ".join(r["text"] for r in results).lower()
        self.assertIn("reduction of salaries", combined)
        self.assertIn("high courts", combined)

    def test_article_21_does_not_bleed_into_21a(self):
        results = retriever.retrieve("Explain Article 21 in detail", top_k=8)

        self.assertEqual(self._unique_articles(results), ["21"])
        self.assertEqual(len(results), 1)

    def test_emergency_query_adds_articles_358_and_359(self):
        results = retriever.retrieve(
            "During a National Emergency under Article 352, what happens to Articles 19, 20, and 21?",
            top_k=8,
        )

        unique_articles = self._unique_articles(results)
        self.assertEqual(unique_articles[:6], ["352", "19", "20", "21", "358", "359"])
        self.assertIn("358", unique_articles)
        self.assertIn("359", unique_articles)

    def test_explicit_multi_article_query_forces_all_named_articles(self):
        results = retriever.retrieve(
            "Explain how Articles 13, 32, 226, and 368 interact",
            top_k=8,
        )

        unique_articles = self._unique_articles(results)
        self.assertEqual(unique_articles[:4], ["13", "32", "226", "368"])
        self.assertIn("368", unique_articles)

    def test_arrest_without_information_maps_to_article_22_only(self):
        results = retriever.retrieve("Arrest without information", top_k=8)

        unique_articles = self._unique_articles(results)
        self.assertEqual(unique_articles, ["22"])
        self.assertNotIn("32", unique_articles)

    def test_arrest_remedy_query_adds_article_32(self):
        results = retriever.retrieve(
            "What constitutional remedy exists for arrest without information?",
            top_k=8,
        )

        self.assertEqual(self._unique_articles(results)[:2], ["22", "32"])

    def test_article_22_amendment_history_returns_notes_only(self):
        results = retriever.retrieve("Give amendment history of Article 22", top_k=8)

        self.assertEqual(self._unique_articles(results), ["22"])
        self.assertTrue(all(r.get("chunk_type") == "amendment" for r in results))
        self.assertEqual([r.get("page_start") for r in results], [42, 43, 44])
        self.assertIn("First Amendment", results[0]["text"])

    def test_article_360_footnote_query_returns_note_chunks_only(self):
        results = retriever.retrieve("Show footnotes for Article 360", top_k=8)

        self.assertEqual(self._unique_articles(results), ["360"])
        self.assertTrue(all(r.get("chunk_type") == "amendment" for r in results))
        self.assertEqual([r.get("page_start") for r in results], [248, 248, 249])
        self.assertIn("Forty-fourth Amendment", results[0]["text"])

    def test_article_21_substitution_query_returns_grounded_not_found(self):
        results = retriever.retrieve("What was substituted in Article 21", top_k=8)

        self.assertEqual(self._unique_articles(results), ["21"])
        self.assertEqual(self._chunk_ids(results), [-1])
        self.assertIn("No substitution/amendment note for Article 21", results[0]["text"])
        self.assertNotIn("21A. Right to education", results[0]["text"])

    def test_article_21a_insertion_query_returns_grounded_not_found(self):
        results = retriever.retrieve("What was inserted in Article 21A", top_k=8)

        self.assertEqual(self._unique_articles(results), ["21A"])
        self.assertEqual(self._chunk_ids(results), [-1])
        self.assertIn("No insertion/amendment note for Article 21A", results[0]["text"])


if __name__ == "__main__":
    unittest.main()
