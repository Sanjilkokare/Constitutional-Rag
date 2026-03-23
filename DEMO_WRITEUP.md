# Demo Summary

This project is a structure-aware RAG system for the Constitution of India. It combines legal-aware chunking, a persisted legal index, direct routing for articles and structural identifiers, hybrid fallback retrieval, Constitution-only answer mode, amendment-aware retrieval, and source-grounded answer rendering inside a Chainlit UI.

The strongest result is not just that it answers article questions. The stronger engineering outcome is that the system was debugged and tightened around the failure modes that usually break long-PDF legal QA: incomplete article retrieval, article boundary bleed such as `21` vs `21A`, missing structural mappings, amendment-history contamination, and noisy concept-query fallback.

## What Was Built

- direct article retrieval with full-article aggregation
- deterministic multi-article expansion for selected constitutional interactions
- strict amendment-history retrieval for exact articles
- Constitution-only answer mode
- cleaned citation and synthetic fallback handling
- persisted-store validation for metadata, FAISS, and legal-index alignment
- article-boundary and structural-index cleanup
- fixed reusable evaluation benchmark
- small BM25 / hybrid refinement guarded by that benchmark

## What Was Actually Validated

The current validated workspace snapshot uses:

- `constitution of india.pdf`
- `402` pages
- `2215` persisted chunks
- `408` indexed article mappings
- a Step 8 evaluation set with `24/24` passing items

Important limitation:

- the validated benchmark runs were performed on a **PyMuPDF-backed corpus**
- they were **not** validated on a Sarvam-OCR-backed corpus in the current environment
- Sarvam OCR was not usable for the final validated runs, so the store was built through fallback extraction

## Honest Positioning

This is a grounded constitutional QA system over a single legal corpus. It is suitable for demoing legal-document retrieval design, structure-aware indexing, retrieval debugging, and evaluation discipline.

It should not be positioned as:

- a general legal research engine
- a case-law reasoning assistant
- a guarantee of legal correctness beyond retrieved constitutional context
- a production-validated legal platform

## Concise LinkedIn / Showcase Version

Built a structure-aware RAG system for the Constitution of India with Chainlit, FAISS, MiniLM embeddings, legal-aware chunking, and direct routing for Articles / Schedules / Lists / Entries.

The main work was not just "chat with PDF." It was fixing the retrieval layer so constitutional questions behave like legal-structure queries instead of generic semantic search:

- full-article aggregation for direct article questions
- exact-article amendment retrieval without neighboring bleed
- Constitution-only answer mode
- structural routing for schedules, lists, and entries
- persisted-store validation for FAISS, metadata, and legal-index alignment
- a fixed evaluation set to measure regressions

Current benchmark status: `24/24` evaluation items passing on the active Constitution store.

Important caveat: the final validated runs in this workspace were done on a **PyMuPDF-backed corpus**, not a Sarvam-OCR-backed corpus, because Sarvam OCR was not usable in the current environment.
