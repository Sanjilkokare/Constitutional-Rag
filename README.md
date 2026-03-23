# Constitution RAG with Legal Structure Routing

RAG-Sarvam is a Constitution of India question-answering system built around a single legal corpus, structure-aware chunking, direct legal-index routing, and a Chainlit UI.

The current codebase is designed for grounded constitutional QA over a persisted local store:

- PDF ingestion with Sarvam OCR when available, with PyMuPDF fallback
- Legal-aware chunking for articles, clauses, schedules, lists, and entries
- Persisted FAISS vectors plus JSON metadata and legal index
- Direct identifier routing for explicit article and structural queries
- Hybrid fallback retrieval for concept queries
- Amendment-aware retrieval strictness
- Constitution-only answer mode
- Clean source citation rendering plus synthetic retrieval notes for grounded fallbacks

## Current Validated Status

This repository has been validated on the active Constitution corpus in this workspace, but the validation status matters:

- The current validated corpus is **PyMuPDF-backed**, not Sarvam-OCR-backed.
- In the current environment, Sarvam OCR was not usable for the final validated runs, so ingestion fell back to PyMuPDF extraction.
- The active store in this workspace currently contains:
  - `1` Constitution PDF
  - `402` pages
  - `2215` persisted chunks
  - `408` indexed article mappings
  - `12` schedules
  - `3` lists
  - `121` entry mappings
- The fixed Step 8 evaluation set currently passes `24/24` items on this active store.

What this does **not** mean:

- It is not a general legal research assistant.
- It is not validated for case-law reasoning.
- It is not a guarantee of constitutional correctness beyond retrieved context.
- It is not validated here as a Sarvam-OCR-backed benchmarked corpus.

## Problem Statement

Naive PDF RAG over the Constitution tends to fail on:

- long articles split across chunks and pages
- article suffix boundaries such as `21` vs `21A`
- amendment notes mixed with substantive text
- structural queries over schedules, lists, and entries
- interaction questions that require multiple linked constitutional provisions

This project addresses those failures with structure-aware indexing and constrained answer grounding.

## Architecture Summary

### 1. Ingestion Flow

The ingestion pipeline lives in [ingest.py](./ingest.py).

High-level flow:

1. Compute a PDF hash and skip if already ingested.
2. Render each PDF page to PNG with PyMuPDF for source-page previews.
3. Attempt OCR through Sarvam Document Intelligence.
4. Fall back to PyMuPDF text extraction when OCR fails or returns unusable text.
5. Normalize page text.
6. Chunk each page with the legal-aware chunker.
7. Embed chunks with MiniLM.
8. Persist vectors, chunk metadata, document manifest, and legal index.

Relevant files:

- [ingest.py](./ingest.py)
- [chunker.py](./chunker.py)
- [storage.py](./storage.py)
- [legal_index.py](./legal_index.py)
- [embeddings.py](./embeddings.py)

### 2. Chunking and Legal Indexing

The system tries to preserve legal structure during ingestion:

- `article` and `clause` chunks for substantive provisions
- `amendment` / `footnote` chunks for editorial history
- `heading` chunks for structural anchors
- `schedule_id`, `list_id`, and `entry_id` metadata for structural routing

The legal index stores direct mappings for:

- articles
- parts
- schedules
- lists
- entries
- identifier reference tags

This is what enables direct retrieval for explicit queries instead of relying on vector similarity luck.

### 3. Retrieval Flow

The retriever lives in [retriever.py](./retriever.py).

High-level flow:

1. Parse the query for explicit legal identifiers.
2. Apply deterministic constitutional expansions when needed.
3. If identifiers are present, use direct legal-index routing first.
4. For article queries, materialize full article bundles in article order.
5. For amendment-history queries, retrieve only amendment/editorial evidence for the exact article, or return a grounded synthetic fallback note.
6. If no direct route applies, fall back to hybrid retrieval:
   - FAISS vector search
   - lightweight BM25 lexical scoring
   - score merge with chunk-type and noise penalties

### 4. Answer Grounding Flow

The answer layer also lives in [retriever.py](./retriever.py) and is used by [app.py](./app.py).

It currently includes:

- strict prompt construction from retrieved context only
- Constitution-only mode for explicit constitutional-only requests
- synthetic fallback handling for missing amendment evidence
- citation normalization and cleanup
- source-page rendering in the Chainlit UI
- separate `Retrieval Notes` handling for synthetic fallback chunks

## Current Capabilities

### Direct Article Retrieval

The system supports direct article retrieval and full-article aggregation for explicit article queries such as:

- `What does Article 22 say?`
- `Explain Article 21 in detail.`
- `What does Article 360 provide?`

### Multi-Article and Interaction Queries

The system supports forced multi-article retrieval for explicit lists, for example:

- `Explain Articles 20, 21, and 22 together.`
- `Explain how Articles 13, 32, 226, and 368 interact.`

It also includes narrow deterministic query expansion rules for selected constitutional interactions, such as:

- `Article 352 + Articles 19/20/21` -> include `358` and `359`
- arrest/detention + remedy/enforcement language -> include `22` and, where appropriate, `32`

### Amendment-Aware Retrieval

Supported amendment-history style queries include:

- `Give amendment history of Article 22.`
- `Show footnotes for Article 360.`
- `What was substituted in Article 21?`
- `What was inserted in Article 21A?`

Behavior:

- retrieve only amendment/editorial evidence for the exact requested article
- do not use neighboring article text as amendment evidence
- return a grounded fallback when no exact amendment evidence exists

### Constitution-Only Mode

Explicit Constitution-only phrasing activates a stricter answer mode, for example:

- `Based only on the Constitution, what protections does Article 22 provide?`
- `Constitution only: can police arrest without informing grounds?`
- `Answer strictly from the Constitution: what happens to Articles 20 and 21 during Emergency?`

In this mode, the prompt and answer-cleaning path are instructed to stay inside retrieved constitutional context and avoid unsupported external law or case-law references.

### Structural Routing

Supported structural lookup patterns include:

- schedules
- List I / II / III queries
- list-entry queries such as `Entry 10 of List I`

The structural index is materially cleaner than earlier iterations, but structural lookups can still return neighboring structural context rather than only a single isolated entry.

## Supported Query Types

The current system is strongest on:

- direct article queries
- explicit multi-article queries
- amendment-history / footnote queries
- Constitution-only constitutional questions
- schedule / list / entry queries
- a subset of concept queries where the Constitution mapping is explicit enough to recover via direct routing or hybrid fallback

## Repository Structure

Core files:

- [app.py](./app.py): Chainlit app
- [ingest.py](./ingest.py): ingestion pipeline
- [retriever.py](./retriever.py): retrieval, prompt building, answer cleanup
- [legal_index.py](./legal_index.py): legal identifier extraction and persisted structural index
- [chunker.py](./chunker.py): chunking logic
- [storage.py](./storage.py): FAISS and metadata persistence
- [sarvam_client.py](./sarvam_client.py): Sarvam OCR and chat client
- [validate_persisted_store.py](./validate_persisted_store.py): persisted-store sanity checker
- [run_constitution_eval.py](./run_constitution_eval.py): Step 8 evaluation runner
- [constitution_eval_set.json](./constitution_eval_set.json): fixed evaluation set

Persisted data:

- [data/documents.json](./data/documents.json)
- [data/faiss_index/metadata.json](./data/faiss_index/metadata.json)
- [data/faiss_index/legal_index.json](./data/faiss_index/legal_index.json)
- [data/faiss_index/index.faiss](./data/faiss_index/index.faiss)

## Setup

### Prerequisites

- Python 3.10+
- a local install of the dependencies in [requirements.txt](./requirements.txt)
- a Sarvam API key if you want to use Sarvam OCR and Sarvam chat
- a locally cached SentenceTransformer model for `all-MiniLM-L6-v2`

Important:

- [embeddings.py](./embeddings.py) now loads the embedding model with `local_files_only=True`.
- That means hybrid retrieval and the evaluation runner expect the MiniLM model to already exist in the local Hugging Face / SentenceTransformer cache.
- If the model is not cached locally, hybrid queries will fail until it is downloaded in an environment with network access.

### Install

From the repo root:

```bash
pip install -r requirements.txt
```

### Configure

Create `.env` from [.env.example](./.env.example):

```powershell
Copy-Item .env.example .env
```

Then set at least:

- `SARVAM_API_KEY`

Useful config fields are defined in [config.py](./config.py), including:

- OCR page batching threshold
- chunk size and overlap
- `TOP_K`
- `VECTOR_WEIGHT`
- `BM25_WEIGHT`
- `FOOTNOTE_DOWNWEIGHT`

## Run the App

From the repo root:

```bash
chainlit run app.py
```

Then open the Chainlit UI and upload the Constitution PDF.

## Ingest the Constitution PDF

### Normal UI Workflow

1. Start the Chainlit app with `chainlit run app.py`.
2. Upload `constitution of india.pdf`.
3. Wait for ingestion to finish.
4. Ask article, amendment, structural, or Constitution-only questions.

### Scripted Ingestion Workflow

If you want to rebuild the active store without using the UI, run a small Python wrapper from PowerShell:

```powershell
@'
from pathlib import Path
import asyncio
import ingest

pdf_path = Path(r"C:\Users\kokar\Documents\constitution of india.pdf")

async def main():
    rec = await ingest.ingest_pdf(pdf_path.read_bytes(), pdf_path.name)
    print(rec)

asyncio.run(main())
'@ | python -
```

That uses the real ingestion pipeline in [ingest.py](./ingest.py) and writes to the active repo-local store under `data/`.

## Reset and Regenerate the Corpus

The active persisted store is repo-local because [config.py](./config.py) resolves `BASE_DIR` from the project directory.

Active paths:

- `data/documents.json`
- `data/faiss_index/index.faiss`
- `data/faiss_index/metadata.json`
- `data/faiss_index/legal_index.json`
- `data/page_images/`

### Manual Reset

To rebuild from scratch:

1. Stop the app.
2. Delete:
   - `data/faiss_index/index.faiss`
   - `data/faiss_index/metadata.json`
   - `data/faiss_index/legal_index.json`
3. Reset `data/documents.json` to an empty JSON list:
   - `[]`
4. Optionally clear `data/page_images/`.
5. Re-ingest the Constitution PDF through the UI or scripted workflow above.

### PowerShell Reset Example

```powershell
Remove-Item .\data\faiss_index\index.faiss -Force -ErrorAction SilentlyContinue
Remove-Item .\data\faiss_index\metadata.json -Force -ErrorAction SilentlyContinue
Remove-Item .\data\faiss_index\legal_index.json -Force -ErrorAction SilentlyContinue
Set-Content .\data\documents.json "[]"
Remove-Item .\data\page_images\* -Force -ErrorAction SilentlyContinue
```

## Evaluation and Validation

### Persisted-Store Validation

Run the validator on the active store:

```bash
python validate_persisted_store.py
```

The validator checks:

- documents manifest presence
- metadata integrity
- FAISS / metadata alignment
- legal index reference integrity
- target article-family mappings
- structural sample mappings

### Step 8 Evaluation Benchmark

Run the fixed evaluation set:

```bash
python run_constitution_eval.py --output constitution_eval_results.json
```

The runner uses:

- [constitution_eval_set.json](./constitution_eval_set.json)
- the active repo-local persisted store
- deterministic retrieval and prompt-contract checks

Current validated result on the active store:

- `24 / 24` evaluation items passed

What the evaluation covers:

- direct article retrieval
- multi-article retrieval
- amendment-history retrieval
- emergency interaction routing
- Constitution-only mode activation
- structural routing
- repaired boundary families

What it does **not** currently do:

- human-grade answer scoring
- live Sarvam answer-quality benchmarking
- Sarvam-OCR-backed corpus validation in this workspace

## Example Queries

### Article Queries

- `What does Article 22 say?`
- `Explain Article 21 in detail.`
- `What does Article 360 provide?`

### Multi-Article Queries

- `Explain Articles 20, 21, and 22 together.`
- `Explain how Articles 13, 32, 226, and 368 interact.`

### Amendment Queries

- `Give amendment history of Article 22.`
- `Show footnotes for Article 360.`
- `What was substituted in Article 21?`
- `What was inserted in Article 21A?`

### Constitution-Only Queries

- `Based only on the Constitution, what protections does Article 22 provide?`
- `Constitution only: can police arrest without informing grounds?`
- `Based only on the Constitution, what is the remedy for violation of fundamental rights?`

### Structural Queries

- `What does Schedule I contain?`
- `What is in List III?`
- `What is Entry 10 of List I?`
- `What does Entry I:1 cover?`

## What Has Actually Been Validated

Validated in this workspace:

- persisted-store loading and alignment
- repaired article boundaries for the major known defect families
- structural cleanup for schedules / lists / entries
- direct article bundle retrieval
- amendment-aware retrieval strictness
- Constitution-only mode activation and prompt contract
- citation cleanup and synthetic retrieval-note suppression
- hybrid retrieval refinement guarded by the fixed Step 8 evaluation set

Current benchmarked store facts:

- single corpus: Constitution of India PDF
- persisted active store under repo-local `data/`
- all current chunk records show `source_extraction="pymupdf"`

## Known Limitations

These limitations are important and intentional to state clearly:

- The current validated runs were performed on a **PyMuPDF-backed corpus**, not a Sarvam-OCR-backed corpus.
- Sarvam OCR was not usable for the final validated runs in this environment, so the benchmarked corpus was built through fallback extraction.
- The system is limited to the retrieved corpus. It is not a case-law research engine.
- Answer quality depends on retrieved context quality.
- Structural queries can still include neighboring schedule/list/entry context.
- Concept queries can still include nearby constitutional context even when the top result is correct.
- The evaluation runner is a deterministic retrieval / prompt-contract benchmark, not a human-graded legal-answer benchmark.
- Hybrid queries depend on the embedding model being present in the local cache.
- This repository does not establish production-grade legal reliability.

## Future Work

Reasonable next steps after the current validated state:

- rerun ingestion and benchmarking on a Sarvam-OCR-backed corpus once OCR is stable and available
- tighten structural entry isolation for neighboring entry noise
- expand evaluation coverage for more constitutional concept queries
- add a small reproducible ingestion CLI wrapper if repeated rebuilds become common
- add answer-quality review on top of the existing retrieval benchmark

## Demo Positioning

This project is strongest when presented as:

- a structure-aware Constitution RAG system
- a retrieval-debugging and indexing-quality exercise
- a grounded constitutional QA demo with explicit validation artifacts

It should **not** be presented as:

- a general legal reasoning platform
- a case-law assistant
- a fully validated production legal product

