"""
Persisted-store validator for the Constitution RAG corpus.

Checks:
  - documents.json
  - metadata.json
  - legal_index.json
  - FAISS index
  - metadata/index/legal-index alignment
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import faiss

import config
import legal_index as li

REQUIRED_METADATA_FIELDS = (
    "doc_id",
    "filename",
    "page",
    "chunk_id",
    "text",
    "image_path",
    "page_start",
    "page_end",
    "chunk_type",
    "article_id",
    "schedule_id",
    "list_id",
    "entry_id",
    "section_id",
    "source_extraction",
)
TARGET_ARTICLES = ("21", "21A", "22", "32", "226", "352", "358", "359", "360", "368")
HEADING_ARTICLE_RE = re.compile(
    r"(?im)^\[Article\s+(\d+[A-Z]?)\]|^(?:Article|Art\.?)\s+(\d+[A-Z]?)\b|^(\d+[A-Z]?)\.\s+[A-Z]"
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _snippet(text: str, limit: int = 140) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:limit]


def _print_header(title: str):
    print(f"\n{title}")


def _resolve_paths(base_dir: Path | None = None, data_dir: Path | None = None) -> dict[str, Path]:
    if data_dir is not None:
        resolved_data_dir = data_dir.resolve()
        resolved_base_dir = resolved_data_dir.parent
    elif base_dir is not None:
        resolved_base_dir = base_dir.resolve()
        resolved_data_dir = resolved_base_dir / "data"
    else:
        resolved_base_dir = config.BASE_DIR.resolve()
        resolved_data_dir = config.DATA_DIR.resolve()

    faiss_index_dir = resolved_data_dir / "faiss_index"
    return {
        "base_dir": resolved_base_dir,
        "data_dir": resolved_data_dir,
        "documents.json": resolved_data_dir / "documents.json",
        "metadata.json": faiss_index_dir / "metadata.json",
        "legal_index.json": faiss_index_dir / "legal_index.json",
        "index.faiss": faiss_index_dir / "index.faiss",
    }


def _load_artifacts(base_dir: Path | None = None, data_dir: Path | None = None):
    resolved_paths = _resolve_paths(base_dir=base_dir, data_dir=data_dir)

    artifact_paths = {
        "documents.json": resolved_paths["documents.json"],
        "metadata.json": resolved_paths["metadata.json"],
        "legal_index.json": resolved_paths["legal_index.json"],
        "index.faiss": resolved_paths["index.faiss"],
    }

    missing = [name for name, path in artifact_paths.items() if not path.exists()]
    _print_header("FILES")
    print(f"base_dir={resolved_paths['base_dir']}")
    print(f"data_dir={resolved_paths['data_dir']}")
    for name, path in artifact_paths.items():
        print(
            f"{name}: path={path} exists={path.exists()} "
            f"size={path.stat().st_size if path.exists() else None}"
        )

    if missing:
        return {
            "paths": resolved_paths,
            "missing": missing,
            "documents": _read_json(artifact_paths["documents.json"]) if artifact_paths["documents.json"].exists() else [],
            "metadata": None,
            "legal_index": None,
            "index": None,
        }

    return {
        "paths": resolved_paths,
        "missing": [],
        "documents": _read_json(artifact_paths["documents.json"]),
        "metadata": _read_json(artifact_paths["metadata.json"]),
        "legal_index": _read_json(artifact_paths["legal_index.json"]),
        "index": faiss.read_index(str(artifact_paths["index.faiss"])),
    }


def _validate_metadata(metadata: list[dict]) -> dict:
    issues = {
        "missing_required": Counter(),
        "wrong_types": Counter(),
        "empty_text_non_heading": [],
        "page_issues": [],
        "chunk_id_row_mismatch": [],
        "duplicate_chunk_ids": [],
        "chunk_type_counts": Counter(),
        "doc_ids": defaultdict(set),
        "article_chunks": defaultdict(list),
    }
    seen_chunk_ids: dict[int, int] = {}

    for idx, row in enumerate(metadata):
        for field in REQUIRED_METADATA_FIELDS:
            if field not in row:
                issues["missing_required"][field] += 1

        for field in ("chunk_id", "page", "page_start", "page_end"):
            if not isinstance(row.get(field), int):
                issues["wrong_types"][f"{field}_not_int"] += 1

        if idx != row.get("chunk_id"):
            issues["chunk_id_row_mismatch"].append((idx, row.get("chunk_id")))

        cid = row.get("chunk_id")
        if cid in seen_chunk_ids:
            issues["duplicate_chunk_ids"].append((cid, seen_chunk_ids[cid], idx))
        else:
            seen_chunk_ids[cid] = idx

        if not (row.get("text") or "").strip() and row.get("chunk_type") != "heading":
            issues["empty_text_non_heading"].append(idx)

        page = row.get("page")
        page_start = row.get("page_start")
        page_end = row.get("page_end")
        if all(isinstance(v, int) for v in (page, page_start, page_end)):
            if page <= 0 or page_start <= 0 or page_end <= 0 or page_start > page_end or not (page_start <= page <= page_end):
                issues["page_issues"].append((idx, page, page_start, page_end))

        issues["chunk_type_counts"][row.get("chunk_type", "")] += 1
        issues["doc_ids"][row.get("doc_id", "")].add(row.get("filename", ""))

        article_id = row.get("article_id") or ""
        if article_id:
            issues["article_chunks"][article_id].append(idx)

    return issues


def _validate_legal_index(metadata: list[dict], data: dict) -> dict:
    invalid_refs = defaultdict(list)
    mismatches = {
        "articles": [],
        "schedules": [],
        "lists": [],
        "entries": [],
    }

    for group_name in ("articles", "parts", "schedules", "lists", "entries", "tags"):
        group = data.get(group_name, {})
        for key, refs in group.items():
            if not isinstance(refs, list):
                invalid_refs[group_name].append((key, "not_list", type(refs).__name__))
                continue
            for cid in refs:
                if not isinstance(cid, int):
                    invalid_refs[group_name].append((key, "non_int", cid))
                    continue
                if not (0 <= cid < len(metadata)):
                    invalid_refs[group_name].append((key, "out_of_range", cid))
                    continue
                row = metadata[cid]
                if group_name == "articles":
                    if (row.get("article_id") or "").upper() != key.upper():
                        mismatches["articles"].append(
                            (key, cid, row.get("article_id"), row.get("chunk_type"), _snippet(row.get("text", "")))
                        )
                elif group_name == "schedules":
                    if (row.get("schedule_id") or "").upper() != key.upper() and key.upper() != "VII":
                        mismatches["schedules"].append((key, cid, row.get("schedule_id"), row.get("list_id"), row.get("entry_id")))
                elif group_name == "lists":
                    if (row.get("list_id") or "").upper() != key.upper():
                        mismatches["lists"].append((key, cid, row.get("list_id"), row.get("entry_id")))
                elif group_name == "entries":
                    list_id, entry_id = key.split(":", 1)
                    if (row.get("list_id") or "").upper() != list_id.upper() or (row.get("entry_id") or "").upper() != entry_id.upper():
                        mismatches["entries"].append((key, cid, row.get("list_id"), row.get("entry_id")))

    return {"invalid_refs": invalid_refs, "mismatches": mismatches}


def _discover_heading_articles(metadata: list[dict]) -> dict[str, list[int]]:
    hits: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(metadata):
        match = HEADING_ARTICLE_RE.search(row.get("text") or "")
        if match:
            article = next(group for group in match.groups() if group)
            hits[article.upper()].append(idx)
    return hits


def _print_target_articles(metadata: list[dict], legal_index_data: dict, heading_hits: dict[str, list[int]]):
    _print_header("TARGET ARTICLES")
    for article in TARGET_ARTICLES:
        refs = legal_index_data.get("articles", {}).get(article, [])
        print(f"Article {article}: legal_index_refs={refs}")
        if not refs:
            print(f"  inferred_heading_hits={heading_hits.get(article, [])[:10]}")
        for cid in refs[:10]:
            row = metadata[cid]
            print(
                " ",
                {
                    "cid": cid,
                    "article_id": row.get("article_id"),
                    "page": row.get("page_start"),
                    "type": row.get("chunk_type"),
                    "text": _snippet(row.get("text", "")),
                },
            )


def _print_structural_samples(metadata: list[dict], legal_index_data: dict):
    _print_header("STRUCTURAL SAMPLES")
    for label, key in (("schedule", "schedules"), ("list", "lists"), ("entry", "entries")):
        mapping = legal_index_data.get(key, {})
        if not mapping:
            print(f"{label}: no mappings")
            continue
        first_key = sorted(mapping.keys())[0]
        refs = mapping[first_key][:5]
        print(f"{label} {first_key}: refs={refs}")
        for cid in refs:
            row = metadata[cid]
            print(
                " ",
                {
                    "cid": cid,
                    "schedule_id": row.get("schedule_id"),
                    "list_id": row.get("list_id"),
                    "entry_id": row.get("entry_id"),
                    "page": row.get("page_start"),
                    "type": row.get("chunk_type"),
                    "text": _snippet(row.get("text", "")),
                },
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, help="Project base directory containing data/")
    parser.add_argument("--data-dir", type=Path, help="Explicit data directory containing documents.json and faiss_index/")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifacts = _load_artifacts(base_dir=args.base_dir, data_dir=args.data_dir)
    documents = artifacts["documents"]

    _print_header("DOCUMENTS")
    print(f"documents_count={len(documents)}")
    print(documents)

    if artifacts["missing"]:
        _print_header("STATUS")
        print(f"Missing persisted artifacts: {artifacts['missing']}")
        print("Target article and structural mapping validation cannot run until those files are restored.")
        return 1

    metadata = artifacts["metadata"]
    legal_index_data = artifacts["legal_index"]
    index = artifacts["index"]

    _print_header("COUNTS")
    print(f"metadata_count={len(metadata)}")
    print(f"faiss_ntotal={index.ntotal}")
    print(f"faiss_dim={index.d}")
    for key in ("articles", "parts", "schedules", "lists", "entries", "tags"):
        print(f"legal_index_{key}={len(legal_index_data.get(key, {}))}")

    metadata_issues = _validate_metadata(metadata)
    _print_header("METADATA INTEGRITY")
    print(f"missing_required={dict(metadata_issues['missing_required'])}")
    print(f"wrong_types={dict(metadata_issues['wrong_types'])}")
    print(f"empty_text_non_heading={len(metadata_issues['empty_text_non_heading'])} sample={metadata_issues['empty_text_non_heading'][:10]}")
    print(f"page_issues={len(metadata_issues['page_issues'])} sample={metadata_issues['page_issues'][:10]}")
    print(f"chunk_id_row_mismatch={len(metadata_issues['chunk_id_row_mismatch'])} sample={metadata_issues['chunk_id_row_mismatch'][:10]}")
    print(f"duplicate_chunk_ids={len(metadata_issues['duplicate_chunk_ids'])} sample={metadata_issues['duplicate_chunk_ids'][:10]}")
    print(f"doc_id_filename_map={ {k: sorted(v) for k, v in metadata_issues['doc_ids'].items()} }")
    print(f"chunk_type_counts={dict(metadata_issues['chunk_type_counts'])}")

    _print_header("FAISS ALIGNMENT")
    print(f"metadata_vs_faiss_match={len(metadata) == index.ntotal}")

    legal_index_issues = _validate_legal_index(metadata, legal_index_data)
    _print_header("LEGAL INDEX INTEGRITY")
    for group in ("articles", "parts", "schedules", "lists", "entries", "tags"):
        bad = legal_index_issues["invalid_refs"][group]
        print(f"{group}_invalid_refs={len(bad)} sample={bad[:10]}")
    for group in ("articles", "schedules", "lists", "entries"):
        bad = legal_index_issues["mismatches"][group]
        print(f"{group}_mismatches={len(bad)} sample={bad[:10]}")

    heading_hits = _discover_heading_articles(metadata)
    missing_from_legal = sorted([article for article in heading_hits if article not in legal_index_data.get("articles", {})])
    _print_header("ARTICLE DISCOVERY")
    print(f"metadata_article_ids={len(metadata_issues['article_chunks'])}")
    print(f"heading_detected_articles={len(heading_hits)}")
    print(f"legal_index_articles={len(legal_index_data.get('articles', {}))}")
    print(f"heading_articles_missing_from_legal_index={len(missing_from_legal)} sample={missing_from_legal[:50]}")

    _print_target_articles(metadata, legal_index_data, heading_hits)
    _print_structural_samples(metadata, legal_index_data)

    _print_header("STABILITY SIGNALS")
    print(f"doc_records_chunk_total={sum(doc.get('chunk_count', 0) for doc in documents)}")
    print(f"source_extraction_counts={dict(Counter((row.get('source_extraction') or '') for row in metadata))}")
    print(
        f"articles_with_most_chunks={Counter({article: len(refs) for article, refs in metadata_issues['article_chunks'].items()}).most_common(20)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
