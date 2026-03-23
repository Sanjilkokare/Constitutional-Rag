import argparse
import json
from collections import defaultdict
from pathlib import Path

import legal_index as li
import retriever
import storage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("constitution_eval_set.json"),
        help="Path to the fixed evaluation-set JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Default top_k to use unless an item overrides it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the full JSON report.",
    )
    return parser.parse_args()


def _load_items(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inspect_active_store() -> dict:
    metadata = storage.get_all_metadata()
    legal_idx = li.get_legal_index()
    return {
        "metadata_count": len(metadata),
        "doc_record_count": len(storage.get_doc_records()),
        "legal_index_articles": len(legal_idx.articles),
        "legal_index_schedules": len(legal_idx.schedules),
        "legal_index_lists": len(legal_idx.lists),
        "legal_index_entries": len(legal_idx.entries),
    }


def _unique(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _chunk_articles(chunks: list[dict]) -> list[str]:
    return _unique([
        chunk.get("article_id") or chunk.get("article") or ""
        for chunk in chunks
    ])


def _chunk_schedules(chunks: list[dict]) -> list[str]:
    return _unique([chunk.get("schedule_id") or "" for chunk in chunks])


def _chunk_lists(chunks: list[dict]) -> list[str]:
    return _unique([chunk.get("list_id") or "" for chunk in chunks])


def _chunk_entries(chunks: list[dict]) -> list[str]:
    return _unique([chunk.get("entry_id") or "" for chunk in chunks])


def _chunk_types(chunks: list[dict]) -> list[str]:
    return [chunk.get("chunk_type") or "" for chunk in chunks]


def _combined_text(chunks: list[dict]) -> str:
    return "\n".join((chunk.get("text") or "") for chunk in chunks)


def _add_issue(issues: list[dict], phase: str, tag: str, message: str):
    issues.append({"phase": phase, "tag": tag, "message": message})


def _tag_for_article_issue(category: str) -> str:
    if category == "repaired_boundary":
        return "indexing"
    if category == "amendment_history":
        return "amendment_evidence_handling"
    return "retrieval"


def _check_subset(
    expected: list[str],
    actual: list[str],
    phase: str,
    tag: str,
    label: str,
    issues: list[dict],
):
    missing = [value for value in expected if value not in actual]
    if missing:
        _add_issue(issues, phase, tag, f"missing {label}: {missing}")


def _evaluate_item(item: dict, default_top_k: int) -> dict:
    query = item["query"]
    top_k = item.get("top_k", default_top_k)
    category = item["category"]
    issues: list[dict] = []

    try:
        chunks = retriever.retrieve(query, top_k=top_k)
    except Exception as exc:
        _add_issue(issues, "retrieval", "runner_error", f"retrieve() raised {type(exc).__name__}: {exc}")
        return {
            "id": item["id"],
            "category": category,
            "query": query,
            "top_k": top_k,
            "issues": issues,
            "retrieval_pass": False,
            "mode_pass": False,
            "answer_contract_pass": False,
            "overall_pass": False,
        }

    articles = _chunk_articles(chunks)
    schedules = _chunk_schedules(chunks)
    lists = _chunk_lists(chunks)
    entries = _chunk_entries(chunks)
    chunk_types = _chunk_types(chunks)
    combined_text = _combined_text(chunks)
    synthetic_only = bool(chunks) and all(chunk.get("synthetic") for chunk in chunks)

    prompt_messages = retriever.build_rag_prompt(query, chunks)
    prompt_text = "\n".join(message.get("content", "") for message in prompt_messages)

    retrieval_criteria = item.get("retrieval_pass_criteria", {})
    article_tag = _tag_for_article_issue(category)

    _check_subset(
        retrieval_criteria.get("required_article_ids", item.get("expected_article_ids", [])),
        articles,
        "retrieval",
        article_tag,
        "required articles",
        issues,
    )
    _check_subset(
        retrieval_criteria.get("required_schedule_ids", item.get("expected_schedule_ids", [])),
        schedules,
        "retrieval",
        "structural_routing",
        "required schedules",
        issues,
    )
    _check_subset(
        retrieval_criteria.get("required_list_ids", item.get("expected_list_ids", [])),
        lists,
        "retrieval",
        "structural_routing",
        "required lists",
        issues,
    )
    _check_subset(
        retrieval_criteria.get("required_entry_ids", item.get("expected_entry_ids", [])),
        entries,
        "retrieval",
        "structural_routing",
        "required entries",
        issues,
    )

    exact_articles = retrieval_criteria.get("exact_article_set")
    if exact_articles is not None and articles != exact_articles:
        _add_issue(
            issues,
            "retrieval",
            article_tag,
            f"article set mismatch: expected {exact_articles}, got {articles}",
        )

    article_prefix = retrieval_criteria.get("article_prefix")
    if article_prefix is not None and articles[: len(article_prefix)] != article_prefix:
        _add_issue(
            issues,
            "retrieval",
            article_tag,
            f"article prefix mismatch: expected {article_prefix}, got {articles}",
        )

    for forbidden_article in retrieval_criteria.get("forbidden_article_ids", []):
        if forbidden_article in articles:
            _add_issue(
                issues,
                "retrieval",
                article_tag,
                f"forbidden neighboring article retrieved: {forbidden_article}",
            )

    allowed_chunk_types_only = retrieval_criteria.get("allowed_chunk_types_only")
    if allowed_chunk_types_only is not None:
        disallowed = [
            chunk_type for chunk_type in chunk_types
            if chunk_type and chunk_type not in allowed_chunk_types_only
        ]
        if disallowed:
            _add_issue(
                issues,
                "retrieval",
                "amendment_evidence_handling" if category == "amendment_history" else article_tag,
                f"unexpected chunk types present: {sorted(set(disallowed))}",
            )

    forbidden_chunk_types = retrieval_criteria.get("forbidden_chunk_types", [])
    seen_forbidden_types = sorted({chunk_type for chunk_type in chunk_types if chunk_type in forbidden_chunk_types})
    if seen_forbidden_types:
        _add_issue(
            issues,
            "retrieval",
            "amendment_evidence_handling" if category == "amendment_history" else article_tag,
            f"forbidden chunk types present: {seen_forbidden_types}",
        )

    combined_text_lower = combined_text.lower()
    for snippet in retrieval_criteria.get("required_text_substrings", []):
        if snippet.lower() not in combined_text_lower:
            _add_issue(
                issues,
                "retrieval",
                "structural_routing" if category == "structural_routing" else article_tag,
                f"missing expected text snippet: {snippet}",
            )

    for snippet in retrieval_criteria.get("forbidden_text_substrings", []):
        if snippet.lower() in combined_text_lower:
            _add_issue(
                issues,
                "retrieval",
                article_tag,
                f"forbidden text snippet present: {snippet}",
            )

    expected_mode = item.get("expected_mode", {})
    amendment_expected = expected_mode.get("amendment_expected")
    constitution_only_expected = expected_mode.get("constitution_only_expected")

    actual_amendment_mode = retriever._query_wants_amendments(query)
    actual_constitution_only_mode = retriever._query_wants_constitution_only(query)

    if amendment_expected is not None and actual_amendment_mode != amendment_expected:
        _add_issue(
            issues,
            "mode",
            "amendment_mode",
            f"amendment mode mismatch: expected {amendment_expected}, got {actual_amendment_mode}",
        )

    if constitution_only_expected is not None and actual_constitution_only_mode != constitution_only_expected:
        _add_issue(
            issues,
            "mode",
            "constitution_only_enforcement",
            f"constitution-only mode mismatch: expected {constitution_only_expected}, got {actual_constitution_only_mode}",
        )

    answer_criteria = item.get("answer_pass_criteria", {})
    synthetic_only_expected = answer_criteria.get("synthetic_only_expected")
    if synthetic_only_expected is not None and synthetic_only != synthetic_only_expected:
        _add_issue(
            issues,
            "answer",
            "synthetic_fallback",
            f"synthetic-only mismatch: expected {synthetic_only_expected}, got {synthetic_only}",
        )

    for snippet in answer_criteria.get("prompt_must_contain", []):
        if snippet not in prompt_text:
            tag = "constitution_only_enforcement"
            if "synthetic" in snippet.lower():
                tag = "synthetic_fallback"
            elif "constitution-only" not in snippet.lower() and "retrieved constitutional context" not in snippet.lower():
                tag = "answer_grounding"
            _add_issue(
                issues,
                "answer",
                tag,
                f"prompt missing required snippet: {snippet}",
            )

    for snippet in answer_criteria.get("prompt_must_not_contain", []):
        if snippet in prompt_text:
            _add_issue(
                issues,
                "answer",
                "answer_grounding",
                f"prompt contains forbidden snippet: {snippet}",
            )

    retrieval_pass = not any(issue["phase"] == "retrieval" for issue in issues)
    mode_pass = not any(issue["phase"] == "mode" for issue in issues)
    answer_contract_pass = not any(issue["phase"] == "answer" for issue in issues)

    return {
        "id": item["id"],
        "category": category,
        "query": query,
        "top_k": top_k,
        "notes": item.get("notes", ""),
        "retrieved_chunk_ids": [chunk.get("chunk_id") for chunk in chunks],
        "retrieved_articles": articles,
        "retrieved_schedules": schedules,
        "retrieved_lists": lists,
        "retrieved_entries": entries,
        "chunk_types": chunk_types,
        "synthetic_only": synthetic_only,
        "amendment_mode": actual_amendment_mode,
        "constitution_only_mode": actual_constitution_only_mode,
        "retrieval_pass": retrieval_pass,
        "mode_pass": mode_pass,
        "answer_contract_pass": answer_contract_pass,
        "overall_pass": retrieval_pass and mode_pass and answer_contract_pass,
        "issues": issues,
    }


def _summarize(results: list[dict]) -> dict:
    category_totals = defaultdict(lambda: {"passed": 0, "total": 0})
    tag_totals = defaultdict(int)
    for result in results:
        stats = category_totals[result["category"]]
        stats["total"] += 1
        if result["overall_pass"]:
            stats["passed"] += 1
        for issue in result["issues"]:
            tag_totals[issue["tag"]] += 1

    return {
        "items_total": len(results),
        "items_passed": sum(1 for result in results if result["overall_pass"]),
        "items_failed": sum(1 for result in results if not result["overall_pass"]),
        "categories": dict(sorted(category_totals.items())),
        "failure_tags": dict(sorted(tag_totals.items())),
        "answer_evaluation_mode": "deterministic_prompt_contract",
    }


def _print_report(dataset_path: Path, summary: dict, results: list[dict]):
    print(f"DATASET: {dataset_path}")
    print(f"ANSWER_EVAL_MODE: {summary['answer_evaluation_mode']}")
    print(
        "SUMMARY: "
        f"{summary['items_passed']}/{summary['items_total']} passed, "
        f"{summary['items_failed']} failed"
    )
    print("CATEGORY SUMMARY:")
    for category, stats in summary["categories"].items():
        print(f"  {category}: {stats['passed']}/{stats['total']} passed")
    if summary["failure_tags"]:
        print("FAILURE TAGS:")
        for tag, count in summary["failure_tags"].items():
            print(f"  {tag}: {count}")

    failing = [result for result in results if not result["overall_pass"]]
    if failing:
        print("FAILURES:")
        for result in failing:
            print(f"  [{result['id']}] category={result['category']}")
            print(f"    query={result['query']}")
            print(f"    retrieved_articles={result['retrieved_articles']}")
            print(
                "    retrieved_structures="
                f"schedules={result['retrieved_schedules']} "
                f"lists={result['retrieved_lists']} "
                f"entries={result['retrieved_entries']}"
            )
            print(f"    chunk_types={result['chunk_types']}")
            for issue in result["issues"]:
                print(f"    - {issue['phase']}::{issue['tag']} -> {issue['message']}")


def main():
    args = _parse_args()
    store_info = _inspect_active_store()
    if store_info["metadata_count"] == 0 or store_info["legal_index_articles"] == 0:
        print(f"ACTIVE_STORE: {store_info}")
        print("STORE_ERROR: active persisted store is empty or not visible; rerun after restoring/regenerating the active store.")
        raise SystemExit(2)

    items = _load_items(args.dataset)
    results = [_evaluate_item(item, args.top_k) for item in items]
    summary = _summarize(results)
    report = {
        "dataset": str(args.dataset),
        "active_store": store_info,
        "summary": summary,
        "results": results,
    }

    print(f"ACTIVE_STORE: {store_info}")
    _print_report(args.dataset, summary, results)

    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE_JSON: {args.output}")


if __name__ == "__main__":
    main()
