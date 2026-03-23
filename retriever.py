"""
Hybrid retriever with identifier-aware routing.

Query flow:
  1. Parse query for legal identifiers (Article X, Schedule Y, etc.)
  2. If found → direct lookup via Legal Locator Index → return exact chunks
  3. If not found or lookup empty → hybrid FAISS + BM25 search
  4. Build RAG prompt optimized for sarvam-m
"""

import logging
import math
import re
from collections import Counter

import numpy as np

import config
import embeddings
import legal_index as li
import storage

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight BM25
# ═══════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"\w{2,}", text.lower())


class _BM25:
    """Minimal BM25 over the stored chunk metadata (rebuilt on each query)."""

    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = documents
        self.doc_tokens = [_tokenize(d["text"]) for d in documents]
        self.avgdl = (
            sum(len(t) for t in self.doc_tokens) / len(self.doc_tokens)
            if self.doc_tokens
            else 1
        )
        self.n = len(documents)
        # document frequency
        self.df: Counter = Counter()
        for tokens in self.doc_tokens:
            for t in set(tokens):
                self.df[t] += 1

    def score(self, query: str) -> np.ndarray:
        """Return BM25 scores for all documents given *query*."""
        q_tokens = _tokenize(query)
        scores = np.zeros(self.n, dtype=np.float32)

        for qt in q_tokens:
            df = self.df.get(qt, 0)
            if df == 0:
                continue
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)
            for i, doc_toks in enumerate(self.doc_tokens):
                tf = doc_toks.count(qt)
                dl = len(doc_toks)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * num / den

        return scores


# ═══════════════════════════════════════════════════════════════════════════
# Identifier-aware direct lookup
# ═══════════════════════════════════════════════════════════════════════════

_DIRECT_PREFERRED_TYPES = frozenset({"article", "clause", "amendment"})
_ARTICLE_BODY_TYPES = frozenset({"article", "clause"})
_ARTICLE_NOTE_TYPES = frozenset({"amendment", "footnote"})

_LEADING_ARTICLE_PREFIX = re.compile(
    r"^\[Article\s+\d+[A-Z]?\](?:\s+\(continued\))?\s*",
    re.IGNORECASE,
)
_BRACKETED_NEW_ARTICLE = re.compile(
    r"^\[Article\s+\d+[A-Z]?\](?!\s+\(continued\))",
    re.IGNORECASE,
)
_NEW_ARTICLE_START = re.compile(
    r"^(?:(?:Article|Art\.?)\s+\d+[A-Z]?\.?|\d+[A-Z]?\.)",
    re.IGNORECASE,
)
_PART_HEADER_LINE = re.compile(r"^\(Part\b.+\)$", re.IGNORECASE)
_CONTINUATION_OPENERS = re.compile(
    r"^(?:\(\d+[A-Z]?\)|\([a-z]\)|Provided\b|Explanation\b|Nothing\b|Any\b|No\b|The\b|Such\b|In\b|For\b|Subject\b|Notwithstanding\b)",
    re.IGNORECASE,
)
_EDITORIAL_TAIL_LINE = re.compile(
    r"^\s*(?:\*+\s*)?(?:"
    r"\d+\.\s+(?:Subs\.|Ins\.|Omitted|Added|Rep\.|The\s+Constitution\s+\()|"
    r"(?:Subs\.|Ins\.|Omitted|Added|Rep\.)\s+by|"
    r"Sub-clause\b|Cl\.\s*\(\d+\)\s+shall|Art\.\s*\d+\s+re[- ]|"
    r"(?:\{\}\s*)?\^?\d+\[\s*$|w\.e\.f\.)",
    re.IGNORECASE,
)
_EDITORIAL_BODY_ONLY = re.compile(
    r"^\(\d+[A-Z]?\)\s+shall\s+stand\s+(?:substituted|omitted|re-lettered)\b",
    re.IGNORECASE,
)
_ARREST_OR_DETENTION_QUERY = re.compile(
    r"\b(?:arrest|arrested|detention|detained|custody|preventive detention|grounds of arrest)\b",
    re.IGNORECASE,
)
_REMEDY_OR_ENFORCEMENT_QUERY = re.compile(
    r"\b(?:remedy|remedies|enforce|enforcement|writ|writs|habeas corpus|constitutional remedy|move any court|approach (?:the )?(?:court|supreme court)|petition)\b",
    re.IGNORECASE,
)
_AMENDMENT_QUERY_LABELS = (
    ("footnote", re.compile(r"\bfootnotes?\b", re.IGNORECASE)),
    ("substitution", re.compile(r"\b(?:substitution|substituted)\b", re.IGNORECASE)),
    ("omission", re.compile(r"\b(?:omission|omitted)\b", re.IGNORECASE)),
    ("insertion", re.compile(r"\b(?:insertion|inserted)\b", re.IGNORECASE)),
)
_AMENDMENT_EVIDENCE_START = re.compile(
    r"^\s*(?:"
    r"\d+\.\s+(?:Subs\.|Ins\.|Omitted|Added|Rep\.|The\s+Constitution\s+\()|"
    r"(?:Subs\.|Ins\.|Omitted|Added|Rep\.)\s+by|"
    r"Cl\.\s*\(\d+\)\s+shall|"
    r"Sub-clause\s*\([a-z]\)\s+shall|"
    r"\(\d+[A-Z]?\)\s+shall\s+stand\s+(?:substituted|omitted|re-lettered)\b|"
    r"\([a-z]\)\s+shall\s+stand\s+(?:substituted|omitted|re-lettered)\b|"
    r"(?:\{\}\s*)?\^?\d+\[\s*$|w\.e\.f\.)",
    re.IGNORECASE,
)
_CONSTITUTION_ONLY_QUERY_PATTERNS = (
    re.compile(r"\bbased\s+only\s+on\s+(?:the\s+)?constitution(?:al\s+text)?\b", re.IGNORECASE),
    re.compile(r"\banswer\s+only\s+from\s+(?:the\s+)?constitution(?:al\s+text)?\b", re.IGNORECASE),
    re.compile(r"\bstrictly\s+from\s+(?:the\s+)?constitution(?:al\s+text)?\b", re.IGNORECASE),
    re.compile(r"\bconstitution\s+only\b", re.IGNORECASE),
    re.compile(r"\bonly\s+from\s+(?:the\s+)?constitutional\s+text\b", re.IGNORECASE),
    re.compile(r"\bonly\s+use\s+(?:the\s+)?constitution\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+case\s+law\b", re.IGNORECASE),
    re.compile(r"\bno\s+case\s+law\b", re.IGNORECASE),
    re.compile(r"\b(?:do\s+not\s+use|without|no)\s+external\s+law\b", re.IGNORECASE),
    re.compile(r"\b(?:do\s+not\s+use|without|no)\s+external\s+legal\s+knowledge\b", re.IGNORECASE),
)
_CONSTITUTION_ONLY_EXTERNAL_LINE_PATTERNS = (
    re.compile(
        r"\b(?:ipc|indian\s+penal\s+code|penal\s+code|crpc|cr\.?\s*p\.?\s*c\.?|"
        r"code\s+of\s+criminal\s+procedure|evidence\s+act|indian\s+evidence\s+act|"
        r"cpc|code\s+of\s+civil\s+procedure)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:case\s+law|precedent|precedents|judicial\s+precedent|judicial\s+precedents|"
        r"supreme\s+court\s+held|high\s+court\s+held|court\s+held|judgment|judgement|ruling|ratio\s+decidendi)\b",
        re.IGNORECASE,
    ),
)
_FUNDAMENTAL_RIGHTS_QUERY = re.compile(r"\bfundamental\s+rights?\b", re.IGNORECASE)
_CONTENTS_CHUNK = re.compile(r"^\s*contents\b", re.IGNORECASE)


def _direct_article_rank(meta: dict) -> tuple[int, int, int, int, int]:
    """
    Prefer substantive article text over short TOC-like matches while keeping
    requested article order stable for direct legal-index routing.
    """
    text = (meta.get("text") or "").strip()
    page = meta.get("page_start") or meta.get("page") or 10**9
    chunk_type = meta.get("chunk_type", "generic")
    toc_penalty = 1 if page <= 10 and len(text) < 180 else 0
    substantive_penalty = 0 if len(text) >= 180 or "(1)" in text else 1
    type_penalty = 0 if chunk_type in _DIRECT_PREFERRED_TYPES else 1
    return (toc_penalty, substantive_penalty, type_penalty, page, -len(text))


def _query_wants_amendments(query: str) -> bool:
    return bool(_AMENDMENT_QUERY_KEYWORDS.search(query))


def _query_wants_constitution_only(query: str) -> bool:
    return any(pattern.search(query or "") for pattern in _CONSTITUTION_ONLY_QUERY_PATTERNS)


def _amendment_query_label(query: str) -> str:
    for label, pattern in _AMENDMENT_QUERY_LABELS:
        if pattern.search(query):
            return label
    return "amendment/editorial note"


def _append_unique(values: list[str], value: str):
    if value and value not in values:
        values.append(value)


def _explicit_article_mentions(query: str) -> list[str]:
    articles: list[str] = []
    for match in re.finditer(r"\bArticles?\s+([^\n?.;:]+)", query, re.IGNORECASE):
        for token in re.findall(r"\d+[A-Z]?", match.group(1), re.IGNORECASE):
            _append_unique(articles, token.upper())
    return articles


def _expand_identifiers(query: str, identifiers: dict) -> dict:
    """
    Apply small, deterministic related-article rules on top of explicit IDs.
    """
    expanded = {k: list(v) for k, v in identifiers.items()}
    articles = expanded["articles"]
    explicit_mentions = _explicit_article_mentions(query)
    if explicit_mentions:
        articles[:] = explicit_mentions
        for article in identifiers.get("articles", []):
            _append_unique(articles, article.upper())
    explicit_articles = set(a.upper() for a in (explicit_mentions or identifiers.get("articles", [])))

    if "352" in explicit_articles:
        if "19" in explicit_articles:
            _append_unique(articles, "358")
        if "20" in explicit_articles or "21" in explicit_articles:
            _append_unique(articles, "359")

    if _ARREST_OR_DETENTION_QUERY.search(query):
        _append_unique(articles, "22")
        if _REMEDY_OR_ENFORCEMENT_QUERY.search(query):
            _append_unique(articles, "32")

    if articles != identifiers.get("articles", []):
        log.info(
            "Expanded article identifiers: explicit=%s expanded=%s",
            identifiers.get("articles", []),
            articles,
        )

    return expanded


def _strip_article_prefix(text: str) -> str:
    stripped = _LEADING_ARTICLE_PREFIX.sub("", text.strip(), count=1)
    lines = stripped.splitlines()
    if lines and _PART_HEADER_LINE.match(lines[0].strip()):
        stripped = "\n".join(lines[1:]).strip()
    return stripped


def _line_looks_like_new_article(line: str) -> bool:
    if not line:
        return False
    if re.match(
        r"^\d+[A-Z]?\.\s+(?:Subs\.|Ins\.|Omitted|Added|Rep\.)",
        line,
        re.IGNORECASE,
    ):
        return False
    return bool(_NEW_ARTICLE_START.match(line))


def _article_heading_match(article: str, text: str):
    article = re.escape(article.upper())
    patterns = [
        rf"(?im)^\[Article\s+{article}\](?![A-Z0-9])",
        rf"(?im)^(?:Article|Art\.?)\s+{article}(?![A-Z0-9])\.?",
        r"(?im)^(?:\^\{?\d+\}?\[?)?" + article + r"\.(?![A-Z0-9])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match
    return None


def _trim_to_article_start(text: str, article: str) -> str:
    match = _article_heading_match(article, text)
    if match:
        return text[match.start():].strip()
    return text.strip()


def _retag_article_prefix(text: str, article: str) -> str:
    return _LEADING_ARTICLE_PREFIX.sub(f"[Article {article}] ", text, count=1).strip()


def _looks_like_new_article(text: str) -> bool:
    raw = text.strip()
    if _BRACKETED_NEW_ARTICLE.match(raw):
        return True
    stripped = _strip_article_prefix(raw)
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False
    if _line_looks_like_new_article(lines[0]):
        return True
    if len(lines) > 1 and _line_looks_like_new_article(lines[1]):
        return True
    return False


def _looks_like_article_continuation(text: str) -> bool:
    stripped = _strip_article_prefix(text)
    if not stripped or _looks_like_new_article(text):
        return False
    return bool(_CONTINUATION_OPENERS.match(stripped))


def _strip_editorial_tail(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    for i, line in enumerate(lines):
        candidate = _strip_article_prefix(line).strip()
        if _EDITORIAL_TAIL_LINE.search(candidate) or _EDITORIAL_BODY_ONLY.search(candidate):
            return "\n".join(kept).strip()
        kept.append(line)
    return "\n".join(kept).strip()


def _extract_amendment_evidence(meta: dict) -> str:
    if meta.get("chunk_type") in _ARTICLE_NOTE_TYPES:
        text = (meta.get("text") or "").strip()
        return text if re.search(r"[A-Za-z]{3,}", text) else ""

    evidence: list[str] = []
    started = False
    for line in (meta.get("text") or "").splitlines():
        candidate = _strip_article_prefix(line).strip()
        if not candidate:
            if started:
                evidence.append("")
            continue
        if (
            _AMENDMENT_EVIDENCE_START.search(candidate)
            or _EDITORIAL_BODY_ONLY.search(candidate)
        ):
            started = True
        if started:
            evidence.append(candidate)

    text = "\n".join(evidence).strip()
    return text if re.search(r"[A-Za-z]{3,}", text) else ""


def _amendment_not_found_chunk(article: str, query: str, all_meta: list[dict]) -> dict:
    template = next(
        (meta for meta in all_meta if meta.get("article_id") == article),
        all_meta[0] if all_meta else {},
    )
    label = _amendment_query_label(query)
    if label == "footnote":
        message = f"No footnote for Article {article} was found in the retrieved Constitution context."
    elif label in {"substitution", "omission", "insertion"}:
        message = f"No {label}/amendment note for Article {article} was found in the retrieved Constitution context."
    else:
        message = f"No amendment/editorial note for Article {article} was found in the retrieved Constitution context."
    return {
        "doc_id": template.get("doc_id", ""),
        "filename": template.get("filename", ""),
        "page": None,
        "page_start": None,
        "page_end": None,
        "chunk_id": -1,
        "text": message,
        "image_path": "",
        "chunk_type": "amendment",
        "article_id": article,
        "schedule_id": "",
        "list_id": "",
        "entry_id": "",
        "section_id": "",
        "source_extraction": "",
        "score": 1.0,
        "retrieval_method": "legal_index",
        "synthetic": True,
        "synthetic_reason": "no_exact_amendment_evidence",
        "display_citation": False,
    }


def _article_chunk_allowed(
    article: str,
    meta: dict,
    include_amendments: bool,
    is_primary_chunk: bool = False,
) -> bool:
    chunk_type = meta.get("chunk_type", "generic")
    if chunk_type == "heading":
        return is_primary_chunk and bool(_article_heading_match(article, meta.get("text", "")))
    if chunk_type == "generic":
        cleaned = _strip_editorial_tail(meta.get("text", ""))
        if not cleaned:
            return False
        if is_primary_chunk:
            return bool(_article_heading_match(article, cleaned))
        return _looks_like_article_continuation(cleaned)
    if chunk_type in _ARTICLE_BODY_TYPES:
        cleaned = _strip_editorial_tail(meta.get("text", ""))
        if not cleaned:
            return False
        if chunk_type == "article" and not is_primary_chunk and not _looks_like_article_continuation(cleaned):
            return False
        return True
    if include_amendments and chunk_type in _ARTICLE_NOTE_TYPES:
        return True
    return False


def _recover_adjacent_article_chunks(
    article: str,
    base_ids: list[int],
    all_meta: list[dict],
    include_amendments: bool,
) -> list[int]:
    """
    Recover immediately adjacent continuation chunks when the stored metadata
    dropped or mis-tagged the article id on later pages.
    """
    if not base_ids:
        return []

    valid_ids = [cid for cid in base_ids if 0 <= cid < len(all_meta)]
    if not valid_ids:
        return []

    doc_id = all_meta[valid_ids[0]].get("doc_id")
    last_page = max(
        all_meta[cid].get("page_start") or all_meta[cid].get("page") or 0
        for cid in valid_ids
    )
    recovered: list[int] = []
    started = False

    for idx in range(max(valid_ids) + 1, len(all_meta)):
        meta = all_meta[idx]
        if meta.get("doc_id") != doc_id:
            break
        if meta.get("schedule_id") or meta.get("list_id") or meta.get("entry_id"):
            break

        page = meta.get("page_start") or meta.get("page") or 0
        if last_page and page and page - last_page > 1:
            break

        chunk_type = meta.get("chunk_type", "generic")
        if _looks_like_new_article(meta.get("text", "")):
            break

        if not started:
            if chunk_type in _ARTICLE_NOTE_TYPES:
                if include_amendments:
                    recovered.append(idx)
                last_page = page or last_page
                continue
            if chunk_type in _ARTICLE_BODY_TYPES and _looks_like_article_continuation(meta.get("text", "")):
                recovered.append(idx)
                started = True
                last_page = page or last_page
                continue
            break

        if chunk_type in _ARTICLE_BODY_TYPES:
            recovered.append(idx)
            last_page = page or last_page
            continue
        if chunk_type in _ARTICLE_NOTE_TYPES:
            if include_amendments:
                recovered.append(idx)
            last_page = page or last_page
            continue
        break

    if recovered:
        log.info("Recovered %d adjacent continuation chunks for Article %s", len(recovered), article)
    return recovered


def _fallback_article_seed_chunk_ids(article: str, all_meta: list[dict]) -> list[int]:
    seeds: list[int] = []
    for idx, meta in enumerate(all_meta):
        if meta.get("schedule_id") or meta.get("list_id") or meta.get("entry_id"):
            continue
        if _article_heading_match(article, meta.get("text", "")):
            seeds.append(idx)
    if not seeds:
        return []

    def _seed_rank(cid: int):
        text = all_meta[cid].get("text", "")
        heading_count = len(re.findall(r"(?im)^(?:\^\{?\d+\}?\[?)?\d+[A-Z]?\.", text))
        toc_penalty = 1 if heading_count > 2 else 0
        return (toc_penalty, *_direct_article_rank(all_meta[cid]))

    seeds.sort(key=_seed_rank)
    best_seed = seeds[0]
    log.info("Using fallback article heading seed %d for Article %s (from %d candidates)", best_seed, article, len(seeds))
    return [best_seed]


def _article_bundle_chunk_ids(
    article: str,
    legal_idx: li.LegalIndex,
    all_meta: list[dict],
    include_amendments: bool,
) -> list[int]:
    direct_ids = [
        cid
        for cid in legal_idx.articles.get(article.upper(), [])
        if 0 <= cid < len(all_meta)
    ]
    if not direct_ids:
        direct_ids = _fallback_article_seed_chunk_ids(article, all_meta)
    ordered: list[int] = []
    seen: set[int] = set()
    primary_direct_id = next(
        (
            cid for cid in sorted(direct_ids)
            if all_meta[cid].get("chunk_type", "generic") in (_ARTICLE_BODY_TYPES | {"generic", "heading"})
        ),
        -1,
    )

    def _add(cid: int):
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)

    for cid in sorted(direct_ids):
        if _article_chunk_allowed(article, all_meta[cid], include_amendments, is_primary_chunk=(cid == primary_direct_id)):
            _add(cid)

    for cid in _recover_adjacent_article_chunks(article, direct_ids, all_meta, include_amendments):
        if _article_chunk_allowed(article, all_meta[cid], include_amendments):
            _add(cid)

    return ordered


def _materialize_article_bundle(
    article: str,
    query: str,
    legal_idx: li.LegalIndex,
    all_meta: list[dict],
) -> list[dict]:
    include_amendments = _query_wants_amendments(query)
    results: list[dict] = []
    bundle_ids = _article_bundle_chunk_ids(article, legal_idx, all_meta, include_amendments)
    first_bundle_id = bundle_ids[0] if bundle_ids else -1

    for cid in bundle_ids:
        meta = dict(all_meta[cid])
        original_article_id = meta.get("article_id")
        if cid == first_bundle_id:
            meta["text"] = _trim_to_article_start(meta.get("text", ""), article)
        if original_article_id != article:
            meta["article_id"] = article
            meta["text"] = _retag_article_prefix(meta.get("text", ""), article)
        if include_amendments:
            meta["text"] = _extract_amendment_evidence(meta)
            if not meta["text"]:
                continue
            meta["chunk_type"] = "amendment"
        elif meta.get("chunk_type") in (_ARTICLE_BODY_TYPES | {"generic"}):
            meta["text"] = _strip_editorial_tail(meta.get("text", ""))
            if not meta["text"]:
                continue
        meta["score"] = 1.0
        meta["retrieval_method"] = "legal_index"
        results.append(meta)

    if include_amendments and not results:
        results.append(_amendment_not_found_chunk(article, query, all_meta))

    return results


def _ordered_direct_chunk_ids(
    legal_idx: li.LegalIndex,
    identifiers: dict,
    all_meta: list[dict],
) -> list[int]:
    """Order direct hits by article order first, then append remaining direct matches."""
    ordered: list[int] = []
    seen: set[int] = set()

    def _add(cid: int):
        if 0 <= cid < len(all_meta) and cid not in seen:
            seen.add(cid)
            ordered.append(cid)

    article_groups: list[list[int]] = []
    for art in identifiers.get("articles", []):
        group = [
            cid for cid in legal_idx.articles.get(art.upper(), [])
            if 0 <= cid < len(all_meta)
        ]
        group.sort(key=lambda cid: _direct_article_rank(all_meta[cid]))
        article_groups.append(group)

    for group in article_groups:
        if group:
            _add(group[0])
    for group in article_groups:
        for cid in group[1:]:
            _add(cid)

    structural_groups: list[list[int]] = []
    if identifiers.get("lists") and identifiers.get("entries"):
        for lst in identifiers.get("lists", []):
            for ent in identifiers.get("entries", []):
                group = [
                    cid for cid in legal_idx.entries.get(f"{lst.upper()}:{ent.upper()}", [])
                    if 0 <= cid < len(all_meta)
                ]
                structural_groups.append(group)
    elif identifiers.get("lists"):
        for lst in identifiers.get("lists", []):
            group = [
                cid for cid in legal_idx.lists.get(lst.upper(), [])
                if 0 <= cid < len(all_meta)
            ]
            structural_groups.append(group)
    elif identifiers.get("schedules"):
        for sch in identifiers.get("schedules", []):
            group = [
                cid for cid in legal_idx.schedules.get(sch.upper(), [])
                if 0 <= cid < len(all_meta)
            ]
            structural_groups.append(group)
    elif identifiers.get("entries"):
        for ent in identifiers.get("entries", []):
            ent_upper = ent.upper()
            group = [
                cid
                for key, cids in legal_idx.entries.items()
                if key.endswith(f":{ent_upper}")
                for cid in cids
                if 0 <= cid < len(all_meta)
            ]
            structural_groups.append(group)

    for group in structural_groups:
        if group:
            _add(group[0])
    for group in structural_groups:
        for cid in group[1:]:
            _add(cid)
    for cid in legal_idx.lookup(identifiers):
        _add(cid)

    return ordered


def _identifier_lookup(query: str, top_k: int) -> list[dict]:
    """
    If the query contains legal identifiers, look them up directly
    in the Legal Locator Index. Returns chunk metadata dicts or [].
    """
    identifiers = li.extract_identifiers(query)
    identifiers = _expand_identifiers(query, identifiers)
    has_ids = any(v for v in identifiers.values())

    if not has_ids:
        return []

    legal_idx = li.get_legal_index()
    all_meta = storage.get_all_metadata()
    results: list[dict] = []

    if identifiers.get("articles"):
        seen_keys: set[str] = set()
        for article in identifiers.get("articles", []):
            for meta in _materialize_article_bundle(article.upper(), query, legal_idx, all_meta):
                key = f"{meta['doc_id']}:{meta['chunk_id']}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    results.append(meta)

        structural_ids = {
            "articles": [],
            "parts": identifiers.get("parts", []),
            "schedules": identifiers.get("schedules", []),
            "lists": identifiers.get("lists", []),
            "entries": identifiers.get("entries", []),
            "amendments": identifiers.get("amendments", []),
        }
        if structural_ids["schedules"] or structural_ids["lists"] or structural_ids["entries"]:
            for cid in _ordered_direct_chunk_ids(legal_idx, structural_ids, all_meta):
                if cid < len(all_meta):
                    meta = dict(all_meta[cid])
                    key = f"{meta['doc_id']}:{meta['chunk_id']}"
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    meta["score"] = 1.0
                    meta["retrieval_method"] = "legal_index"
                    results.append(meta)
                    if len(results) >= top_k:
                        break
    else:
        if identifiers.get("schedules") or identifiers.get("lists") or identifiers.get("entries"):
            chunk_ids = _ordered_direct_chunk_ids(legal_idx, identifiers, all_meta)
        else:
            chunk_ids = legal_idx.lookup(identifiers)

        for cid in chunk_ids:
            if cid < len(all_meta):
                meta = dict(all_meta[cid])
                meta["score"] = 1.0  # Direct match = highest confidence
                meta["retrieval_method"] = "legal_index"
                results.append(meta)
            if len(results) >= top_k:
                break

    if not results:
        log.info("Legal index lookup found identifiers but no matching chunks")
        return []

    log.info("Legal index direct lookup returned %d chunks for identifiers: %s",
             len(results), {k: v for k, v in identifiers.items() if v})
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Chunk-type filtering
# ═══════════════════════════════════════════════════════════════════════════

# Query keywords that signal the user *wants* footnote/amendment content
_AMENDMENT_QUERY_KEYWORDS = re.compile(
    r"\b(?:amendments?|amended|substitution|substituted|omission|omitted|"
    r"insertion|inserted|footnotes?|editorial\s+note|history\s+of\s+change|"
    r"constitutional\s+history|subs\.|ins\.)\b",
    re.IGNORECASE,
)

_DOWNWEIGHT_TYPES = frozenset({"footnote", "amendment"})


def _chunk_type_weight(chunk_type: str, query_wants_amendments: bool) -> float:
    """
    Return a multiplier for the chunk's score based on its type.
    Footnotes/amendments are downweighted unless the query explicitly asks.
    """
    if chunk_type in _DOWNWEIGHT_TYPES and not query_wants_amendments:
        return config.FOOTNOTE_DOWNWEIGHT
    return 1.0


def _hybrid_query_text(query: str) -> str:
    """
    Trim control phrasing and add small lexical expansions for concept queries.
    This only affects BM25/hybrid fallback, not direct legal-index routing.
    """
    hybrid_query = query or ""
    for pattern in _CONSTITUTION_ONLY_QUERY_PATTERNS:
        hybrid_query = pattern.sub(" ", hybrid_query)
    hybrid_query = re.sub(r"\s+", " ", hybrid_query).strip()

    if _REMEDY_OR_ENFORCEMENT_QUERY.search(query) and _FUNDAMENTAL_RIGHTS_QUERY.search(query):
        hybrid_query = (
            f"{hybrid_query} constitutional remedies enforcement rights conferred by this part "
            "move supreme court"
        ).strip()

    return hybrid_query or query


def _hybrid_rank_weight(meta: dict) -> float:
    """
    Downweight contents-page and heading-only noise inside hybrid fallback.
    Direct legal-index retrieval is unaffected.
    """
    weight = 1.0
    text = (meta.get("text") or "").strip()
    chunk_type = meta.get("chunk_type", "generic")
    page = meta.get("page_start") or meta.get("page") or 0

    if chunk_type == "heading":
        weight *= 0.7
    if page and page <= 10:
        if _CONTENTS_CHUNK.match(text):
            weight *= 0.35
        elif chunk_type == "generic":
            weight *= 0.6

    return weight


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid retrieval (FAISS + BM25)
# ═══════════════════════════════════════════════════════════════════════════

def _hybrid_search(
    query: str,
    top_k: int,
    vector_weight: float,
    bm25_weight: float,
) -> list[dict]:
    """Hybrid FAISS + BM25 search with chunk_type filtering."""
    all_meta = storage.get_all_metadata()
    if not all_meta:
        return []

    query_wants_amendments = _query_wants_amendments(query)
    hybrid_query = _hybrid_query_text(query)

    # ── Vector search
    q_vec = embeddings.embed_query(query)
    faiss_k = min(top_k * 3, len(all_meta))
    faiss_results = storage.search(q_vec, top_k=faiss_k)
    vec_scores: dict[int, float] = {}
    for meta, score in faiss_results:
        for i, m in enumerate(all_meta):
            if (m["doc_id"] == meta["doc_id"]
                    and m["chunk_id"] == meta["chunk_id"]):
                vec_scores[i] = score
                break

    # ── BM25 search
    bm25 = _BM25(all_meta)
    bm25_scores = bm25.score(hybrid_query)
    bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
    bm25_norm = bm25_scores / bm25_max

    # ── Combine with chunk_type weighting
    combined: list[tuple[int, float]] = []
    candidate_indices = set(vec_scores.keys())
    bm25_top = np.argsort(bm25_scores)[::-1][: faiss_k]
    candidate_indices.update(bm25_top.tolist())

    for idx in candidate_indices:
        vs = vec_scores.get(idx, 0.0)
        bs = float(bm25_norm[idx])
        raw_score = vector_weight * vs + bm25_weight * bs
        # Apply chunk_type penalty
        ctype = all_meta[idx].get("chunk_type", "generic")
        score = raw_score * _chunk_type_weight(ctype, query_wants_amendments)
        score *= _hybrid_rank_weight(all_meta[idx])
        combined.append((idx, score))

    combined.sort(key=lambda x: x[1], reverse=True)

    seen_keys: set[str] = set()
    results: list[dict] = []
    for idx, score in combined:
        if len(results) >= top_k:
            break
        meta = dict(all_meta[idx])
        key = f"{meta['doc_id']}:{meta['chunk_id']}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        meta["score"] = round(score, 4)
        meta["retrieval_method"] = "hybrid"
        results.append(meta)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Public retrieval API
# ═══════════════════════════════════════════════════════════════════════════

def _log_telemetry(query: str, method: str, results: list[dict]):
    """Log retrieval telemetry: method, top-k sources with chunk_type and scores."""
    if not results:
        log.info("RETRIEVAL | method=%s | query=%.60s | results=0", method, query)
        return

    top_entries = []
    for r in results[:5]:
        ctype = r.get("chunk_type", "generic")
        art = r.get("article_id") or r.get("article", "")
        score = r.get("score", "?")
        page = r.get("page_start") or r.get("page", "?")
        top_entries.append(f"p{page}/{ctype}/art{art}/s{score}")

    log.info(
        "RETRIEVAL | method=%s | query=%.60s | top-%d: [%s]",
        method, query, len(results), ", ".join(top_entries),
    )


def retrieve(
    query: str,
    top_k: int = config.TOP_K,
    vector_weight: float = config.VECTOR_WEIGHT,
    bm25_weight: float = config.BM25_WEIGHT,
) -> list[dict]:
    """
    Identifier-aware hybrid retrieval.

    1. If query has legal identifiers → direct lookup (legal_index first)
    2. If no identifiers or lookup empty → hybrid FAISS + BM25
    3. Apply chunk_type filters (downweight footnotes/amendments)
    4. Log telemetry
    """
    all_meta = storage.get_all_metadata()
    if not all_meta:
        return []

    # ── Step 1: Try identifier-based direct lookup
    direct_results = _identifier_lookup(query, top_k)

    if direct_results:
        log.info("Using legal index direct lookup (%d results)", len(direct_results))
        _log_telemetry(query, "legal_index", direct_results)
        return direct_results

    # ── Step 2: Fallback to hybrid search
    results = _hybrid_search(query, top_k, vector_weight, bm25_weight)
    _log_telemetry(query, "hybrid", results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# RAG prompt builder (optimized for sarvam-m)
# ═══════════════════════════════════════════════════════════════════════════

def _chunk_citation_header(c: dict) -> str:
    """Build a structured citation header from chunk metadata."""
    if c.get("display_citation") is False:
        return ""

    parts = []
    filename = re.sub(r"\s+", " ", (c.get("filename") or "")).strip()
    if filename:
        parts.append(filename)
    if c.get("article_id"):
        parts.append(f"article:{c['article_id']}")
    elif c.get("article"):
        parts.append(f"article:{c['article']}")
    if c.get("schedule_id"):
        parts.append(f"schedule:{c['schedule_id']}")
    if c.get("list_id"):
        parts.append(f"list:{c['list_id']}")
    if c.get("entry_id"):
        parts.append(f"entry:{c['entry_id']}")
    page = c.get("page_start")
    if page is None:
        page = c.get("page")
    page_end = c.get("page_end")
    if page_end is None:
        page_end = page
    if page is not None:
        if page == page_end:
            parts.append(f"p:{page}")
        else:
            parts.append(f"p:{page}-{page_end}")
    chunk_type = (c.get("chunk_type") or "").strip()
    if chunk_type and chunk_type not in {"generic", "article"}:
        parts.append(chunk_type)
    return "[" + " | ".join(parts) + "]" if parts else ""


def _chunk_context_header(index: int, chunk: dict) -> str:
    """Build a prompt header without making synthetic notes look like source citations."""
    if chunk.get("synthetic"):
        article = chunk.get("article_id") or chunk.get("article")
        reason = chunk.get("synthetic_reason", "synthetic")
        article_label = f" for Article {article}" if article else ""
        return f"--- Chunk {index} (synthetic retrieval note{article_label}; reason: {reason}) ---"

    citation = _chunk_citation_header(chunk)
    if citation:
        return f"--- Chunk {index} {citation} ---"
    return f"--- Chunk {index} ---"


_HIDDEN_REASONING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|analysis|reasoning)\b[^>]*>.*?</(?:think|thinking|analysis|reasoning)>",
    re.IGNORECASE | re.DOTALL,
)
_HIDDEN_REASONING_TAG_RE = re.compile(
    r"</?(?:think|thinking|analysis|reasoning)\b[^>]*>",
    re.IGNORECASE,
)
_XML_CITATION_RE = re.compile(
    r"<citation\b(?P<attrs>[^>]*)>(?P<body>.*?)</citation>",
    re.IGNORECASE | re.DOTALL,
)
_CITATION_BRACKET_RE = re.compile(r"\[([^\[\]]{1,400})\]")
_CITATION_SIGNAL_RE = re.compile(
    r"(?:\bdoc\b|\.pdf\b|\barticle\b|\bschedule\b|\blist\b|\bentry\b|\bp:|\btype\b)",
    re.IGNORECASE,
)
_UNSUPPORTED_EXPANSION_TERMS = (
    "privacy",
    "dignity",
    "health",
    "pollution-free environment",
    "pollution free environment",
    "maneka gandhi",
)


def _strip_hidden_reasoning(text: str) -> str:
    """Remove leaked hidden-reasoning tags before displaying an answer."""
    text = _HIDDEN_REASONING_BLOCK_RE.sub("", text)
    text = _HIDDEN_REASONING_TAG_RE.sub("", text)
    return text.strip()


def _strip_xml_citation_tags(text: str) -> str:
    """Drop synthetic citation tags and unwrap any real citation wrappers."""

    def _rewrite(match: re.Match) -> str:
        attrs = re.sub(r"\s+", " ", match.group("attrs") or "").strip().lower()
        body = re.sub(r"\s+", " ", match.group("body") or "").strip()
        lowered = f"{attrs} {body}".strip().lower()
        if "synthetic" in lowered:
            return ""
        return body

    return _XML_CITATION_RE.sub(_rewrite, text)


def _synthetic_fallback_answer(chunks: list[dict]) -> str:
    """Return the grounded fallback text directly when all retrieved chunks are synthetic."""
    if not chunks or not all(c.get("synthetic") for c in chunks):
        return ""

    for chunk in chunks:
        text = re.sub(r"\s+", " ", (chunk.get("text") or "")).strip()
        if text:
            return text
    return ""


def _drop_constitution_only_external_lines(text: str, chunks: list[dict], query: str) -> str:
    """Drop unsupported external-law references when Constitution-only mode is explicit."""
    if not _query_wants_constitution_only(query):
        return text

    context_text = "\n".join(c.get("text", "") for c in chunks).lower()
    lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if line == "Not found in provided context.":
            lines.append("Not found in retrieved constitutional context.")
            continue

        lowered = line.lower()
        if any(
            pattern.search(lowered) and not pattern.search(context_text)
            for pattern in _CONSTITUTION_ONLY_EXTERNAL_LINE_PATTERNS
        ):
            continue
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _normalize_citation_brackets(text: str) -> str:
    """Normalize citation brackets into the existing structured schema."""
    allowed_keys = {"doc", "article", "schedule", "list", "entry", "p", "type"}

    def _rewrite(match: re.Match) -> str:
        body = re.sub(r"\s+", " ", match.group(1)).strip()
        if not _CITATION_SIGNAL_RE.search(body):
            return match.group(0)

        fields: dict[str, str] = {}
        for raw_part in body.split("|"):
            part = re.sub(r"\s+", " ", raw_part).strip()
            if not part:
                continue

            lower = part.lower()
            if ":" not in part:
                if lower.startswith("doc "):
                    key = "doc"
                    value = part[4:].strip()
                elif ".pdf" in lower:
                    key = "doc"
                    value = part
                elif lower.startswith("type "):
                    key = "type"
                    value = part[5:].strip()
                else:
                    continue
            else:
                key, value = part.split(":", 1)
                key = key.strip().lower()
                value = re.sub(r"\s+", " ", value).strip()

            if key not in allowed_keys or not value:
                continue
            fields[key] = value

        normalized_parts: list[str] = []
        if fields.get("doc"):
            normalized_parts.append(fields["doc"])
        for key in ("article", "schedule", "list", "entry", "p"):
            if fields.get(key):
                normalized_parts.append(f"{key}:{fields[key]}")
        if fields.get("type") and fields["type"].lower() not in {"generic", "article"}:
            normalized_parts.append(fields["type"])

        if not normalized_parts:
            return ""
        return "[" + " | ".join(normalized_parts) + "]"

    return _CITATION_BRACKET_RE.sub(_rewrite, text)


def _drop_unsupported_expansion_lines(text: str, chunks: list[dict]) -> str:
    """
    Remove high-risk legal expansions when those topics are absent from retrieved text.

    This is intentionally narrow: it targets the unsupported Article 21-style
    expansions observed in runtime answers without changing retrieval behavior.
    """
    context_text = "\n".join(c.get("text", "") for c in chunks).lower()
    lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if line == "Not found in provided context.":
            lines.append(line)
            continue

        lowered = line.lower()
        if any(term in lowered and term not in context_text for term in _UNSUPPORTED_EXPANSION_TERMS):
            continue
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_rag_answer(answer: str, chunks: list[dict], query: str = "") -> str:
    """Sanitize and lightly ground the model answer before UI display."""
    synthetic_only = _synthetic_fallback_answer(chunks)
    if synthetic_only:
        return synthetic_only

    cleaned = _strip_hidden_reasoning(answer)
    cleaned = _strip_xml_citation_tags(cleaned)
    cleaned = _normalize_citation_brackets(cleaned)
    cleaned = _drop_unsupported_expansion_lines(cleaned, chunks)
    cleaned = _drop_constitution_only_external_lines(cleaned, chunks, query)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if cleaned:
        return cleaned
    if _query_wants_constitution_only(query):
        return "Not found in retrieved constitutional context."
    return "Not found in provided context."


def build_rag_prompt(query: str, chunks: list[dict]) -> list[dict[str, str]]:
    """
    Build the message list for Sarvam chat completion.
    Optimized for sarvam-m: short system prompt, structured context.
    """
    context_parts: list[str] = []
    has_synthetic = any(c.get("synthetic") for c in chunks)
    constitution_only = _query_wants_constitution_only(query)
    for i, c in enumerate(chunks, 1):
        # Truncate very long chunks to avoid token overflow
        text = c["text"][:1500]
        context_parts.append(f"{_chunk_context_header(i, c)}\n{text}")

    context_block = "\n\n".join(context_parts)
    synthetic_rule = ""
    if has_synthetic:
        synthetic_rule = (
            "- Synthetic retrieval notes are fallback context only. "
            "Do not cite them as extracted constitutional text.\n"
        )
        if all(c.get("synthetic") for c in chunks):
            synthetic_rule += (
                "- If the only context is a synthetic retrieval note, repeat that fallback "
                "plainly and do not add citations.\n"
            )
    constitution_only_rule = ""
    constitution_only_user_note = ""
    if constitution_only:
        constitution_only_rule = (
            "- Constitution-only mode is active.\n"
            "- Answer only from the retrieved constitutional context.\n"
            "- Do NOT use IPC, CrPC, Evidence Act, other statutes, case law, or external legal knowledge unless that exact material appears in the context.\n"
            "- Prefer direct constitutional text or close paraphrase before brief explanation.\n"
            "- Remain grounded and concise.\n"
            "- If the retrieved constitutional context is insufficient, say: \"Not found in retrieved constitutional context.\"\n"
        )
        constitution_only_user_note = (
            "CONSTITUTION-ONLY MODE: Answer only from the retrieved constitutional text. "
            "If the context is insufficient, say: Not found in retrieved constitutional context.\n\n"
        )

    system_msg = (
        "You answer questions ONLY from the provided CONTEXT.\n"
        "Rules:\n"
        "- If the answer is not in the context, say: \"Not found in provided context.\"\n"
        "- Use only facts explicitly stated in the context. Do not use outside legal knowledge.\n"
        "- Do NOT add case law, judicial expansion, doctrine, or constitutional interpretation unless that exact material appears in the context.\n"
        "- Do NOT mention privacy, dignity, health, pollution-free environment, Maneka Gandhi, or similar expansions unless those exact topics appear in the context.\n"
        "- Every material factual claim must be supported by one or more citations.\n"
        "- Copy citations exactly from the chunk header tags; do not invent, rewrite, abbreviate, or leave empty fields.\n"
        "- Do not output <think>, <thinking>, or any hidden-reasoning tags.\n"
        "- Be precise. Quote exact text when relevant.\n"
        "- Do NOT make up information.\n"
        "- If a chunk is tagged type:footnote or type:amendment, note it as editorial annotation.\n"
        f"{constitution_only_rule}"
        f"{synthetic_rule}"
    )

    user_msg = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        f"{constitution_only_user_note}"
        "Answer only from the context. If a requested detail is missing, say so.\n"
        "End each factual paragraph or bullet with exact copied citations:"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
