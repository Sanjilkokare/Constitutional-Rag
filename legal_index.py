"""
Legal Locator Index â€” structure-aware parsing for Indian Constitution and
other legal PDFs.

Builds a side-car index (data/faiss_index/legal_index.json) mapping
legal identifiers (Article numbers, Schedule/List/Entry, Parts) to
chunk IDs + page numbers, enabling direct lookup before vector search.

Also provides legal-aware chunking that respects Article boundaries.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import config

log = logging.getLogger(__name__)

LEGAL_INDEX_FILE = config.FAISS_INDEX_DIR / "legal_index.json"

# â”€â”€ Regex patterns for Indian Constitution structure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Pattern A: "Article 21." / "Art. 21" heading at start of line
_ARTICLE_HEADING_EXPLICIT = re.compile(
    r"^(?:\*{0,2})"                     # optional markdown bold
    r"(?:Article|Art\.?)\s+"            # "Article" or "Art." or "Art"
    r"(\d+[A-Z]?)"                      # number like 21, 21A, 14
    r"(?:\.\s*|\s+)"                    # dot or space
    r"(.*)",                            # heading text (rest of line)
    re.IGNORECASE | re.MULTILINE,
)

# Pattern B: "21. Protection of life..." / "21A. Right to education..."
# Numbered heading WITHOUT "Article" prefix â€” must look like a heading:
#   - starts at beginning of line
#   - number + dot + space
#   - followed by a capitalised word (title-case heading) or em-dash
# Guards against false positives like numbered list items or footnotes.
_ARTICLE_HEADING_NUMBERED = re.compile(
    r"^(?:\*{0,2})"                         # optional bold
    r"(?:\d+\[)?(\d+[A-Z]?)\.\s+"           # optional footnote marker + "21. "
    r"(?:\d+\[)?"
    r"([A-Z][A-Za-z]"                        # heading starts with capital letter
    r"[^\n]{2,}?"                            # at least a few chars of heading
    r"(?:[.â€”â€“\-]|\s*$))",                   # ends with dot, dash, emdash, or EOL
    re.MULTILINE,
)
_ARTICLE_HEADING_GLUED_FOOTNOTE = re.compile(
    r"^(?:\*{0,2})"
    r"\d(\d{2,3}[A-Z]?)\.\s+"
    r"(\[[^\n]{2,}?(?:[.â€”â€“\-]|\s*$))",
    re.MULTILINE,
)
_ARTICLE_EDITORIAL_HEADING = re.compile(
    r"^(?:"
    r"(?:Subs|Ins|Omitted|Added|Rep)\.?\b|"
    r"(?:Subs|Ins|Omitted|Added|Rep)\.?\s+by\b|"
    r"The\s+Constitution\b|"
    r"Art\.?\s+\d+[A-Z]?\s+re(?:-?numbered)?\b|"
    r"Cl\.?\s*\(\d+\)\s+(?:re-?numbered|renumbered)\b|"
    r"Clause\s+\(\d+\)\s+(?:re-?numbered|renumbered)\b"
    r")",
    re.IGNORECASE,
)

# Broader: finds "Article 21" references anywhere in text (not for segmentation)
_ARTICLE_REF = re.compile(
    r"(?:Article|Art\.?)\s+(\d+[A-Z]?)",
    re.IGNORECASE,
)
_ARTICLE_ID_TOKEN = re.compile(r"\b(\d{1,3}[A-Z]?)\b", re.IGNORECASE)
_ARTICLE_QUERY_RANGE = re.compile(
    r"\b(?:Articles?|Arts?\.?)\s+(\d{1,3}[A-Z]?)\s*(?:-|â€“|â€”|\bto\b)\s*(\d{1,3}[A-Z]?)",
    re.IGNORECASE,
)
_ARTICLE_QUERY_SERIES = re.compile(
    r"\b(?:Articles?|Arts?\.?)\s+((?:\d{1,3}[A-Z]?\s*(?:,|/|&|\band\b)\s*)+\d{1,3}[A-Z]?)",
    re.IGNORECASE,
)
_ARTICLE_COMPARE_SERIES = re.compile(
    r"\b(?:compare|between|vs\.?|versus)\b[^\d]{0,20}"
    r"((?:\d{1,3}[A-Z]?\s*(?:,|/|&|\band\b)\s*)+\d{1,3}[A-Z]?)",
    re.IGNORECASE,
)
_ARTICLE_BARE_SERIES = re.compile(
    r"^\s*((?:\d{1,3}[A-Z]?\s*(?:,|/|&|\band\b)\s*)+\d{1,3}[A-Z]?)\s*$",
    re.IGNORECASE,
)

# Part headings: "PART III" / "PART IV-A"
_PART_HEADING = re.compile(
    r"^(?:\*{0,2})PART\s+([IVXLC]+(?:-[A-Z])?)\b\s*(.*)",
    re.IGNORECASE | re.MULTILINE,
)

_ORDINAL_TO_ROMAN = {
    "FIRST": "I",
    "SECOND": "II",
    "THIRD": "III",
    "FOURTH": "IV",
    "FIFTH": "V",
    "SIXTH": "VI",
    "SEVENTH": "VII",
    "EIGHTH": "VIII",
    "NINTH": "IX",
    "TENTH": "X",
    "ELEVENTH": "XI",
    "TWELFTH": "XII",
}
_LIST_NAME_TO_ROMAN = {
    "UNION LIST": "I",
    "STATE LIST": "II",
    "CONCURRENT LIST": "III",
}
_ROMAN_NUMERAL = re.compile(r"^[IVXLC]+$", re.IGNORECASE)
_ORDINAL_SUFFIX = re.compile(r"^(\d+)(?:ST|ND|RD|TH)$", re.IGNORECASE)

# Schedule/List/Entry headings and references
_SCHEDULE_HEADING = re.compile(
    r"^(?:[\[(\s]*)?(?:\d+\[)?(?P<label>(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|"
    r"EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH|[IVXLC]+|\d{1,2}(?:ST|ND|RD|TH)?))\s+SCHEDULE\b",
    re.IGNORECASE | re.MULTILINE,
)
_SCHEDULE_CONTINUATION_HEADER = re.compile(
    r"^\((?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|"
    r"ELEVENTH|TWELFTH|[IVXLC]+)\s+SCHEDULE\)\s*$",
    re.IGNORECASE,
)
_SCHEDULE_QUERY_NAMED = re.compile(
    r"\b(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH|"
    r"[IVXLC]+|\d{1,2}(?:ST|ND|RD|TH)?)\s+SCHEDULE\b",
    re.IGNORECASE,
)
_SCHEDULE_QUERY_PREFIX = re.compile(
    r"\b(?:Schedule|Sch\.?)\s+([IVXLC]+|\d{1,2}(?:ST|ND|RD|TH)?)\b",
    re.IGNORECASE,
)
_LIST_HEADING = re.compile(
    r"^(?:[\[(\s]*)?LIST\s+(?P<id>[I1]{1,3}|2|3)\b"
    r"(?:\s*[^A-Za-z0-9\n]{0,8}\s*(?P<name>UNION\s+LIST|STATE\s+LIST|CONCURRENT\s+LIST))?",
    re.IGNORECASE | re.MULTILINE,
)
_LIST_ALIAS_HEADING = re.compile(
    r"^(?:[\[(\s]*)?(?P<name>UNION\s+LIST|STATE\s+LIST|CONCURRENT\s+LIST)\b"
    r"(?:\s*[^A-Za-z0-9\n]{0,8}\s*LIST\s+(?P<id>[I1]{1,3}|2|3))?",
    re.IGNORECASE | re.MULTILINE,
)
_LIST_QUERY_NUMERIC = re.compile(
    r"\bLIST\s+(I|II|III|1|2|3)\b",
    re.IGNORECASE,
)
_LIST_QUERY_ALIAS = re.compile(
    r"\b(UNION\s+LIST|STATE\s+LIST|CONCURRENT\s+LIST)\b",
    re.IGNORECASE,
)
_ENTRY_REF = re.compile(
    r"\bEntry\s+(\d+[A-Z]?)\b",
    re.IGNORECASE,
)
_ENTRY_HEADING = re.compile(
    r"^(?:\^?\[)?(?P<entry>\d+[A-Z]?)\.(?:\]|\s)+(?P<title>.+)",
    re.IGNORECASE,
)
_ENTRY_EDITORIAL_TITLE = re.compile(
    r"^(?:Entries?\b|Subs?\.|Ins?\.|Omitted\b|Added\b|Rep\.|"
    r"The\s+(?:word|words|letter|letters|figure|figures)\b|"
    r"Now\b|"
    r"The\s+Constitution\b|Amendment\b)",
    re.IGNORECASE,
)
_CONTENTS_PAGE_HEADING = re.compile(r"^CONTENTS$", re.IGNORECASE)
_STRUCTURAL_EDITORIAL_LINE = re.compile(
    r"^(?:\d+\.\s+)?(?:"
    r"Entries?\b|"
    r"Entry\s+\d+[A-Z]?\b|"
    r"The\s+(?:word|words|letter|letters|figure|figures)\b|"
    r"Subs?\.|Ins?\.|Omitted\b|Added\b|Rep\.|"
    r"Now\b|"
    r"Amendment\b|"
    r"w\.e\.f\."
    r")",
    re.IGNORECASE,
)

_SCHEDULE_OR_LIST_HEADING = re.compile(
    r"^(?:[\[(\s]*\*{0,2})?(?:"
    r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)\s+SCHEDULE|"
    r"SCHEDULE|"
    r"LIST\s+[IVXLC]+|"
    r"UNION\s+LIST|STATE\s+LIST|CONCURRENT\s+LIST"
    r")\b",
    re.IGNORECASE | re.MULTILINE,
)

_ARTICLE_BODY_HINT = re.compile(
    r"^(?:\(\d+\)|\([a-z]\)|No\b|The\b|There\b|Nothing\b|Notwithstanding\b|"
    r"Subject\b|In\s+this\b|All\b|Any\b|Every\b|Parliament\b)",
    re.IGNORECASE,
)

_NON_ARTICLE_CONTEXT_HINTS = (
    re.compile(r"\bschedule\b", re.IGNORECASE),
    re.compile(r"\bparagraph\b", re.IGNORECASE),
    re.compile(r"\bsub-paragraph\b", re.IGNORECASE),
    re.compile(r"\bunion\s+list\b", re.IGNORECASE),
    re.compile(r"\bstate\s+list\b", re.IGNORECASE),
    re.compile(r"\bconcurrent\s+list\b", re.IGNORECASE),
    re.compile(r"\blist\s+[IVXLC]+\b", re.IGNORECASE),
    re.compile(r"\bentry\b", re.IGNORECASE),
)

_INLINE_BODY_SEPARATOR = re.compile(
    r"^(?P<heading>.+?)(?:\.\s*[â€”â€“-]|[â€”â€“-])\s*"
    r"(?P<body>(?:\(\d+\)|\([a-z]\)|No\b|The\b|There\b|Nothing\b|Notwithstanding\b|"
    r"Subject\b|In\s+this\b|All\b|Any\b|Every\b|Provided\b).*)$",
    re.IGNORECASE,
)

_INLINE_BODY_FROM_LINE = re.compile(
    r"^(?P<heading>.+?)\.(?:\s|[^\x00-\x7F]|[^\w\s])*"
    r"(?P<body>(?:\(\d+\)|\([a-z]\)|No\b|The\b|There\b|Nothing\b|Notwithstanding\b|"
    r"Subject\b|In\s+this\b|All\b|Any\b|Every\b|Provided\b).*)$",
    re.IGNORECASE,
)
_WRAPPED_HEADING_BODY_LINE = re.compile(
    r"^(?P<heading_tail>[a-z][^\n]{2,}?)\s*(?:â€”|â€“|—|–|-)\s*"
    r"(?P<body>(?:\(\d+\)|\([a-z]\)|No\b|The\b|There\b|Nothing\b|Notwithstanding\b|"
    r"Subject\b|In\s+this\b|All\b|Any\b|Every\b|Provided\b).*)$"
)

# Amendment footnotes
_AMENDMENT_REF = re.compile(
    r"(?:Constitution\s+\()?((?:First|Second|Third|\w+)\s+Amendment)\s+Act",
    re.IGNORECASE,
)


def _int_to_roman(value: int) -> str:
    numerals = [
        (10, "X"),
        (9, "IX"),
        (8, "VIII"),
        (7, "VII"),
        (6, "VI"),
        (5, "V"),
        (4, "IV"),
        (3, "III"),
        (2, "II"),
        (1, "I"),
    ]
    result = []
    remaining = value
    for num, roman in numerals:
        while remaining >= num:
            result.append(roman)
            remaining -= num
    return "".join(result)


def _canonical_schedule_id(value: str) -> str:
    token = re.sub(r"[\[\](){}.]", "", (value or "").strip()).upper()
    if not token:
        return ""
    if token in _ORDINAL_TO_ROMAN:
        return _ORDINAL_TO_ROMAN[token]
    suffix_m = _ORDINAL_SUFFIX.match(token)
    if suffix_m:
        token = suffix_m.group(1)
    if token.isdigit():
        try:
            number = int(token)
        except ValueError:
            return ""
        if 1 <= number <= 12:
            return _int_to_roman(number)
        return token
    if _ROMAN_NUMERAL.match(token):
        return token
    return ""


def _canonical_list_id(value: str) -> str:
    token = re.sub(r"[\[\](){}.]", "", (value or "").strip()).upper()
    if not token:
        return ""
    if token in _LIST_NAME_TO_ROMAN:
        return _LIST_NAME_TO_ROMAN[token]
    if set(token) <= {"I", "1"}:
        token = token.replace("1", "I")
    if token.startswith("LIST "):
        token = token.split()[-1]
    if token == "1":
        return "I"
    if token == "2":
        return "II"
    if token == "3":
        return "III"
    if token in {"I", "II", "III"}:
        return token
    return ""


def _canonical_entry_id(value: str) -> str:
    return re.sub(r"[\[\](){}.]", "", (value or "").strip()).upper()


def _looks_like_structural_heading_remainder(remainder: str) -> bool:
    stripped = (remainder or "").strip()
    if not stripped:
        return True
    if stripped in {".", ":", ";", "-", "—", "–"}:
        return True
    if stripped[0] in "[(:-—–":
        return True
    return False


def _looks_like_contents_page(lines: list[str]) -> bool:
    top_lines = [line.strip() for line in lines[:30] if line.strip()]
    if not top_lines:
        return False
    if not any(_CONTENTS_PAGE_HEADING.match(line) for line in top_lines[:3]):
        return False
    structural_hits = sum(
        1 for line in top_lines
        if _match_schedule_heading(line) or _match_list_heading(line)
    )
    return structural_hits >= 3


def _looks_like_structural_editorial_line(line: str) -> bool:
    return bool(_STRUCTURAL_EDITORIAL_LINE.match((line or "").strip()))


def _split_wrapped_heading_body(line: str) -> tuple[str, str]:
    match = _WRAPPED_HEADING_BODY_LINE.match((line or "").strip())
    if not match:
        return "", ""
    return (
        match.group("heading_tail").strip().rstrip(".*â€”â€“—–-"),
        match.group("body").strip(),
    )

# â”€â”€ Query identifier extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def extract_identifiers(query: str) -> dict:
    """
    Parse a user query for legal identifiers.

    Returns dict with keys:
      articles: list[str]  e.g. ["21", "21A", "20"]
      parts: list[str]     e.g. ["III", "IV"]
      schedules: list[str] e.g. ["VII"]
      lists: list[str]     e.g. ["I", "II"]
      entries: list[str]   e.g. ["1", "2"]
      amendments: list[str]

    Also handles ranges like "Articles 20-22" -> ["20", "21", "22"]
    """
    result = {
        "articles": [],
        "parts": [],
        "schedules": [],
        "lists": [],
        "entries": [],
        "amendments": [],
    }

    def _add_article(article_num: str):
        article_num = article_num.upper()
        num_m = re.match(r"(\d+)", article_num)
        if not num_m or int(num_m.group(1)) > 400:
            return
        if article_num not in result["articles"]:
            result["articles"].append(article_num)

    def _add_article_series(series: str):
        for token in _ARTICLE_ID_TOKEN.findall(series):
            _add_article(token)

    def _add_schedule(value: str):
        schedule_id = _canonical_schedule_id(value)
        if schedule_id and schedule_id not in result["schedules"]:
            result["schedules"].append(schedule_id)

    def _add_list(value: str):
        list_id = _canonical_list_id(value)
        if list_id and list_id not in result["lists"]:
            result["lists"].append(list_id)

    def _add_entry(value: str):
        entry_id = _canonical_entry_id(value)
        if entry_id and entry_id not in result["entries"]:
            result["entries"].append(entry_id)

    def _add_article_range(start: str, end: str):
        start = start.upper()
        end = end.upper()
        start_m = re.match(r"(\d+)([A-Z]?)", start)
        end_m = re.match(r"(\d+)([A-Z]?)", end)
        if not start_m or not end_m:
            _add_article(start)
            _add_article(end)
            return

        start_num, start_suffix = int(start_m.group(1)), start_m.group(2)
        end_num, end_suffix = int(end_m.group(1)), end_m.group(2)
        if start_suffix or end_suffix:
            _add_article(start)
            if end != start:
                _add_article(end)
            return

        step = 1 if start_num <= end_num else -1
        for n in range(start_num, end_num + step, step):
            _add_article(str(n))

    # Article ranges: "Articles 20-22", "Arts. 14 to 18"
    for m in _ARTICLE_QUERY_RANGE.finditer(query):
        _add_article_range(m.group(1), m.group(2))

    # Article series: "Articles 20, 21 and 22", "Article 20, 21, 22"
    for m in _ARTICLE_QUERY_SERIES.finditer(query):
        _add_article_series(m.group(1))

    # Bare article series: "20/21/22", "20, 21 & 22", "Compare 20, 21 & 22"
    bare_series = _ARTICLE_BARE_SERIES.search(query)
    if bare_series:
        _add_article_series(bare_series.group(1))
    else:
        for m in _ARTICLE_COMPARE_SERIES.finditer(query):
            _add_article_series(m.group(1))

    # Individual articles
    for m in _ARTICLE_REF.finditer(query):
        _add_article(m.group(1))

    # Parts
    for m in _PART_HEADING.finditer(query):
        result["parts"].append(m.group(1).upper())
    # Also catch "Part III" in query without heading format
    for m in re.finditer(r"Part\s+([IVXLC]+(?:-[A-Z])?)", query, re.IGNORECASE):
        p = m.group(1).upper()
        if p not in result["parts"]:
            result["parts"].append(p)

    # Schedules
    for m in _SCHEDULE_QUERY_NAMED.finditer(query):
        _add_schedule(m.group(1))
    for m in _SCHEDULE_QUERY_PREFIX.finditer(query):
        _add_schedule(m.group(1))

    # Lists
    list_matches = []
    for m in _LIST_QUERY_ALIAS.finditer(query):
        list_matches.append((m.start(), m.group(1)))
    for m in _LIST_QUERY_NUMERIC.finditer(query):
        list_matches.append((m.start(), m.group(1)))
    for _, value in sorted(list_matches, key=lambda item: item[0]):
        _add_list(value)

    # Entries
    for m in _ENTRY_REF.finditer(query):
        _add_entry(m.group(1))

    # Amendments
    for m in _AMENDMENT_REF.finditer(query):
        result["amendments"].append(m.group(1))

    has_ids = any(v for v in result.values())
    if has_ids:
        log.info("Extracted identifiers from query: %s", {k: v for k, v in result.items() if v})

    return result


# â”€â”€ Legal-aware text segmentation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _find_article_headings(full_text: str) -> list[dict]:
    """
    Find all article heading positions using both explicit ("Article 21.")
    and numbered ("21. Protection...") patterns.

    Returns sorted, deduplicated list of
      {"start": int, "article": str, "heading": str}
    """
    seen_starts: set[int] = set()
    positions: list[dict] = []

    def _add(m_start: int, article: str, heading: str, source: str):
        if m_start in seen_starts:
            return
        inline_match = _INLINE_BODY_SEPARATOR.match(heading.strip())
        inline_body = ""
        if inline_match:
            heading = inline_match.group("heading")
            inline_body = inline_match.group("body")
        seen_starts.add(m_start)
        positions.append({
            "start": m_start,
            "article": article.upper(),
            "heading": heading.strip().rstrip(".*â€”â€“-"),
            "source": source,
            "inline_body": inline_body.strip(),
        })

    def _looks_like_editorial_heading(heading: str) -> bool:
        return bool(_ARTICLE_EDITORIAL_HEADING.match((heading or "").strip().lstrip("[")))

    def _inline_body_from_segment_line(line: str, article: str) -> str:
        prefix = re.compile(
            rf"^(?:\*{{0,2}})?(?:(?:Article|Art\.?)\s+)?{re.escape(article)}(?:\.\s*|\s+)",
            re.IGNORECASE,
        )
        stripped = prefix.sub("", line.strip(), count=1)
        inline_match = _INLINE_BODY_FROM_LINE.match(stripped)
        return inline_match.group("body").strip() if inline_match else ""

    # Pattern A: explicit "Article 21." headings
    for m in _ARTICLE_HEADING_EXPLICIT.finditer(full_text):
        _add(m.start(), m.group(1), m.group(2), "explicit")

    # Pattern B: numbered "21. Protection of..." headings
    for m in _ARTICLE_HEADING_NUMBERED.finditer(full_text):
        art_num = m.group(1)
        heading = m.group(2)
        source = (
            "footnoted"
            if re.match(r"^(?:\*{0,2})?(?:\d+\[|\d+[A-Z]?\.\s+\d+\[)", m.group(0))
            else "numbered"
        )
        # Guard: skip if this looks like a footnote reference
        # (e.g. "1. Subs. by the Constitution (Forty-second Amendment)...")
        if _looks_like_editorial_heading(heading):
            continue
        # Guard: skip very high numbers that are likely list/entry items
        try:
            num = int(re.match(r"\d+", art_num).group())
            if num > 400:
                continue
        except (AttributeError, ValueError):
            pass
        _add(m.start(), art_num, heading, source)

    for m in _ARTICLE_HEADING_GLUED_FOOTNOTE.finditer(full_text):
        art_num = m.group(1)
        heading = m.group(2)
        if _looks_like_editorial_heading(heading):
            continue
        try:
            num = int(re.match(r"\d+", art_num).group())
            if num > 400:
                continue
        except (AttributeError, ValueError):
            pass
        _add(m.start(), art_num, heading, "footnoted")

    positions.sort(key=lambda p: p["start"])
    if not positions:
        return positions

    schedule_or_list_page = bool(
        _SCHEDULE_OR_LIST_HEADING.search("\n".join(full_text.splitlines()[:12]))
    )

    filtered: list[dict] = []
    for i, pos in enumerate(positions):
        if schedule_or_list_page and pos["source"] == "numbered":
            continue

        end = positions[i + 1]["start"] if i + 1 < len(positions) else len(full_text)
        segment_text = full_text[pos["start"] : end].strip()
        body_lines = [line.strip() for line in segment_text.splitlines() if line.strip()]
        heading_text = pos["heading"]
        inline_body = pos.get("inline_body") or _inline_body_from_segment_line(
            body_lines[0] if body_lines else "",
            pos["article"],
        )
        remaining_body_lines = body_lines[1:] if len(body_lines) > 1 else []
        if not inline_body and len(body_lines) > 1:
            wrapped_tail, wrapped_body = _split_wrapped_heading_body(body_lines[1])
            if wrapped_tail and wrapped_body:
                heading_text = f"{heading_text} {wrapped_tail}".strip()
                inline_body = wrapped_body
                remaining_body_lines = body_lines[2:]
        body_parts = []
        if inline_body:
            body_parts.append(inline_body)
        if remaining_body_lines:
            body_parts.append("\n".join(remaining_body_lines).strip())
        body = "\n".join(part for part in body_parts if part).strip()
        context_before = full_text[max(0, pos["start"] - 800) : pos["start"]]
        non_article_context_hits = sum(
            1 for pattern in _NON_ARTICLE_CONTEXT_HINTS if pattern.search(context_before)
        )

        if not body:
            continue

        if pos["source"] == "footnoted":
            filtered.append({
                "start": pos["start"],
                "article": pos["article"],
                "heading": heading_text,
            })
            continue

        if pos["source"] == "numbered":
            first_body_line = inline_body or (remaining_body_lines[0] if remaining_body_lines else "")
            if non_article_context_hits >= 2:
                continue
            if re.search(r"\(\d+\)|\([a-z]\)", body):
                filtered.append({
                    "start": pos["start"],
                    "article": pos["article"],
                    "heading": heading_text,
                })
                continue
            if len(body) >= 40 and _ARTICLE_BODY_HINT.match(first_body_line):
                filtered.append({
                    "start": pos["start"],
                    "article": pos["article"],
                    "heading": heading_text,
                })
                continue
            continue

        first_body_line = inline_body or (remaining_body_lines[0] if remaining_body_lines else "")
        if len(body) >= 30 or re.search(r"\(\d+\)|\([a-z]\)", body) or _ARTICLE_BODY_HINT.match(first_body_line):
            filtered.append({
                "start": pos["start"],
                "article": pos["article"],
                "heading": heading_text,
            })

    return filtered


def segment_articles(full_text: str, page_num: int) -> list[dict]:
    """
    Split a page's text into article-level segments.

    Returns list of dicts:
      {
        "article": "21",        # or None if not an article block
        "heading": "Protection of life...",
        "text": "full text of article...",
        "page": page_num,
        "type": "article" | "part_heading" | "text_block"
      }
    """
    segments = []

    # Find all article headings in the text (both formats)
    article_positions = _find_article_headings(full_text)

    if not article_positions:
        # No article headings found â€” return the whole page as one block
        if full_text.strip():
            segments.append({
                "article": None,
                "heading": "",
                "text": full_text.strip(),
                "page": page_num,
                "type": "text_block",
            })
        return segments

    # Text before first article
    if article_positions[0]["start"] > 0:
        pre_text = full_text[: article_positions[0]["start"]].strip()
        if pre_text:
            # Check if it's a Part heading
            part_m = _PART_HEADING.search(pre_text)
            segments.append({
                "article": None,
                "heading": part_m.group(2).strip() if part_m else "",
                "text": pre_text,
                "page": page_num,
                "type": "part_heading" if part_m else "text_block",
            })

    # Each article: text from heading to next heading
    for i, pos in enumerate(article_positions):
        start = pos["start"]
        end = article_positions[i + 1]["start"] if i + 1 < len(article_positions) else len(full_text)
        text = full_text[start:end].strip()

        segments.append({
            "article": pos["article"],
            "heading": pos["heading"],
            "text": text,
            "page": page_num,
            "type": "article",
        })

    return segments


def _prepare_structure_text(text: str) -> str:
    """Add soft boundaries around embedded list headings or entry markers."""
    prepared = re.sub(
        r"(?<!\n)(?=\s*LIST\s+(?:[I1]{1,3}|2|3)\s*[^A-Za-z0-9\n]{0,8}\s*(?:UNION\s+LIST|STATE\s+LIST|CONCURRENT\s+LIST)\b)",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    prepared = re.sub(
        r"(?<![\n\d])(?=\s*(?:(?:\d+\[)?\^?\[?\d+[A-Z]?\.\s+[A-Z]))",
        "\n",
        prepared,
    )
    return prepared


def _match_schedule_heading(line: str) -> str:
    stripped = line.strip()
    match = _SCHEDULE_HEADING.match(stripped)
    if not match:
        return ""
    if not _looks_like_structural_heading_remainder(stripped[match.end() :]):
        return ""
    return _canonical_schedule_id(match.group("label"))


def _match_list_heading(line: str) -> str:
    stripped = line.strip()
    match = _LIST_HEADING.match(stripped)
    if match:
        name = match.group("name") or ""
        if not name and len(stripped) > 12:
            return ""
        if not _looks_like_structural_heading_remainder(stripped[match.end() :]):
            return ""
        return _canonical_list_id(match.group("id") or name or "")
    match = _LIST_ALIAS_HEADING.match(stripped)
    if match:
        if not _looks_like_structural_heading_remainder(stripped[match.end() :]):
            return ""
        return _canonical_list_id(match.group("name") or match.group("id") or "")
    return ""


def _match_entry_heading(line: str) -> tuple[str, str]:
    stripped = line.strip()
    match = _ENTRY_HEADING.match(stripped)
    if not match:
        match = re.match(
            r"^(?:\d+\[)?(?:\^?\[)?(?P<entry>\d+[A-Z]?)\.(?:\]|\s)+(?P<title>.+)",
            stripped,
            re.IGNORECASE,
        )
    if not match:
        return "", ""
    title = match.group("title").strip()
    if _ENTRY_EDITORIAL_TITLE.match(title):
        return "", ""
    return _canonical_entry_id(match.group("entry")), title


def _looks_like_schedule_or_list_page(
    full_text: str,
    carry_schedule: str = "",
    carry_list: str = "",
) -> bool:
    prepared = _prepare_structure_text(full_text)
    lines = [line.strip() for line in prepared.splitlines() if line.strip()]
    top_lines = lines[:16]
    if _looks_like_contents_page(lines):
        return False
    if any(_match_schedule_heading(line) or _match_list_heading(line) for line in top_lines):
        return True

    if carry_list:
        if any(_match_entry_heading(line)[0] for line in top_lines[:12]):
            return True

    if carry_schedule and any("schedule" in line.lower() for line in top_lines[:8]):
        return True

    return False


def segment_schedule_entries(
    full_text: str,
    page_num: int,
    carry_schedule: str = "",
    carry_list: str = "",
    carry_entry: str = "",
) -> list[dict]:
    """
    Split schedule/list pages into structural segments.

    Returns list of dicts with schedule_id/list_id/entry_id set where applicable.
    Only activates on true schedule/list pages or continuations.
    """
    if not _looks_like_schedule_or_list_page(full_text, carry_schedule, carry_list):
        return []

    prepared = _prepare_structure_text(full_text)
    lines = prepared.splitlines()
    segments: list[dict] = []
    current_schedule = _canonical_schedule_id(carry_schedule)
    current_list = _canonical_list_id(carry_list)
    if current_list and not current_schedule:
        current_schedule = "VII"
    current_entry = _canonical_entry_id(carry_entry)
    current_type = "entry" if current_entry else "text_block"
    current_heading = ""
    buffer: list[str] = []
    saw_structure = bool(current_schedule or current_list)

    def _flush():
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if not text:
            return
        segments.append({
            "article": None,
            "heading": current_heading,
            "text": text,
            "page": page_num,
            "type": current_type,
            "schedule_id": current_schedule,
            "list_id": current_list,
            "entry_id": current_entry if current_type == "entry" else "",
        })

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if buffer:
                buffer.append("")
            continue

        if current_list and current_entry and not buffer and _SCHEDULE_CONTINUATION_HEADER.match(line):
            continue

        if (current_schedule or current_list) and _looks_like_structural_editorial_line(line):
            _flush()
            buffer = [line]
            current_entry = ""
            current_type = "text_block"
            current_heading = ""
            saw_structure = True
            continue

        schedule_id = _match_schedule_heading(line)
        if schedule_id:
            _flush()
            buffer = [line]
            current_schedule = schedule_id
            current_list = ""
            current_entry = ""
            current_type = "schedule_heading"
            current_heading = line
            saw_structure = True
            continue

        list_id = _match_list_heading(line)
        if list_id:
            _flush()
            buffer = [line]
            current_list = list_id
            if not current_schedule:
                current_schedule = "VII"
            current_entry = ""
            current_type = "list_heading"
            current_heading = line
            saw_structure = True
            continue

        entry_id, title = _match_entry_heading(line)
        if current_list and entry_id:
            _flush()
            buffer = [line]
            current_entry = entry_id
            current_type = "entry"
            current_heading = title
            saw_structure = True
            continue

        if not buffer:
            if current_list and current_entry:
                current_type = "entry"
            else:
                current_type = "text_block"
            current_heading = ""
            buffer = [line]
        else:
            buffer.append(line)

    _flush()

    if not saw_structure:
        return []

    return segments


# â”€â”€ Legal Locator Index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class LegalIndex:
    """
    Maps legal identifiers to chunk IDs for direct lookup.
    Stored as JSON at data/faiss_index/legal_index.json.
    """

    def __init__(self):
        # article_num -> list of chunk_ids
        self.articles: dict[str, list[int]] = {}
        # part_num -> list of chunk_ids
        self.parts: dict[str, list[int]] = {}
        # schedule_num -> list of chunk_ids
        self.schedules: dict[str, list[int]] = {}
        # list_num -> list of chunk_ids
        self.lists: dict[str, list[int]] = {}
        # composite list_id:entry_id -> list of chunk_ids
        self.entries: dict[str, list[int]] = {}
        # generic keyword tags -> chunk_ids
        self.tags: dict[str, list[int]] = {}

    def add_article(self, article_num: str, chunk_id: int):
        self.articles.setdefault(article_num.upper(), []).append(chunk_id)

    def add_part(self, part_num: str, chunk_id: int):
        self.parts.setdefault(part_num.upper(), []).append(chunk_id)

    def add_schedule(self, schedule_num: str, chunk_id: int):
        self.schedules.setdefault(schedule_num.upper(), []).append(chunk_id)

    def add_list(self, list_num: str, chunk_id: int):
        self.lists.setdefault(list_num.upper(), []).append(chunk_id)

    def add_entry(self, list_num: str, entry_num: str, chunk_id: int):
        if not list_num or not entry_num:
            return
        key = f"{list_num.upper()}:{entry_num.upper()}"
        self.entries.setdefault(key, []).append(chunk_id)

    def add_tag(self, tag: str, chunk_id: int):
        self.tags.setdefault(tag.lower(), []).append(chunk_id)

    def register_chunk(
        self,
        chunk_id: int,
        text: str,
        article: Optional[str] = None,
        schedule: str = "",
        list_id: str = "",
        entry_id: str = "",
    ):
        """Register a chunk, extracting all legal identifiers from its text."""
        if article:
            self.add_article(article, chunk_id)
        schedule_id = _canonical_schedule_id(schedule)
        if schedule_id:
            self.add_schedule(schedule_id, chunk_id)
        list_num = _canonical_list_id(list_id)
        if list_num:
            self.add_list(list_num, chunk_id)
            if not schedule_id:
                self.add_schedule("VII", chunk_id)
                schedule_id = "VII"
        entry_num = _canonical_entry_id(entry_id)
        if list_num and entry_num:
            self.add_entry(list_num, entry_num, chunk_id)

        # Also find article references within the text body
        for m in _ARTICLE_REF.finditer(text):
            ref = m.group(1).upper()
            if ref not in self.articles or chunk_id not in self.articles.get(ref, []):
                self.add_tag(f"ref_article_{ref}", chunk_id)

        # Part references
        for m in re.finditer(r"Part\s+([IVXLC]+(?:-[A-Z])?)", text, re.IGNORECASE):
            self.add_part(m.group(1).upper(), chunk_id)

    def lookup(self, identifiers: dict) -> list[int]:
        """
        Given extracted identifiers, return matching chunk IDs (ordered, deduplicated).
        Articles get priority â€” direct article chunks come first.
        """
        chunk_ids: list[int] = []
        seen: set[int] = set()

        def _add(cids: list[int]):
            for cid in cids:
                if cid not in seen:
                    seen.add(cid)
                    chunk_ids.append(cid)

        # Direct article matches (highest priority)
        for art in identifiers.get("articles", []):
            _add(self.articles.get(art.upper(), []))
            # Also include chunks that reference this article
            _add(self.tags.get(f"ref_article_{art.upper()}", []))

        # Parts
        for part in identifiers.get("parts", []):
            _add(self.parts.get(part.upper(), []))

        # Direct list-entry combinations
        if identifiers.get("lists") and identifiers.get("entries"):
            for lst in identifiers.get("lists", []):
                for ent in identifiers.get("entries", []):
                    _add(self.entries.get(f"{lst.upper()}:{ent.upper()}", []))

        # Schedules, Lists, Entries
        for sch in identifiers.get("schedules", []):
            _add(self.schedules.get(sch.upper(), []))
        for lst in identifiers.get("lists", []):
            _add(self.lists.get(lst.upper(), []))
        if not identifiers.get("lists"):
            for ent in identifiers.get("entries", []):
                ent_upper = ent.upper()
                matching = [
                    cid
                    for key, cids in self.entries.items()
                    if key.endswith(f":{ent_upper}")
                    for cid in cids
                ]
                _add(matching)

        if chunk_ids:
            log.info("Legal index lookup returned %d chunk IDs", len(chunk_ids))

        return chunk_ids

    def save(self):
        """Persist to disk."""
        data = {
            "articles": self.articles,
            "parts": self.parts,
            "schedules": self.schedules,
            "lists": self.lists,
            "entries": self.entries,
            "tags": self.tags,
        }
        LEGAL_INDEX_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Saved legal index: %d articles, %d parts, %d schedules, %d lists, %d entries, %d tags",
                 len(self.articles), len(self.parts), len(self.schedules),
                 len(self.lists), len(self.entries), len(self.tags))

    @classmethod
    def load(cls) -> "LegalIndex":
        """Load from disk, or return empty index."""
        idx = cls()
        if LEGAL_INDEX_FILE.exists():
            try:
                data = json.loads(LEGAL_INDEX_FILE.read_text(encoding="utf-8-sig"))
                idx.articles = data.get("articles", {})
                idx.parts = data.get("parts", {})
                idx.schedules = data.get("schedules", {})
                idx.lists = data.get("lists", {})
                idx.entries = data.get("entries", {})
                idx.tags = data.get("tags", {})
                log.info("Loaded legal index: %d articles, %d parts, %d schedules, %d lists, %d entries, %d tags",
                         len(idx.articles), len(idx.parts), len(idx.schedules),
                         len(idx.lists), len(idx.entries), len(idx.tags))
            except Exception as e:
                log.warning("Failed to load legal index: %s", e)
        return idx


# â”€â”€ Singleton for runtime use â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_legal_index: LegalIndex | None = None


def get_legal_index() -> LegalIndex:
    global _legal_index
    if _legal_index is None:
        _legal_index = LegalIndex.load()
    return _legal_index


def reset_legal_index():
    """Reset to empty (called before re-ingestion)."""
    global _legal_index
    _legal_index = LegalIndex()
    return _legal_index

