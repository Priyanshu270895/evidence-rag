import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SOURCE_CITATION_PATTERN = re.compile(r"\[SOURCE (?P<index>\d+)\]")
WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_']+")
PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore (all )?(previous|prior|above) instructions\b", re.IGNORECASE),
    re.compile(
        r"\b(disregard|override|bypass) (the )?(instructions|system|developer)\b", re.IGNORECASE
    ),
    re.compile(r"\b(system|developer) prompt\b", re.IGNORECASE),
    re.compile(r"\byou are now\b", re.IGNORECASE),
    re.compile(r"\bforget (the )?(rules|instructions)\b", re.IGNORECASE),
    re.compile(r"\breveal (the )?(prompt|secrets|system message)\b", re.IGNORECASE),
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class CitationCheck:
    cited_source_indexes: list[int]
    invalid_source_indexes: list[int]
    has_valid_citation: bool


def detect_prompt_injection(text: str) -> list[str]:
    return [pattern.pattern for pattern in PROMPT_INJECTION_PATTERNS if pattern.search(text)]


def validate_source_citations(answer: str, evidence_count: int) -> CitationCheck:
    cited_indexes = [
        int(match.group("index")) for match in SOURCE_CITATION_PATTERN.finditer(answer)
    ]
    invalid_indexes = sorted(
        {index for index in cited_indexes if index < 1 or index > evidence_count}
    )
    valid_indexes = sorted({index for index in cited_indexes if 1 <= index <= evidence_count})
    return CitationCheck(
        cited_source_indexes=valid_indexes,
        invalid_source_indexes=invalid_indexes,
        has_valid_citation=bool(valid_indexes) and not invalid_indexes,
    )


def support_score(answer: str, evidence: list[Mapping[str, Any]]) -> float:
    answer_tokens = content_tokens(SOURCE_CITATION_PATTERN.sub("", answer))
    if not answer_tokens:
        return 0.0
    evidence_tokens = content_tokens(" ".join(str(row.get("text", "")) for row in evidence))
    if not evidence_tokens:
        return 0.0
    return len(answer_tokens & evidence_tokens) / len(answer_tokens)


def evidence_prompt_injection_risk(evidence: list[Mapping[str, Any]]) -> bool:
    return any(detect_prompt_injection(str(row.get("text", ""))) for row in evidence)


def content_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in WORD_PATTERN.findall(text)
        if len(token) > 2 and token.casefold() not in STOPWORDS
    }
