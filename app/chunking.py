import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    index: int


def chunk_text(text: str, page: int, chunk_size: int = 900, overlap: int = 120) -> list[Chunk]:
    """Split on paragraph/sentence boundaries while retaining a small overlap."""
    clean = re.sub(r"[ \t]+", " ", text).strip()
    if not clean:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    start = 0
    while start < len(clean):
        hard_end = min(start + chunk_size, len(clean))
        end = hard_end
        if hard_end < len(clean):
            candidates = [clean.rfind("\n", start, hard_end), clean.rfind(". ", start, hard_end)]
            boundary = max(candidates)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        value = clean[start:end].strip()
        if value:
            chunks.append(Chunk(text=value, page=page, index=len(chunks)))
        if end >= len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks
