"""Structure-aware chunking for Markdown.

Each chunk stays under the heading it came from, so a passage like "Refunds take
5-7 days" is stored as "Billing > Refunds: Refunds take 5-7 days" and still makes
sense when it is retrieved on its own.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    heading_path: tuple[str, ...]
    text: str
    metadata: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def id(self) -> str:
        return f"{self.doc_id}#{self.index}"

    @property
    def content(self) -> str:
        """What gets embedded and indexed: the heading path, then the passage."""
        if self.heading_path:
            return " > ".join(self.heading_path) + ": " + self.text
        return self.text


def content_hash(text: str) -> str:
    """Stable hash used to skip re-embedding a document that hasn't changed."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sections(markdown: str) -> list[tuple[tuple[str, ...], str]]:
    """Split a document into (heading path, body) sections."""
    path: list[tuple[int, str]] = []
    sections: list[tuple[tuple[str, ...], list[str]]] = [((), [])]
    for line in markdown.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, title))
            sections.append((tuple(t for _, t in path), []))
        else:
            sections[-1][1].append(line)
    return [(p, "\n".join(body).strip()) for p, body in sections if "\n".join(body).strip()]


def chunk_markdown(
    doc_id: str,
    markdown: str,
    max_words: int = 120,
    overlap_words: int = 20,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split Markdown into chunks of at most ``max_words`` words.

    Sections shorter than the limit become one chunk. Longer sections are split
    on a sliding window with ``overlap_words`` of overlap, so a sentence that
    straddles a boundary is still found whole in one of the two chunks.
    """
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    chunks: list[Chunk] = []
    for heading_path, body in _sections(markdown):
        words = body.split()
        step = max_words - overlap_words
        start = 0
        while True:
            window = words[start : start + max_words]
            chunks.append(
                Chunk(doc_id, len(chunks), heading_path, " ".join(window), dict(metadata or {}))
            )
            if start + max_words >= len(words):
                break
            start += step
    return chunks
