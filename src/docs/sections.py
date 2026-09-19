"""Documentation sections and the chunker that turns them into embeddable units.

Every loader — HTML archives, EPUB, PDF, Markdown — reduces its input to a flat
list of ``DocSection``: body text plus the heading trail it sits under. Chunking
is shared, so every source gets the same size discipline and the same guarantee
that a code block is never cut in half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


@dataclass(slots=True)
class DocSection:
    heading_path: list[str]
    text: str
    # Where a reader finds this section: "library/asyncio.html#streams", "p. 212".
    location: str = ""


@dataclass(slots=True)
class DocChunk:
    heading_path: list[str]
    location: str
    content: str
    chunk_index: int = 0
    part: int = 0
    parts: int = 1
    extra: dict = field(default_factory=dict)

    def embed_text(self, doc_title: str) -> str:
        """The breadcrumb travels with the body into the vector.

        A section titled "Examples" means nothing on its own; "asyncio > Streams >
        Examples" is what makes it findable.
        """
        trail = (
            " > ".join([doc_title, *self.heading_path])
            if doc_title
            else " > ".join(self.heading_path)
        )
        return f"{trail}\n\n{self.content}" if trail else self.content


def split_blocks(text: str) -> list[str]:
    """Paragraph-level blocks, with each fenced code block kept as one block."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            if not in_fence:
                if current:
                    blocks.append("\n".join(current))
                    current = []
                in_fence = True
                current.append(line)
                continue
            current.append(line)
            blocks.append("\n".join(current))
            current = []
            in_fence = False
            continue
        if in_fence:
            current.append(line)
        elif line.strip():
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return [b for b in blocks if b.strip()]


def _hard_split(block: str, limit: int) -> list[str]:
    """Last resort for a single block larger than the limit: split on lines."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in block.splitlines():
        while len(line) > limit:
            if current:
                pieces.append("\n".join(current))
                current, size = [], 0
            pieces.append(line[:limit])
            line = line[limit:]
        if size + len(line) + 1 > limit and current:
            pieces.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        pieces.append("\n".join(current))
    return pieces


def chunk_sections(
    sections: list[DocSection], *, target_chars: int, max_chars: int
) -> list[DocChunk]:
    """Pack each section's blocks into chunks of roughly ``target_chars``.

    A chunk never spans two sections, so every chunk has exactly one heading
    trail. A code block is never split unless it alone exceeds ``max_chars``;
    a prose block larger than ``target_chars`` is split on line boundaries.
    """
    chunks: list[DocChunk] = []
    for section in sections:
        body = _BLANK_RUN_RE.sub("\n\n", section.text).strip()
        if not body:
            continue
        groups: list[str] = []
        current: list[str] = []
        size = 0
        for block in split_blocks(body):
            # Code may run to the hard limit, since splitting a listing costs more
            # than an oversized chunk. Prose has no such reason: a PDF page with
            # no paragraph breaks is one "block" and is cut at the target.
            limit = max_chars if _FENCE_RE.match(block) else target_chars
            pieces = _hard_split(block, limit) if len(block) > limit else [block]
            for piece in pieces:
                if current and size + len(piece) + 2 > target_chars:
                    groups.append("\n\n".join(current))
                    current, size = [], 0
                current.append(piece)
                size += len(piece) + 2
        if current:
            groups.append("\n\n".join(current))
        for i, group in enumerate(groups):
            chunks.append(
                DocChunk(
                    heading_path=list(section.heading_path),
                    location=section.location,
                    content=group,
                    part=i,
                    parts=len(groups),
                )
            )
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index
    return chunks
