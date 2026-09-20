"""Chunk model and language dispatch shared by all parsers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

# Extensions handled by a real grammar. Milestone 1 is Python + TypeScript, per
# the spec; .tsx and .cts/.mts use TypeScript dialect grammars but report
# language="typescript" so a single filter value covers them.
AST_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
}

# Extensions handled by the line-based fallback chunker.
GENERIC_EXTENSIONS: dict[str, str] = {
    ".yml": "yaml",
    ".yaml": "yaml",
}

GENERIC_FILENAMES: dict[str, str] = {
    "dockerfile": "dockerfile",
    "containerfile": "dockerfile",
}


@dataclass(slots=True)
class CodeChunk:
    """One indexable unit of code.

    Repository-level fields (repo_name, commit_sha, dirty) are attached later by
    the sync orchestrator, which is the only layer that knows about them.
    """

    language: str
    node_type: str
    identifier: str
    node_path: str
    start_line: int
    end_line: int
    content: str
    parent_identifier: str | None = None
    # chunk_index/chunk_total describe an oversized node split across several
    # points. is_truncated is reserved for content genuinely dropped, which only
    # happens when a single line exceeds the model window (e.g. a minified file).
    chunk_index: int = 0
    chunk_total: int = 1
    is_truncated: bool = False
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8", "replace")).hexdigest()


def detect_language(path: Path | str) -> str | None:
    """Return the language label for a path, or None if it is not indexable."""
    p = Path(path)
    name = p.name.lower()
    if name in GENERIC_FILENAMES:
        return GENERIC_FILENAMES[name]
    # Dockerfile.prod, Dockerfile.base, ...
    if name.startswith("dockerfile."):
        return "dockerfile"
    suffix = p.suffix.lower()
    if suffix in AST_EXTENSIONS:
        return AST_EXTENSIONS[suffix]
    if suffix in GENERIC_EXTENSIONS:
        return GENERIC_EXTENSIONS[suffix]
    return None


def split_oversized(chunk: CodeChunk, max_chars: int) -> list[CodeChunk]:
    """Split a chunk that exceeds the embedding window into ordered parts.

    Each part after the first is prefixed with the node's own signature line so
    the fragment still carries the context a reader (or a reranker) needs. This
    replaces the silent truncation the spec's 512-token model would have applied
    to every large class.
    """
    if len(chunk.content) <= max_chars:
        return [chunk]

    comment = "#" if chunk.language in ("python", "yaml", "dockerfile") else "//"
    lines = chunk.content.splitlines(keepends=True)
    signature = lines[0].strip() if lines else chunk.identifier

    parts: list[list[str]] = []
    current: list[str] = []
    size = 0
    truncated = False
    for line in lines:
        # A single line longer than the whole budget is the only case where
        # content is actually lost.
        if len(line) > max_chars:
            line = line[:max_chars]
            truncated = True
        if size + len(line) > max_chars and current:
            parts.append(current)
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        parts.append(current)

    total = len(parts)
    out: list[CodeChunk] = []
    line_cursor = chunk.start_line
    for i, part in enumerate(parts):
        body = "".join(part)
        header = (
            "" if i == 0 else f"{comment} {chunk.node_path} (part {i + 1}/{total}): {signature}\n"
        )
        span = len(part)
        out.append(
            CodeChunk(
                language=chunk.language,
                node_type=chunk.node_type,
                identifier=chunk.identifier,
                node_path=chunk.node_path,
                parent_identifier=chunk.parent_identifier,
                start_line=line_cursor,
                end_line=line_cursor + span - 1,
                content=header + body,
                chunk_index=i,
                chunk_total=total,
                is_truncated=truncated,
                extra=dict(chunk.extra),
            )
        )
        line_cursor += span
    return out
