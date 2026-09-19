"""Line-based semantic chunking for Dockerfiles and YAML.

No grammar is used here. Both formats have enough structure in their line layout
to split on meaningfully: Dockerfiles at build-stage boundaries, YAML at
top-level keys and (for Compose files) at each service definition.
"""

from __future__ import annotations

import re

from src.parser.base import CodeChunk, split_oversized

_FROM_RE = re.compile(r"^\s*FROM\s+(?P<image>\S+)(?:\s+AS\s+(?P<alias>\S+))?", re.IGNORECASE)
# A mapping key at a given indentation, e.g. "services:" or "  api:".
_KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_.\-]+)\s*:\s*(?P<inline>.*)$")
# Compose sections whose children are each worth indexing on their own.
_EXPANDED_SECTIONS = {"services", "jobs", "volumes", "networks"}


def _build(
    language: str,
    identifier: str,
    node_path: str,
    parent: str | None,
    lines: list[str],
    start_line: int,
    max_chunk_chars: int,
) -> list[CodeChunk]:
    content = "".join(lines).rstrip("\n")
    if not content.strip():
        return []
    chunk = CodeChunk(
        language=language,
        node_type="block",
        identifier=identifier,
        node_path=node_path,
        parent_identifier=parent,
        start_line=start_line,
        end_line=start_line + len(lines) - 1,
        content=content,
    )
    return split_oversized(chunk, max_chunk_chars)


def parse_dockerfile(source_text: str, max_chunk_chars: int) -> list[CodeChunk]:
    """One chunk per build stage, named by its AS alias where present."""
    lines = source_text.splitlines(keepends=True)
    boundaries = [i for i, line in enumerate(lines) if _FROM_RE.match(line)]
    if not boundaries:
        return _build("dockerfile", "dockerfile", "dockerfile", None, lines, 1, max_chunk_chars)

    chunks: list[CodeChunk] = []
    # Anything before the first FROM (ARGs, comments) is its own preamble chunk.
    if boundaries[0] > 0:
        chunks += _build(
            "dockerfile", "preamble", "preamble", None, lines[: boundaries[0]], 1, max_chunk_chars
        )

    for n, start in enumerate(boundaries):
        end = boundaries[n + 1] if n + 1 < len(boundaries) else len(lines)
        match = _FROM_RE.match(lines[start])
        assert match is not None
        alias = match.group("alias") or f"stage-{n}"
        chunks += _build(
            "dockerfile",
            alias,
            f"stage.{alias}",
            None,
            lines[start:end],
            start + 1,
            max_chunk_chars,
        )
    return chunks


def parse_yaml(source_text: str, max_chunk_chars: int) -> list[CodeChunk]:
    """Chunk per top-level key, expanding service-like sections one level deeper."""
    lines = source_text.splitlines(keepends=True)
    top: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _KEY_RE.match(line)
        if match and match.group("indent") == "":
            top.append((match.group("key"), i))

    if not top:
        return _build("yaml", "document", "document", None, lines, 1, max_chunk_chars)

    chunks: list[CodeChunk] = []
    for n, (key, start) in enumerate(top):
        end = top[n + 1][1] if n + 1 < len(top) else len(lines)
        section = lines[start:end]

        children = _child_keys(section)
        if key in _EXPANDED_SECTIONS and children:
            # Index each service/job separately: "the postgres service" is a
            # distinct thing to search for from the compose file as a whole.
            for m, (child_key, child_offset) in enumerate(children):
                child_end = children[m + 1][1] if m + 1 < len(children) else len(section)
                chunks += _build(
                    "yaml",
                    child_key,
                    f"{key}.{child_key}",
                    key,
                    section[child_offset:child_end],
                    start + child_offset + 1,
                    max_chunk_chars,
                )
            continue

        chunks += _build("yaml", key, key, None, section, start + 1, max_chunk_chars)
    return chunks


def _child_keys(section: list[str]) -> list[tuple[str, int]]:
    """Keys at the first indentation level inside a section, with line offsets."""
    child_indent: int | None = None
    found: list[tuple[str, int]] = []
    for offset, line in enumerate(section[1:], start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _KEY_RE.match(line)
        if not match:
            continue
        indent = len(match.group("indent"))
        if indent == 0:
            break
        if child_indent is None:
            child_indent = indent
        if indent == child_indent:
            found.append((match.group("key"), offset))
    return found


def parse_generic(
    source_text: str, language: str, max_chunk_chars: int
) -> list[CodeChunk]:
    if language == "dockerfile":
        return parse_dockerfile(source_text, max_chunk_chars)
    if language == "yaml":
        return parse_yaml(source_text, max_chunk_chars)
    raise ValueError(f"no generic chunker for language {language!r}")
