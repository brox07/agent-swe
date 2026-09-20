"""Format loaders: raw bytes in, a titled list of sections out.

Each loader is a pure function over bytes, so every format is testable without
the network, a database, or a model.
"""

from __future__ import annotations

import io
import logging
import posixpath
import re
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import dataclass

from src.docs.html import html_sections
from src.docs.sections import DocSection

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoadedDoc:
    title: str
    sections: list[DocSection]


# Pages that are navigation, legal boilerplate, or release history: large, and
# never the answer to a "how do I" question.
_ARCHIVE_SKIP_RE = re.compile(
    r"(^|/)(genindex|py-modindex|modindex|search|searchindex|glossary_index)[^/]*\.html$"
    r"|(^|/)_(static|sources|images|modules|downloads)/"
    r"|(^|/)(changelog|changes|license|copyright|about|bugs|download)[^/]*\.html$"
    r"|(^|/)changelog/"
    r"|(^|/)announce/"
    r"|(^|/)whatsnew/changelog\.html$"
    r"|(^|/)(internals)\.html$",
    re.IGNORECASE,
)


def _strip_top_dir(names: list[str]) -> str:
    """Archives usually wrap everything in one directory; locations omit it."""
    tops = {n.split("/", 1)[0] for n in names if "/" in n}
    if len(tops) == 1 and all("/" in n or n in tops for n in names):
        return next(iter(tops)) + "/"
    return ""


def html_archive(data: bytes, title: str) -> LoadedDoc:
    """A zipped Sphinx HTML build: docs.python.org's archive, a Read the Docs htmlzip."""
    sections: list[DocSection] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        prefix = _strip_top_dir(names)
        for name in sorted(names):
            if not name.lower().endswith((".html", ".htm")):
                continue
            page = name[len(prefix) :]
            if _ARCHIVE_SKIP_RE.search(page):
                continue
            html = archive.read(name).decode("utf-8", errors="replace")
            sections.extend(html_sections(html, page))
    return LoadedDoc(title=title, sections=sections)


# --- EPUB ----------------------------------------------------------------------

# Front and back matter: no prose worth retrieving, and a book's index is an
# alphabetical keyword list that matches almost any query. Prefaces, appendices
# and afterwords are kept — they are real content.
_EPUB_SKIP_RE = re.compile(
    r"(^|/)(cover|titlepage|title-page|copyright[-\w]*|toc|nav|ix|index|colophon|"
    r"dedication|praise)[-\w]*\d*\.x?html?$",
    re.IGNORECASE,
)

_NS = {
    "c": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def epub(data: bytes, fallback_title: str) -> LoadedDoc:
    """Chapters in reading (spine) order. EPUB is a zip of XHTML, so no extra library."""
    with zipfile.ZipFile(io.BytesIO(data)) as book:
        container = ET.fromstring(book.read("META-INF/container.xml"))
        rootfile = container.find(".//c:rootfile", _NS)
        if rootfile is None:
            raise ValueError("EPUB has no rootfile in META-INF/container.xml")
        opf_path = rootfile.attrib["full-path"]
        opf_dir = posixpath.dirname(opf_path)
        opf = ET.fromstring(book.read(opf_path))

        title_el = opf.find(".//dc:title", _NS)
        title = (title_el.text or "").strip() if title_el is not None else ""

        manifest = {
            item.attrib["id"]: item.attrib["href"]
            for item in opf.findall(".//opf:manifest/opf:item", _NS)
        }
        sections: list[DocSection] = []
        for itemref in opf.findall(".//opf:spine/opf:itemref", _NS):
            href = manifest.get(itemref.attrib.get("idref", ""))
            if not href:
                continue
            if _EPUB_SKIP_RE.search(href):
                continue
            path = posixpath.normpath(posixpath.join(opf_dir, href.split("#", 1)[0]))
            try:
                html = book.read(path).decode("utf-8", errors="replace")
            except KeyError:
                logger.warning("EPUB spine entry %s missing from archive", path)
                continue
            sections.extend(html_sections(html, href))
    return LoadedDoc(title=title or fallback_title, sections=sections)


# --- Markdown ------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_ANCHOR_RE = re.compile(r"\s*\{\s*#([\w-]+)\s*\}\s*$")
_MD_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# FastAPI:  {* ../../docs_src/first_steps/tutorial001.py ln[3:7] hl[3] *}
_INCLUDE_STAR_RE = re.compile(r"\{\*\s*(\S+)(.*?)\*\}")
# Older mkdocs include syntax:  {!../../docs_src/app.py!}
_INCLUDE_BANG_RE = re.compile(r"\{!\s*(\S+?)\s*!\}")
_LN_RE = re.compile(r"ln\[(\d+)(?::(\d+))?\]")
# mkdocstrings directives render API pages at build time; the source holds nothing.
_MKDOCSTRINGS_RE = re.compile(r"^:::\s")

Resolver = Callable[[str], str | None]


def _slug(title: str) -> str:
    return re.sub(r"[^\w-]+", "-", title.lower()).strip("-")


def _expand_includes(text: str, resolve: Resolver | None) -> str:
    def star(match: re.Match[str]) -> str:
        path, opts = match.group(1), match.group(2)
        body = resolve(path) if resolve else None
        if body is None:
            return f"[example: {path}]"
        ln = _LN_RE.search(opts)
        if ln:
            lines = body.splitlines()
            start = int(ln.group(1))
            end = int(ln.group(2)) if ln.group(2) else start
            body = "\n".join(lines[start - 1 : end])
        lang = "python" if path.endswith(".py") else ""
        return f"```{lang}\n{body.rstrip()}\n```"

    def bang(match: re.Match[str]) -> str:
        body = resolve(match.group(1)) if resolve else None
        return body.rstrip() if body is not None else f"[example: {match.group(1)}]"

    return _INCLUDE_BANG_RE.sub(bang, _INCLUDE_STAR_RE.sub(star, text))


def markdown_sections(
    text: str, page: str, base_path: list[str] | None = None, resolve: Resolver | None = None
) -> list[DocSection]:
    """Split on ATX headings outside code fences. Markdown is already good text
    to embed, so bodies are kept verbatim apart from resolved includes."""
    text = _FRONT_MATTER_RE.sub("", text)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _expand_includes(text, resolve)

    base = list(base_path or [])
    sections: list[DocSection] = []
    stack: list[tuple[int, str, str]] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if body:
            anchor = stack[-1][2] if stack else ""
            sections.append(
                DocSection(
                    heading_path=[*base, *(t for _, t, _ in stack)],
                    text=body,
                    location=f"{page}#{anchor}" if anchor else page,
                )
            )

    for line in text.splitlines():
        if _MD_FENCE_RE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        if not in_fence:
            if _MKDOCSTRINGS_RE.match(line):
                continue
            heading = _MD_HEADING_RE.match(line)
            if heading:
                flush()
                level = len(heading.group(1))
                title = heading.group(2)
                anchor_match = _MD_ANCHOR_RE.search(title)
                anchor = anchor_match.group(1) if anchor_match else _slug(title)
                title = _MD_ANCHOR_RE.sub("", title).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title, anchor))
                continue
        buf.append(line)
    flush()
    return sections


def github_markdown(
    data: bytes, docs_path: str, title: str, exclude: tuple[str, ...] = ()
) -> LoadedDoc:
    """Markdown docs from a GitHub tarball, resolving includes against the repo.

    FastAPI's docs keep every code example in ``docs_src/`` and pull it in with
    an include directive; without resolving those, the tutorials lose their code.
    """
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        files: dict[str, bytes] = {}
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # Drop the "<repo>-<ref>/" wrapper directory.
            rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if rel.endswith((".md", ".py", ".toml", ".json", ".yaml", ".yml", ".txt")):
                handle = tar.extractfile(member)
                if handle is not None:
                    files[rel] = handle.read()

    root = docs_path.strip("/") + "/"
    sections: list[DocSection] = []
    for rel in sorted(files):
        if not rel.startswith(root) or not rel.endswith(".md"):
            continue
        page = rel[len(root) :]
        if any(page.startswith(prefix) for prefix in exclude):
            continue
        page_dir = posixpath.dirname(rel)

        def resolve(path: str, page_dir: str = page_dir) -> str | None:
            # mkdocs resolves includes against the directory holding mkdocs.yml —
            # the docs directory's parent — not the page. The page's own directory
            # is the fallback for relative includes written the other way.
            for base in (posixpath.dirname(root.rstrip("/")), page_dir):
                raw = files.get(posixpath.normpath(posixpath.join(base, path)))
                if raw is not None:
                    return raw.decode("utf-8", errors="replace")
            return None

        text = files[rel].decode("utf-8", errors="replace")
        sections.extend(markdown_sections(text, page, resolve=resolve))
    return LoadedDoc(title=title, sections=sections)


# --- Obsidian --------------------------------------------------------------------

# .obsidian is editor config; .trash holds notes the author deleted and would be
# startled to see resurface in a search.
VAULT_SKIP_DIRS = {".obsidian", ".trash", ".git", "node_modules", ".stfolder"}
_FRONT_MATTER_BLOCK_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_EMBED_RE = re.compile(r"!\[\[[^\]]*\]\]")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
# Frontmatter worth searching. "updated" and the rest are metadata, not content.
_KEPT_FRONT_MATTER = ("tags", "type", "aliases", "status", "project")


def _front_matter_line(text: str) -> str:
    """Frontmatter tags and aliases as a searchable line, not silently dropped."""
    match = _FRONT_MATTER_BLOCK_RE.match(text)
    if not match:
        return ""
    kept = []
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip().strip("[]").strip()
        if key in _KEPT_FRONT_MATTER and value:
            kept.append(f"{key}: {value}")
    return " · ".join(kept)


def obsidian_note(text: str, rel_path: str, title: str) -> list[DocSection]:
    """One note's sections, titled by its filename.

    Only a minority of notes open with an H1, so the filename carries the title;
    without it a section trail reads "Requirements" with no hint of which note.
    Wikilinks become their display text and image embeds are dropped.
    """
    body = _FRONT_MATTER_BLOCK_RE.sub("", text)
    body = _EMBED_RE.sub("", body)
    body = _WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), body)
    lead = _front_matter_line(text)
    if lead:
        body = f"{lead}\n\n{body}"
    return markdown_sections(body, rel_path, base_path=[title])


def vault(root, name: str):
    """An Obsidian vault as one document: every note, in path order.

    The vault is a single source rather than one source per note, so a re-ingest
    replaces it wholesale and notes deleted since the last run take their chunks
    with them. That costs a full re-embed on any change, which is minutes at the
    scale of a personal vault.
    """
    sections: list[DocSection] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        if any(part in VAULT_SKIP_DIRS for part in rel.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("skipping unreadable note %s: %s", rel, exc)
            continue
        sections.extend(obsidian_note(text, rel.as_posix(), path.stem))
    return LoadedDoc(title=name, sections=sections)


# --- PDF -----------------------------------------------------------------------


def _flatten_outline(reader, outline, depth: int = 0) -> list[tuple[int, str, int]]:
    """pypdf's nested outline as (depth, title, page_index) in reading order."""
    entries: list[tuple[int, str, int]] = []
    for item in outline:
        if isinstance(item, list):
            entries.extend(_flatten_outline(reader, item, depth + 1))
            continue
        try:
            page = reader.get_destination_page_number(item)
        except Exception:  # noqa: BLE001 - a broken bookmark should not sink the book
            continue
        if page is None or page < 0:
            continue
        entries.append((depth, str(item.title).strip(), page))
    return entries


def pdf(data: bytes, fallback_title: str, pages_per_section: int = 3) -> LoadedDoc:
    """Sections from the PDF's bookmarks; fixed page windows if it has none.

    Each page belongs to the last bookmark that starts on or before it, and the
    heading trail is that bookmark's ancestry. Pages before the first bookmark
    are front matter (cover, praise, copyright) and are skipped.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    meta_title = ""
    if reader.metadata and reader.metadata.title:
        meta_title = str(reader.metadata.title).strip()
    title = meta_title or fallback_title

    try:
        outline = _flatten_outline(reader, reader.outline)
    except Exception:  # noqa: BLE001
        outline = []

    page_texts = []
    for page in reader.pages:
        try:
            page_texts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - one bad page should not sink the book
            page_texts.append("")

    sections: list[DocSection] = []
    if not outline:
        for start in range(0, len(page_texts), pages_per_section):
            end = min(start + pages_per_section, len(page_texts))
            text = "\n\n".join(page_texts[start:end]).strip()
            if text:
                sections.append(
                    DocSection(
                        heading_path=[f"Pages {start + 1}-{end}"],
                        text=text,
                        location=f"p. {start + 1}",
                    )
                )
        return LoadedDoc(title=title, sections=sections)

    outline.sort(key=lambda e: e[2])
    for i, (depth, heading, start) in enumerate(outline):
        end = outline[i + 1][2] if i + 1 < len(outline) else len(page_texts)
        # Two bookmarks on one page: the page's text goes to the later one.
        if end <= start:
            continue
        # Ancestry: the nearest earlier entry at each shallower depth.
        trail = [heading]
        want = depth - 1
        for prev_depth, prev_title, _ in reversed(outline[:i]):
            if want < 0:
                break
            if prev_depth == want:
                trail.insert(0, prev_title)
                want -= 1
        text = "\n\n".join(page_texts[start:end]).strip()
        if text:
            sections.append(DocSection(heading_path=trail, text=text, location=f"p. {start + 1}"))
    return LoadedDoc(title=title, sections=sections)
