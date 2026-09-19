"""HTML to sections: Sphinx documentation pages and EPUB chapters.

Both are HTML under the hood, so one walker serves both. Headings open
sections. API entries — Sphinx's ``<dl class="py function">`` and friends — open
sections too, one level below any heading, so ``asyncio.open_connection`` gets
its own chunk instead of being buried in a page-long "Streams" section.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString, Tag

from src.docs.sections import DocSection

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_API_KINDS = {
    "class",
    "function",
    "method",
    "classmethod",
    "staticmethod",
    "attribute",
    "property",
    "exception",
    "data",
    "decorator",
    "decoratormethod",
    "coroutinefunction",
    "coroutinemethod",
    "fixture",
    "module",
    "type",
}
# API entries nest below every real heading level.
_API_BASE_LEVEL = 10
_INLINE_TAGS = {
    "a",
    "abbr",
    "b",
    "br",
    "cite",
    "code",
    "em",
    "i",
    "kbd",
    "mark",
    "q",
    "s",
    "samp",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "tt",
    "u",
    "var",
    "img",
    "wbr",
}
_DROP_SELECTORS = [
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "a.headerlink",
    "div.sphinxsidebar",
    "div.related",
    "div.footer",
    "span.linenos",
    "div.clearer",
    "aside.sidebar",
]
_MAIN_SELECTORS = ["[role=main]", "div.body", "article", "#docs-body", "main", "body"]
_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _api_kind(tag: Tag) -> bool:
    classes = set(tag.get("class") or [])
    return bool(classes & _API_KINDS)


class _Builder:
    def __init__(self, page: str, base_path: list[str]) -> None:
        self.page = page
        self.base_path = base_path
        # (level, title, anchor): the anchor is restored when a nested entry closes.
        self.stack: list[tuple[int, str, str]] = []
        self.page_anchor = ""
        self.buf: list[str] = []
        self.sections: list[DocSection] = []

    def flush(self) -> None:
        text = "\n\n".join(b for b in self.buf if b.strip())
        self.buf = []
        if text.strip():
            anchor = next((a for _, _, a in reversed(self.stack) if a), self.page_anchor)
            location = f"{self.page}#{anchor}" if anchor else self.page
            self.sections.append(
                DocSection(
                    heading_path=[*self.base_path, *(t for _, t, _ in self.stack)],
                    text=text,
                    location=location,
                )
            )

    def open(self, level: int, title: str, anchor: str | None) -> None:
        self.flush()
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title, anchor or ""))

    def close(self, level: int) -> None:
        """End an API entry so trailing prose goes back to the enclosing heading."""
        self.flush()
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()

    def add(self, text: str) -> None:
        if text.strip():
            self.buf.append(text)


def _anchor_of(tag: Tag) -> str | None:
    if tag.get("id"):
        return str(tag["id"])
    parent = tag.parent
    # Sphinx puts the id on the enclosing <section>, not the heading.
    if isinstance(parent, Tag) and parent.name in ("section", "div") and parent.get("id"):
        return str(parent["id"])
    return None


def _heading_level(heading: Tag) -> int:
    """Nesting depth, not tag name, when the page uses HTML5 sectioning.

    O'Reilly EPUBs mark every level with <h1> and express hierarchy through
    nested <section> elements, so the tag alone would flatten "Chapter 5 >
    Recursion" to "Recursion". Sphinx nests sections the same way with matching
    tags, so depth agrees with the tag there. The larger of the two wins, so a
    page title outside any section still ranks above the sections that follow.
    """
    depth = sum(1 for parent in heading.parents if parent.name == "section")
    return max(depth, int(heading.name[1]))


def _signature(dt: Tag) -> str:
    """An API entry's signature as one line, capped: it doubles as a heading."""
    text = _clean(dt.get_text())
    return text if len(text) <= 160 else text[:157] + "..."


def _table_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_clean(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _walk(node: Tag, b: _Builder, api_depth: int = 0) -> None:
    inline: list[str] = []

    def flush_inline() -> None:
        if inline:
            b.add(_clean("".join(inline)))
            inline.clear()

    for child in node.children:
        if isinstance(child, NavigableString):
            if str(child).strip() or inline:
                inline.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in _INLINE_TAGS:
            # Inline elements carry their own spacing in the surrounding text;
            # joining with a separator turns "f()." into "f() .".
            inline.append(child.get_text())
            continue
        flush_inline()
        if name in _HEADING_TAGS:
            b.open(_heading_level(child), _clean(child.get_text()), _anchor_of(child))
        elif name == "pre":
            b.add("```\n" + child.get_text().strip("\n") + "\n```")
        elif name == "table":
            b.add(_table_text(child))
        elif name == "dl" and _api_kind(child):
            level = _API_BASE_LEVEL + api_depth
            for item in child.find_all(["dt", "dd"], recursive=False):
                if item.name == "dt":
                    b.open(level, _signature(item), item.get("id"))
                else:
                    _walk(item, b, api_depth + 1)
            b.close(level)
        elif name == "li":
            if child.find(["p", "pre", "ul", "ol", "div", "dl", "table"], recursive=False):
                _walk(child, b, api_depth)
            else:
                b.add("- " + _clean(child.get_text()))
        elif name in ("p", "dt", "caption", "figcaption"):
            b.add(_clean(child.get_text()))
        else:
            _walk(child, b, api_depth)
    flush_inline()


def html_sections(html: str, page: str, base_path: list[str] | None = None) -> list[DocSection]:
    """Sections from one HTML page, in document order.

    ``page`` is the page's path within its archive or book, used for locations.
    ``base_path`` prefixes every heading trail — an EPUB passes nothing, since its
    chapter titles are the page's own h1.
    """
    soup = BeautifulSoup(html, "html.parser")
    root: Tag | None = None
    for selector in _MAIN_SELECTORS:
        root = soup.select_one(selector)
        if root is not None:
            break
    if root is None:
        return []
    for selector in _DROP_SELECTORS:
        for junk in root.select(selector):
            junk.decompose()
    # A table of contents is navigation, except in a single-page build, where
    # Sphinx inlines every page inside it. Keep the wrapper when it holds content.
    for toc in root.select("div.toctree-wrapper"):
        if toc.find(list(_HEADING_TAGS)) is None:
            toc.decompose()
    builder = _Builder(page, base_path or [])
    _walk(root, builder)
    builder.flush()
    return builder.sections
