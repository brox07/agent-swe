"""Milestone 2: documentation loaders, source validation, ingestion, and search.

Every format is exercised against a small document built in the test, so the
suite stays offline. The loaders were also run against the real Python, pytest,
SQLAlchemy, FastAPI and Pydantic sources; see the design doc.
"""

from __future__ import annotations

import asyncio
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from src.db.models import SyncStatus
from src.docs import loaders
from src.docs.html import html_sections
from src.docs.ingest import DocIngestService
from src.docs.sources import SourceError, check_url, resolve_target
from src.vector.qdrant import QdrantStore
from src.vector.search import SearchService

from .conftest import StubEmbedder

SPHINX_PAGE = """
<html><body>
<nav>site navigation</nav>
<div class="body" role="main">
  <section id="streams">
    <h1>Streams<a class="headerlink" href="#streams">¶</a></h1>
    <p>Streams are high-level <code>async</code> primitives.</p>
    <dl class="py function">
      <dt id="asyncio.open_connection"><em>async</em>
        asyncio.<span>open_connection</span>(<em>host</em>)</dt>
      <dd><p>Establish a network connection.</p>
          <pre>reader, writer = await asyncio.open_connection("x", 1)</pre></dd>
    </dl>
    <p>Prose after the entry belongs to Streams again.</p>
    <section id="examples">
      <h2>Examples</h2>
      <ul><li>First example</li><li>Second example</li></ul>
      <table>
        <tr><th>Name</th><th>Meaning</th></tr>
        <tr><td>limit</td><td>buffer size</td></tr>
      </table>
    </section>
  </section>
</div>
<footer>copyright</footer>
</body></html>
"""


def _by_path(sections):
    return {" > ".join(s.heading_path): s for s in sections}


class TestHtml:
    def test_headings_and_api_entries_become_sections(self):
        sections = _by_path(html_sections(SPHINX_PAGE, "library/asyncio-stream.html"))
        assert set(sections) == {
            "Streams",
            "Streams > async asyncio.open_connection(host)",
            "Streams > Examples",
        }

    def test_api_entry_keeps_its_own_anchor_and_code(self):
        entry = _by_path(html_sections(SPHINX_PAGE, "p.html"))[
            "Streams > async asyncio.open_connection(host)"
        ]
        assert entry.location == "p.html#asyncio.open_connection"
        assert '```\nreader, writer = await asyncio.open_connection("x", 1)\n```' in entry.text

    def test_prose_after_an_api_entry_returns_to_the_enclosing_heading(self):
        streams = [s for s in html_sections(SPHINX_PAGE, "p.html") if s.heading_path == ["Streams"]]
        assert any("Prose after the entry" in s.text for s in streams)
        assert all(s.location == "p.html#streams" for s in streams)

    def test_navigation_footer_and_permalinks_are_dropped(self):
        text = " ".join(s.text for s in html_sections(SPHINX_PAGE, "p.html"))
        assert "site navigation" not in text
        assert "copyright" not in text
        assert "¶" not in text

    def test_inline_markup_does_not_break_spacing(self):
        sections = html_sections(SPHINX_PAGE, "p.html")
        text = " ".join(s.text for s in sections if s.heading_path == ["Streams"])
        assert "Streams are high-level async primitives." in text

    def test_lists_and_tables_survive(self):
        examples = _by_path(html_sections(SPHINX_PAGE, "p.html"))["Streams > Examples"]
        assert "- First example" in examples.text
        assert "limit | buffer size" in examples.text

    def test_a_single_page_build_keeps_content_inside_the_toctree(self):
        page = """<div role="main"><h1>Docs</h1>
        <div class="toctree-wrapper"><section id="a"><h2>Fixtures</h2><p>Body.</p></section></div>
        <div class="toctree-wrapper"><ul><li>nav only</li></ul></div></div>"""
        sections = _by_path(html_sections(page, "index.html"))
        assert sections["Docs > Fixtures"].text == "Body."
        assert "nav only" not in " ".join(s.text for s in sections.values())


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, body in files.items():
            z.writestr(name, body)
    return buf.getvalue()


class TestHtmlArchive:
    def test_wrapper_directory_is_stripped_and_index_pages_skipped(self):
        data = _zip(
            {
                "python-3.14-docs-html/library/asyncio-stream.html": SPHINX_PAGE,
                "python-3.14-docs-html/genindex-A.html": SPHINX_PAGE,
                "python-3.14-docs-html/_sources/x.html": SPHINX_PAGE,
                "python-3.14-docs-html/whatsnew/changelog.html": SPHINX_PAGE,
                "python-3.14-docs-html/_static/style.css": "body{}",
            }
        )
        doc = loaders.html_archive(data, "Python")
        pages = {s.location.split("#")[0] for s in doc.sections}
        assert pages == {"library/asyncio-stream.html"}


def _epub(chapters: list[tuple[str, str]], title: str = "Fluent Testing") -> bytes:
    manifest = "".join(
        f'<item id="c{i}" href="text/{name}" media-type="application/xhtml+xml"/>'
        for i, (name, _) in enumerate(chapters)
    )
    # Spine order deliberately differs from file-name order.
    spine = "".join(f'<itemref idref="c{i}"/>' for i in reversed(range(len(chapters))))
    files = {
        "mimetype": "application/epub+zip",
        "META-INF/container.xml": (
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>'
        ),
        "OEBPS/content.opf": (
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
            f"<dc:title>{title}</dc:title></metadata>"
            f"<manifest>{manifest}</manifest><spine>{spine}</spine></package>"
        ),
    }
    for name, body in chapters:
        files[f"OEBPS/text/{name}"] = f"<html><body>{body}</body></html>"
    return _zip(files)


class TestEpub:
    def test_chapters_follow_the_spine_and_the_title_comes_from_metadata(self):
        data = _epub(
            [
                ("ch01.xhtml", "<section><h1>Fixtures</h1><p>Use fixtures.</p></section>"),
                ("ch02.xhtml", "<section><h1>Mocks</h1><p>Mock sparingly.</p></section>"),
            ]
        )
        doc = loaders.epub(data, "fallback")
        assert doc.title == "Fluent Testing"
        assert [s.heading_path for s in doc.sections] == [["Mocks"], ["Fixtures"]]
        assert doc.sections[0].location == "text/ch02.xhtml"

    def test_nested_sections_keep_the_chapter_when_every_heading_is_h1(self):
        body = (
            '<section data-type="chapter"><h1>Chapter 5. Recursion</h1><p>Intro.</p>'
            '<section data-type="sect1"><h1>Base Cases</h1><p>Stop here.</p>'
            '<section data-type="sect2"><h1>Pitfalls</h1><p>Careful.</p></section>'
            "</section>"
            '<section data-type="sect1"><h1>Infinite Recursion</h1><p>Never ends.</p></section>'
            "</section>"
        )
        doc = loaders.epub(_epub([("ch05.xhtml", body)]), "fallback")
        assert [s.heading_path for s in doc.sections] == [
            ["Chapter 5. Recursion"],
            ["Chapter 5. Recursion", "Base Cases"],
            ["Chapter 5. Recursion", "Base Cases", "Pitfalls"],
            ["Chapter 5. Recursion", "Infinite Recursion"],
        ]

    def test_front_and_back_matter_are_skipped(self):
        data = _epub(
            [
                ("ch01.html", "<h1>Real Chapter</h1><p>Content.</p>"),
                ("ix01.html", "<h1>Index</h1><p>asyncio, 12, 40, 91</p>"),
                ("toc01.html", "<h1>Table of Contents</h1><p>Chapter 1</p>"),
                ("copyright-page01.html", "<h1>Copyright</h1><p>All rights reserved.</p>"),
                ("app01.html", "<h1>Appendix A</h1><p>Kept.</p>"),
            ]
        )
        doc = loaders.epub(data, "fallback")
        assert [s.heading_path for s in doc.sections] == [["Appendix A"], ["Real Chapter"]]

    def test_code_listings_keep_their_formatting(self):
        data = _epub(
            [
                (
                    "ch01.xhtml",
                    '<h1>Code</h1><pre data-type="programlisting">def f():\n    return 1</pre>',
                )
            ]
        )
        section = loaders.epub(data, "fallback").sections[0]
        assert "```\ndef f():\n    return 1\n```" in section.text


class TestMarkdown:
    def test_headings_split_sections_and_explicit_anchors_are_used(self):
        text = "# Tutorial { #tutorial }\n\nIntro.\n\n## First Steps\n\nStep one.\n"
        sections = loaders.markdown_sections(text, "tutorial/index.md")
        assert [(s.heading_path, s.location) for s in sections] == [
            (["Tutorial"], "tutorial/index.md#tutorial"),
            (["Tutorial", "First Steps"], "tutorial/index.md#first-steps"),
        ]

    def test_a_hash_inside_a_code_fence_is_not_a_heading(self):
        text = "# Title\n\n```bash\n# install it\npip install x\n```\n"
        sections = loaders.markdown_sections(text, "p.md")
        assert len(sections) == 1
        assert "# install it" in sections[0].text

    def test_front_matter_comments_and_mkdocstrings_directives_are_removed(self):
        text = "---\ntitle: x\n---\n# API\n<!-- hidden -->\n::: pydantic.BaseModel\nReal text.\n"
        sections = loaders.markdown_sections(text, "p.md")
        assert sections[0].text == "Real text."

    def test_fastapi_includes_are_resolved_with_line_ranges(self):
        source = "line1\nline2\nline3\nline4\n"
        text = "# Example\n\n{* ../../docs_src/app.py ln[2:3] hl[2] *}\n"
        sections = loaders.markdown_sections(
            text, "p.md", resolve=lambda path: source if path.endswith("app.py") else None
        )
        assert "```python\nline2\nline3\n```" in sections[0].text

    def test_an_unresolvable_include_leaves_a_marker(self):
        sections = loaders.markdown_sections(
            "# X\n\n{* missing.py *}\n", "p.md", resolve=lambda p: None
        )
        assert "[example: missing.py]" in sections[0].text


def _tarball(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, body in files.items():
            raw = body.encode()
            info = tarfile.TarInfo(f"fastapi-0.141.1/{name}")
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


class TestGithubMarkdown:
    def test_docs_dir_is_selected_includes_resolve_and_excludes_apply(self):
        data = _tarball(
            {
                "docs/en/docs/tutorial/first-steps.md": (
                    "# First Steps\n\n{* ../../docs_src/first_steps/tutorial001.py *}\n"
                ),
                "docs/en/docs/release-notes.md": "# Release Notes\n\nLots.\n",
                "docs/es/docs/index.md": "# Traducción\n\nTexto.\n",
                "docs_src/first_steps/tutorial001.py": "app = FastAPI()\n",
            }
        )
        doc = loaders.github_markdown(
            data, "docs/en/docs", "FastAPI", exclude=("release-notes.md",)
        )
        assert [s.location for s in doc.sections] == ["tutorial/first-steps.md#first-steps"]
        assert "```python\napp = FastAPI()\n```" in doc.sections[0].text


def _pdf_with_outline() -> bytes:
    """Three pages of real text, bookmarked Chapter 1 (p1) > Section 1.1 (p2), Chapter 2 (p3)."""
    from pypdf import PdfReader, PdfWriter

    objects: list[bytes] = []
    pages = ["Front matter", "Chapter one body", "Chapter two body"]
    # 1: catalog, 2: pages, 3: font, then page + content pairs.
    kids = " ".join(f"{4 + i * 2} 0 R" for i in range(len(pages)))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + i * 2} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for n, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{n} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    )

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(out.getvalue())))
    chapter = writer.add_outline_item("Chapter 1", 1)
    writer.add_outline_item("Section 1.1", 1, parent=chapter)
    writer.add_outline_item("Chapter 2", 2)
    writer.add_metadata({"/Title": "Test Book"})
    final = io.BytesIO()
    writer.write(final)
    return final.getvalue()


class TestPdf:
    def test_bookmarks_give_sections_and_front_matter_is_skipped(self):
        doc = loaders.pdf(_pdf_with_outline(), "fallback")
        assert doc.title == "Test Book"
        by_path = {" > ".join(s.heading_path): s for s in doc.sections}
        # Chapter 1 and Section 1.1 start on the same page: the page goes to the later one.
        assert set(by_path) == {"Chapter 1 > Section 1.1", "Chapter 2"}
        assert "Chapter one body" in by_path["Chapter 1 > Section 1.1"].text
        assert by_path["Chapter 2"].location == "p. 3"
        assert "Front matter" not in " ".join(s.text for s in doc.sections)


class TestSources:
    def test_paths_outside_the_documents_root_are_refused(self, settings, tmp_path):
        settings.docs_root = tmp_path / "data"
        settings.docs_root.mkdir()
        (tmp_path / "secret.pdf").write_bytes(b"x")
        with pytest.raises(SourceError, match="outside"):
            resolve_target(settings, "../secret.pdf")

    def test_non_https_and_unlisted_hosts_are_refused(self, settings):
        with pytest.raises(SourceError, match="https"):
            check_url(settings, "http://docs.python.org/x.zip")
        with pytest.raises(SourceError, match="not in DOCS_ALLOWED_HOSTS"):
            check_url(settings, "https://169.254.169.254/latest/meta-data")
        with pytest.raises(SourceError, match="not in DOCS_ALLOWED_HOSTS"):
            check_url(settings, "https://docs.python.org.evil.example/x.zip")

    def test_allowlisted_hosts_and_their_subdomains_pass(self, settings):
        check_url(settings, "https://docs.python.org/3/archives/x.zip")
        check_url(settings, "https://codeload.github.com/o/r/tar.gz/v1")

    def test_a_directory_expands_to_books_preferring_epub(self, settings, tmp_path):
        settings.docs_root = tmp_path / "data"
        books = settings.docs_root / "books"
        books.mkdir(parents=True)
        for name in ("fluent.pdf", "fluent.epub", "cookbook.pdf", "notes.txt"):
            (books / name).write_bytes(b"x")
        targets = resolve_target(settings, "books")
        assert sorted((t.title, t.source_type) for t in targets) == [
            ("cookbook", "pdf"),
            ("fluent", "epub"),
        ]
        assert {t.source_url for t in targets} == {
            "file://books/cookbook.pdf",
            "file://books/fluent.epub",
        }

    def test_presets_resolve_by_name(self, settings):
        (target,) = resolve_target(settings, "Python")
        assert target.framework == "python"
        assert target.version == "3.14"


async def _wait(docs: DocIngestService, sync, job_id: str) -> dict:
    for _ in range(300):
        status = await sync.status(job_id)
        if status["status"] in (SyncStatus.SUCCEEDED.value, SyncStatus.FAILED.value):
            return status
        await asyncio.sleep(0.01)
    raise AssertionError("ingest did not finish")


@pytest.fixture
def docs(settings, store: QdrantStore, embedder: StubEmbedder, tmp_path: Path) -> DocIngestService:
    settings.docs_root = tmp_path / "data"
    (settings.docs_root / "books").mkdir(parents=True)
    return DocIngestService(settings, store, embedder)


def _write_book(settings, name: str = "guide.md") -> Path:
    path = settings.docs_root / "books" / name
    path.write_text(
        "# Concurrency\n\nUse a TaskGroup to cancel sibling tasks together.\n\n"
        "## Timeouts\n\nWrap awaits in asyncio.timeout to bound them.\n"
    )
    return path


@pytest.mark.usefixtures("database")
class TestIngestion:
    async def test_ingest_then_search_finds_the_section(
        self, docs, sync, settings, search: SearchService
    ):
        _write_book(settings)
        started = await docs.start("books/guide.md", framework="Python", version="3.14")
        status = await _wait(docs, sync, started["job_id"])
        assert status["status"] == "succeeded", status
        assert status["chunks_upserted"] == 2

        results = await search.search_docs("asyncio timeout bound awaits")
        assert results[0].section == "Concurrency > Timeouts"
        assert results[0].framework == "python"
        assert results[0].location == "guide.md#timeouts"

    async def test_unchanged_source_is_skipped_and_force_reindexes(
        self, docs, sync, settings, embedder
    ):
        _write_book(settings)
        await _wait(docs, sync, (await docs.start("books/guide.md"))["job_id"])
        calls = embedder.embed_calls

        again = await _wait(docs, sync, (await docs.start("books/guide.md"))["job_id"])
        assert again["files_skipped"] == 1
        assert embedder.embed_calls == calls

        forced = await _wait(docs, sync, (await docs.start("books/guide.md", force=True))["job_id"])
        assert forced["files_skipped"] == 0
        assert embedder.embed_calls > calls

    async def test_a_shrunken_source_leaves_no_orphan_chunks(self, docs, sync, settings, store):
        path = _write_book(settings)
        await _wait(docs, sync, (await docs.start("books/guide.md"))["job_id"])
        path.write_text("# Only\n\nOne section now.\n")
        await _wait(docs, sync, (await docs.start("books/guide.md"))["job_id"])
        assert await store.count_docs("file://books/guide.md") == 1

    async def test_framework_filter_restricts_results(self, docs, sync, settings, search):
        _write_book(settings, "a.md")
        (settings.docs_root / "books" / "b.md").write_text("# Timeouts\n\nFastAPI timeouts.\n")
        await _wait(docs, sync, (await docs.start("books/a.md", framework="python"))["job_id"])
        await _wait(docs, sync, (await docs.start("books/b.md", framework="fastapi"))["job_id"])
        results = await search.search_docs("timeouts", framework="fastapi", limit=5)
        assert results and all(r.framework == "fastapi" for r in results)

    async def test_list_sources_reports_what_was_ingested(self, docs, sync, settings):
        _write_book(settings)
        await _wait(docs, sync, (await docs.start("books/guide.md", framework="python"))["job_id"])
        (source,) = await docs.list_sources()
        assert source["title"] == "guide"
        assert source["chunks"] == 2
        assert source["framework"] == "python"

    async def test_one_bad_book_does_not_sink_the_batch(self, docs, sync, settings):
        _write_book(settings)
        (settings.docs_root / "books" / "broken.epub").write_bytes(b"not a zip")
        status = await _wait(docs, sync, (await docs.start("books"))["job_id"])
        assert status["status"] == "succeeded"
        assert status["files_done"] == 1
        assert "broken" in status["error"]

    async def test_a_bad_source_fails_the_call_not_a_job(self, docs):
        with pytest.raises(SourceError):
            await docs.start("https://example.com/docs.zip")
