"""Resolving what to ingest: presets, local paths, and allowlisted URLs.

``ingest_document`` accepts a path or a URL from an MCP client, which makes it a
file-read and SSRF primitive unless constrained. Local paths must resolve inside
the data root, and URLs must be https on an allowlisted host — checked again on
every redirect hop, since an allowlisted host can redirect anywhere.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

from src.config import Settings

SOURCE_TYPES = ("html_archive", "github", "epub", "pdf", "markdown")
MAX_DOWNLOAD_BYTES = 300 * 1024 * 1024
_GITHUB_TREE_RE = re.compile(
    r"^https://github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)/tree/(?P<ref>[^/]+)/(?P<path>.+?)/?$"
)


class SourceError(ValueError):
    """A caller-fixable problem with the requested source."""


@dataclass(slots=True)
class DocTarget:
    source_url: str
    source_type: str
    title: str
    framework: str | None = None
    version: str | None = None
    # Prefix that turns a location ("library/os.html#os.walk") into a link.
    link_base: str | None = None
    # github only: pages under the docs directory to skip.
    exclude: tuple[str, ...] = field(default_factory=tuple)
    local_path: Path | None = None


# The documentation this engine is expected to carry. Pinned versions, so a
# re-ingest is reproducible and the version filter means something.
PRESETS: dict[str, DocTarget] = {
    "python": DocTarget(
        source_url="https://docs.python.org/3/archives/python-3.14-docs-html.zip",
        source_type="html_archive",
        title="Python 3.14 documentation",
        framework="python",
        version="3.14",
        link_base="https://docs.python.org/3.14/",
    ),
    "pytest": DocTarget(
        source_url="https://docs.pytest.org/_/downloads/en/stable/htmlzip/",
        source_type="html_archive",
        title="pytest documentation",
        framework="pytest",
        version="stable",
    ),
    "sqlalchemy": DocTarget(
        source_url="https://docs.sqlalchemy.org/20/sqlalchemy_20.zip",
        source_type="html_archive",
        title="SQLAlchemy 2.0 documentation",
        framework="sqlalchemy",
        version="2.0",
        link_base="https://docs.sqlalchemy.org/en/20/",
    ),
    "fastapi": DocTarget(
        source_url="https://github.com/fastapi/fastapi/tree/0.141.1/docs/en/docs",
        source_type="github",
        title="FastAPI documentation",
        framework="fastapi",
        version="0.141.1",
        exclude=(
            "release-notes.md",
            "fastapi-people.md",
            "external-links.md",
            "management",
            "contributing.md",
            "newsletter.md",
        ),
    ),
    "pydantic": DocTarget(
        source_url="https://github.com/pydantic/pydantic/tree/v2.13.5/docs",
        source_type="github",
        title="Pydantic documentation",
        framework="pydantic",
        version="2.13.5",
        exclude=("contributing.md", "pydantic_people.md", "extra/", "theme/", "logos/"),
    ),
}


def detect_type(path: str) -> str | None:
    lowered = path.lower()
    if lowered.endswith(".epub"):
        return "epub"
    if lowered.endswith(".pdf"):
        return "pdf"
    if lowered.endswith((".md", ".markdown")):
        return "markdown"
    if lowered.endswith(".zip"):
        return "html_archive"
    if _GITHUB_TREE_RE.match(path):
        return "github"
    return None


def resolve_local(settings: Settings, raw: str) -> Path:
    root = settings.docs_root.resolve()
    candidate = Path(raw)
    target = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if target != root and not target.is_relative_to(root):
        raise SourceError(f"{raw!r} resolves outside the documents root {root}")
    if not target.exists():
        raise SourceError(f"{raw!r} does not exist under {root}")
    return target


def check_url(settings: Settings, url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SourceError(f"only https URLs are accepted, got {url!r}")
    host = (parsed.hostname or "").lower()
    allowed = settings.docs_allowed_host_list
    if not any(host == a or host.endswith("." + a) for a in allowed):
        raise SourceError(f"host {host!r} is not in DOCS_ALLOWED_HOSTS ({', '.join(allowed)})")


def resolve_target(
    settings: Settings,
    source: str,
    source_type: str | None = None,
    framework: str | None = None,
    version: str | None = None,
    title: str | None = None,
) -> list[DocTarget]:
    """Everything a single ingest request expands to, validated up front.

    A directory expands to every book in it, one target per file, preferring
    EPUB where the same title also exists as a PDF — EPUB keeps headings and
    code listings as markup, while PDF text extraction loses both.
    """
    framework = framework.lower() if framework else None
    preset = PRESETS.get(source.lower())
    if preset is not None:
        return [preset]

    if source.startswith(("http://", "https://")):
        check_url(settings, source)
        kind = source_type or detect_type(source)
        if kind not in SOURCE_TYPES:
            raise SourceError(f"cannot tell the type of {source!r}; pass source_type")
        target = DocTarget(
            source_url=source,
            source_type=kind,
            title=title or source.rstrip("/").rsplit("/", 1)[-1],
            framework=framework,
            version=version,
        )
        if kind == "github" and _GITHUB_TREE_RE.match(source) is None:
            raise SourceError(
                "github sources look like https://github.com/<owner>/<repo>/tree/<ref>/<docs dir>"
            )
        return [target]

    path = resolve_local(settings, source)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    books: dict[str, Path] = {}
    for file in files:
        kind = source_type if path.is_file() and source_type else detect_type(file.name)
        if kind not in ("epub", "pdf", "markdown"):
            continue
        key = str(file.with_suffix("")).lower()
        if key in books and books[key].suffix.lower() == ".epub":
            continue
        books[key] = file
    if not books:
        raise SourceError(f"no EPUB, PDF, or Markdown files found at {source!r}")
    root = settings.docs_root.resolve()
    return [
        DocTarget(
            source_url=f"file://{file.relative_to(root).as_posix()}",
            source_type=source_type if path.is_file() and source_type else detect_type(file.name),
            title=title if path.is_file() and title else file.stem,
            framework=framework,
            version=version,
            local_path=file,
        )
        for file in books.values()
    ]


def github_docs_path(target: DocTarget) -> str:
    match = _GITHUB_TREE_RE.match(target.source_url)
    if match is None:
        raise SourceError(f"not a GitHub tree URL: {target.source_url!r}")
    return match.group("path")


def download_url(target: DocTarget) -> str:
    if target.source_type == "github":
        match = _GITHUB_TREE_RE.match(target.source_url)
        assert match is not None
        return (
            f"https://codeload.github.com/{match.group('owner')}/{match.group('repo')}"
            f"/tar.gz/{match.group('ref')}"
        )
    return target.source_url


async def fetch(settings: Settings, target: DocTarget) -> bytes:
    """Read a local file, or download with the cache as a fallback.

    Downloads are cached under the data root keyed by URL, so a force re-index
    does not re-download a 13MB archive.
    """
    if target.local_path is not None:
        return target.local_path.read_bytes()

    url = download_url(target)
    cache = settings.docs_root / "cache" / (hashlib.sha256(url.encode()).hexdigest()[:24])
    if cache.exists():
        return cache.read_bytes()

    async def guard(request: httpx.Request) -> None:
        check_url(settings, str(request.url))

    async with httpx.AsyncClient(
        follow_redirects=True, timeout=120, event_hooks={"request": [guard]}
    ) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise SourceError(f"{url} exceeds {MAX_DOWNLOAD_BYTES // 2**20}MB")
                chunks.append(chunk)
    data = b"".join(chunks)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    except OSError:
        pass  # a read-only data root only costs the cache
    return data
