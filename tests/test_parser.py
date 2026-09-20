"""Parser behaviour: nested chunking, span boundaries, oversize handling, fallbacks."""

from __future__ import annotations

from pathlib import Path

from src.parser.base import CodeChunk, detect_language, split_oversized
from src.parser.generic_chunker import parse_generic
from src.parser.tree_sitter_ast import parse_code

MAX = 24_000

PYTHON_SOURCE = '''import os


def module_function(a, b):
    """Adds two things."""

    def closure():
        return 1

    return a + b + closure()


class AuthService:
    """Docstring."""

    def __init__(self, secret):
        self.secret = secret

    @property
    def ready(self):
        return bool(self.secret)

    @staticmethod
    def validate_token(token):
        return len(token) > 10

    class Inner:
        def deep(self):
            return 2
'''

TS_SOURCE = """export interface User { id: string; }

export function makeUser(id: string): User {
  return { id };
}

export const handler = async (req: Request) => {
  return new Response("ok");
};

export class TokenStore {
  get(key: string): string | undefined {
    return undefined;
  }

  async refresh(key: string): Promise<void> {}
}
"""


def _by_path(chunks: list[CodeChunk]) -> dict[str, CodeChunk]:
    return {c.node_path: c for c in chunks}


class TestPythonChunking:
    def test_emits_class_and_each_method(self):
        chunks = _by_path(parse_code(PYTHON_SOURCE, "python", "a.py", MAX))
        assert "AuthService" in chunks
        assert "AuthService.__init__" in chunks
        assert "AuthService.validate_token" in chunks
        # The class chunk covers the whole class, methods are separate points.
        assert chunks["AuthService"].node_type == "class"
        assert chunks["AuthService.validate_token"].node_type == "method"
        assert chunks["AuthService.validate_token"].parent_identifier == "AuthService"

    def test_closures_stay_inside_their_parent(self):
        """A nested function should not compete as a separate result."""
        chunks = _by_path(parse_code(PYTHON_SOURCE, "python", "a.py", MAX))
        assert "module_function" in chunks
        assert "module_function.closure" not in chunks
        assert "def closure()" in chunks["module_function"].content

    def test_decorators_are_part_of_the_chunk(self):
        chunks = _by_path(parse_code(PYTHON_SOURCE, "python", "a.py", MAX))
        assert chunks["AuthService.ready"].content.startswith("@property")
        assert chunks["AuthService.validate_token"].content.startswith("@staticmethod")

    def test_nested_classes_are_recursed(self):
        chunks = _by_path(parse_code(PYTHON_SOURCE, "python", "a.py", MAX))
        assert chunks["AuthService.Inner"].node_type == "class"
        assert chunks["AuthService.Inner.deep"].parent_identifier == "Inner"

    def test_line_numbers_are_one_indexed_and_bracket_the_node(self):
        chunks = _by_path(parse_code(PYTHON_SOURCE, "python", "a.py", MAX))
        lines = PYTHON_SOURCE.splitlines()
        chunk = chunks["AuthService.validate_token"]
        assert lines[chunk.start_line - 1].strip() == "@staticmethod"
        assert lines[chunk.end_line - 1].strip() == "return len(token) > 10"


class TestTypeScriptChunking:
    def test_extracts_declared_shapes(self):
        chunks = _by_path(parse_code(TS_SOURCE, "typescript", "a.ts", MAX))
        assert chunks["User"].node_type == "interface"
        assert chunks["makeUser"].node_type == "function"
        assert chunks["TokenStore"].node_type == "class"
        assert chunks["TokenStore.refresh"].node_type == "method"

    def test_exported_arrow_functions_are_found(self):
        """Called out explicitly in the spec; they carry no `name` field."""
        chunks = _by_path(parse_code(TS_SOURCE, "typescript", "a.ts", MAX))
        assert "handler" in chunks
        assert chunks["handler"].node_type == "function"

    def test_export_keyword_is_included_in_the_span(self):
        chunks = _by_path(parse_code(TS_SOURCE, "typescript", "a.ts", MAX))
        assert chunks["makeUser"].content.startswith("export function")

    def test_tsx_uses_the_tsx_grammar(self):
        source = "export const View = () => <div className='x'>hi</div>;\n"
        chunks = _by_path(parse_code(source, "typescript", "view.tsx", MAX))
        assert "View" in chunks


class TestOversizedNodes:
    """The spec's 512-token model silently truncated large classes. It must not."""

    def test_large_node_is_split_not_truncated(self):
        body = "".join(f"    x{i} = {i}\n" for i in range(4000))
        source = f"class Big:\n{body}"
        chunks = parse_code(source, "python", "big.py", 2_000)
        parts = [c for c in chunks if c.node_path == "Big"]
        assert len(parts) > 1
        assert not any(c.is_truncated for c in parts)
        # Every original line survives somewhere.
        joined = "".join(c.content for c in parts)
        assert "x0 = 0" in joined
        assert "x3999 = 3999" in joined

    def test_parts_are_ordered_and_labelled(self):
        chunk = CodeChunk(
            language="python",
            node_type="class",
            identifier="Big",
            node_path="Big",
            start_line=1,
            end_line=3,
            content="def f():\n" + "".join(f"    y{i}\n" for i in range(500)),
        )
        parts = split_oversized(chunk, 500)
        assert [p.chunk_index for p in parts] == list(range(len(parts)))
        assert all(p.chunk_total == len(parts) for p in parts)
        # Continuation parts carry the signature so the fragment still has context.
        assert "part 2/" in parts[1].content
        assert "def f():" in parts[1].content

    def test_truncation_flag_only_when_content_is_actually_lost(self):
        """A single line longer than the whole budget is the one lossy case."""
        chunk = CodeChunk(
            language="typescript",
            node_type="function",
            identifier="min",
            node_path="min",
            start_line=1,
            end_line=1,
            content="const a=" + "z" * 5_000 + ";\n",
        )
        parts = split_oversized(chunk, 1_000)
        assert any(p.is_truncated for p in parts)

    def test_small_node_is_untouched(self):
        chunks = parse_code("def f():\n    return 1\n", "python", "a.py", MAX)
        assert len(chunks) == 1
        assert chunks[0].chunk_total == 1
        assert chunks[0].is_truncated is False


class TestGenericChunking:
    def test_compose_services_become_separate_chunks(self):
        compose = (
            'version: "3"\n\n'
            "services:\n"
            "  api:\n    build: .\n"
            "  postgres:\n    image: postgres:16-alpine\n\n"
            "volumes:\n  pg_data:\n"
        )
        paths = {c.node_path: c for c in parse_generic(compose, "yaml", MAX)}
        assert "services.api" in paths
        assert "services.postgres" in paths
        assert paths["services.postgres"].parent_identifier == "services"
        assert "postgres:16-alpine" in paths["services.postgres"].content

    def test_dockerfile_splits_on_build_stages(self):
        dockerfile = (
            "ARG PY=3.11\n\n"
            "FROM python:${PY}-slim AS builder\nRUN pip install uv\n\n"
            'FROM python:${PY}-slim\nCMD ["uvicorn"]\n'
        )
        paths = {c.node_path for c in parse_generic(dockerfile, "dockerfile", MAX)}
        assert "stage.builder" in paths
        assert "preamble" in paths
        assert any(p.startswith("stage.stage-") for p in paths)


class TestLanguageDetection:
    def test_known_and_unknown_extensions(self):
        assert detect_language(Path("a.py")) == "python"
        assert detect_language(Path("a.ts")) == "typescript"
        assert detect_language(Path("a.tsx")) == "typescript"
        assert detect_language(Path("docker-compose.yml")) == "yaml"
        assert detect_language(Path("Dockerfile")) == "dockerfile"
        assert detect_language(Path("Dockerfile.prod")) == "dockerfile"
        assert detect_language(Path("README.md")) is None
        assert detect_language(Path("logo.png")) is None
