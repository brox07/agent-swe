"""Tree-sitter AST chunker for Python and TypeScript.

Chunking is *nested*: a class yields a chunk for the whole class and a further
chunk per method, each carrying its parent in the payload. That is the only way
both "the auth service" and "how does token refresh work" retrieve well. The
cost is overlapping hits, which the retrieval layer collapses.

Closures (a function defined inside another function) are deliberately not
emitted separately; they travel inside their enclosing chunk. Classes are always
recursed into, at any nesting depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import tree_sitter_python as ts_python
import tree_sitter_typescript as ts_typescript
from tree_sitter import Language, Node, Parser

from src.parser.base import CodeChunk, split_oversized

# tree-sitter 0.26: a Language wraps the grammar pointer and Parser takes it
# positionally. This idiom changed across 0.22-0.24, hence the pinned floors.
_PY_LANGUAGE = Language(ts_python.language())
_TS_LANGUAGE = Language(ts_typescript.language_typescript())
_TSX_LANGUAGE = Language(ts_typescript.language_tsx())


@dataclass(frozen=True)
class _Spec:
    """Which node types matter for a language, and how to classify them."""

    classes: frozenset[str]
    functions: frozenset[str]
    methods: frozenset[str]
    # Wrappers that should contribute their own start position (so decorators
    # and `export` keywords are included) but delegate identity to an inner node.
    wrappers: dict[str, str]
    interfaces: frozenset[str] = frozenset()


_PYTHON_SPEC = _Spec(
    classes=frozenset({"class_definition"}),
    functions=frozenset({"function_definition"}),
    methods=frozenset({"function_definition"}),
    wrappers={"decorated_definition": "definition"},
)

_TS_SPEC = _Spec(
    classes=frozenset({"class_declaration", "abstract_class_declaration"}),
    functions=frozenset({"function_declaration", "generator_function_declaration"}),
    methods=frozenset({"method_definition"}),
    wrappers={"export_statement": "declaration"},
    interfaces=frozenset({"interface_declaration"}),
)


def _language_for(language: str, file_path: str) -> tuple[Language, _Spec]:
    if language == "python":
        return _PY_LANGUAGE, _PYTHON_SPEC
    if file_path.endswith(".tsx"):
        return _TSX_LANGUAGE, _TS_SPEC
    return _TS_LANGUAGE, _TS_SPEC


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _identifier(node: Node, source: bytes) -> str | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return _text(name, source)
    return None


def _arrow_identifier(node: Node, source: bytes) -> str | None:
    """Name of an exported/assigned arrow function, e.g. `const foo = () => {}`.

    The spec calls these out explicitly for TypeScript; they carry no `name`
    field, so the declarator has to be inspected.
    """
    if node.type != "lexical_declaration":
        return None
    for child in node.named_children:
        if child.type != "variable_declarator":
            continue
        value = child.child_by_field_name("value")
        if value is not None and value.type in ("arrow_function", "function_expression"):
            name = child.child_by_field_name("name")
            if name is not None:
                return _text(name, source)
    return None


def parse_code(
    source_text: str,
    language: str,
    file_path: str,
    max_chunk_chars: int,
) -> list[CodeChunk]:
    """Extract nested class/function/method chunks from a source file."""
    lang, spec = _language_for(language, file_path)
    parser = Parser(lang)
    source = source_text.encode("utf-8")
    tree = parser.parse(source)

    chunks: list[CodeChunk] = []

    def emit(
        node: Node,
        node_type: str,
        identifier: str,
        scope: list[str],
        start_node: Node,
    ) -> None:
        path = ".".join([*scope, identifier])
        chunk = CodeChunk(
            language=language,
            node_type=node_type,
            identifier=identifier,
            node_path=path,
            parent_identifier=scope[-1] if scope else None,
            # tree-sitter points are 0-indexed; editors and humans are 1-indexed.
            start_line=start_node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            content=_text(start_node, source),
        )
        chunks.extend(split_oversized(chunk, max_chunk_chars))

    def walk(node: Node, scope: list[str], in_function: bool, wrapper: Node | None = None) -> None:
        # Unwrap decorated definitions and export statements, keeping the outer
        # node's span so decorators and `export` are part of the indexed text.
        if node.type in spec.wrappers:
            inner = node.child_by_field_name(spec.wrappers[node.type])
            if inner is not None:
                walk(inner, scope, in_function, wrapper=node)
                return

        span_node = wrapper if wrapper is not None else node
        identifier = _identifier(node, source) or _arrow_identifier(node, source)

        if node.type in spec.classes and identifier:
            emit(node, "class", identifier, scope, span_node)
            body = node.child_by_field_name("body")
            if body is not None:
                for child in body.named_children:
                    walk(child, [*scope, identifier], in_function=False)
            return

        if node.type in spec.interfaces and identifier:
            emit(node, "interface", identifier, scope, span_node)
            return

        is_method = node.type in spec.methods and bool(scope)
        is_function = node.type in spec.functions or _arrow_identifier(node, source) is not None

        if identifier and (is_method or is_function):
            # Closures stay inside their enclosing chunk rather than competing
            # with it as a separate result.
            if not in_function:
                emit(node, "method" if is_method else "function", identifier, scope, span_node)
            return

        for child in node.named_children:
            walk(child, scope, in_function)

    for child in tree.root_node.named_children:
        walk(child, [], in_function=False)
    return chunks
