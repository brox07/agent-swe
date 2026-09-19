"""Shared documentation chunking: size discipline and code-block integrity."""

from __future__ import annotations

from src.docs.sections import DocChunk, DocSection, chunk_sections, split_blocks


def test_a_fenced_code_block_is_one_block_even_with_blank_lines():
    text = "Intro.\n\n```python\ndef f():\n\n    return 1\n```\n\nOutro."
    blocks = split_blocks(text)
    assert blocks == ["Intro.", "```python\ndef f():\n\n    return 1\n```", "Outro."]


def test_small_sections_become_one_chunk_each():
    sections = [
        DocSection(["asyncio", "Streams"], "Streams are high-level.", "a.html#s"),
        DocSection(["asyncio", "Queues"], "Queues are FIFO.", "a.html#q"),
    ]
    chunks = chunk_sections(sections, target_chars=500, max_chars=2000)
    assert [c.heading_path for c in chunks] == [["asyncio", "Streams"], ["asyncio", "Queues"]]
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert chunks[0].location == "a.html#s"


def test_a_long_section_packs_paragraphs_up_to_the_target():
    para = "word " * 40  # ~200 chars
    text = "\n\n".join([para.strip()] * 10)
    chunks = chunk_sections([DocSection(["S"], text)], target_chars=500, max_chars=2000)
    assert len(chunks) > 1
    assert all(len(c.content) <= 500 for c in chunks)
    assert [c.part for c in chunks] == list(range(len(chunks)))
    assert all(c.parts == len(chunks) for c in chunks)


def test_code_is_never_split_below_the_hard_limit():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(60)) + "\n```"
    text = "Before.\n\n" + code + "\n\nAfter."
    chunks = chunk_sections([DocSection(["S"], text)], target_chars=200, max_chars=5000)
    assert any(c.content == code for c in chunks)


def test_a_block_beyond_the_hard_limit_is_split_rather_than_dropped():
    text = "\n".join(f"line {i} of a very long listing" for i in range(400))
    chunks = chunk_sections([DocSection(["S"], text)], target_chars=1000, max_chars=1000)
    assert all(len(c.content) <= 1000 for c in chunks)
    assert "line 399" in chunks[-1].content


def test_empty_sections_are_skipped():
    chunks = chunk_sections(
        [DocSection(["Empty"], "  \n\n "), DocSection(["Full"], "Text.")],
        target_chars=500,
        max_chars=2000,
    )
    assert [c.heading_path for c in chunks] == [["Full"]]


def test_the_breadcrumb_is_embedded_with_the_body():
    chunk = DocChunk(heading_path=["asyncio", "Examples"], location="", content="body")
    assert chunk.embed_text("Python 3.14") == "Python 3.14 > asyncio > Examples\n\nbody"
