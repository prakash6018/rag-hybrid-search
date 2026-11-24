from ragkit.chunking import chunk_markdown, content_hash

DOC = """# Returns
## Refunds
Refunds take 5 to 7 days.
## Exchanges
Return the item and place a new order.
"""


def test_chunks_keep_their_heading_path():
    chunks = chunk_markdown("returns", DOC)
    assert [c.heading_path for c in chunks] == [("Returns", "Refunds"), ("Returns", "Exchanges")]
    assert chunks[0].content.startswith("Returns > Refunds: Refunds take")


def test_long_sections_split_with_overlap():
    body = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_markdown("d", f"# T\n{body}", max_words=100, overlap_words=20)
    assert len(chunks) == 3
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]          # the overlap
    assert chunks[-1].text.split()[-1] == "w249"  # nothing dropped at the end


def test_overlap_must_be_smaller_than_window():
    import pytest

    with pytest.raises(ValueError):
        chunk_markdown("d", "# T\nx", max_words=10, overlap_words=10)


def test_content_hash_is_stable():
    assert content_hash("abc") == content_hash("abc") != content_hash("abd")
