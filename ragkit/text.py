"""Small text helpers shared by the keyword index and the hashing embedder."""

import re

_WORD = re.compile(r"[a-z0-9]+")

# A short stop list: enough to keep BM25 and the hashing embedder from being
# dominated by glue words, short enough to read at a glance.
STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from has have how i if in
    is it its me my of on or our so than that the their them then there these
    this to was we were what when where which who why will with you your""".split()
)


def stem(token: str) -> str:
    """A deliberately tiny suffix stripper, so "refunds" and "refund" match.

    A real system would use a proper stemmer or lean on the embedding model;
    this keeps the project dependency-free.
    """
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3 and not token.endswith("ss"):
            return token[: -len(suffix)] + repl
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, stemmed word tokens with stop words removed."""
    return [stem(t) for t in _WORD.findall(text.lower()) if t not in STOPWORDS]
