from __future__ import annotations

from functools import lru_cache
from string import punctuation

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

_PUNCT_TRANSLATION = str.maketrans("", "", punctuation)
_FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


@lru_cache(maxsize=1)
def _english_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return _FALLBACK_STOPWORDS


def clean_text(text: str) -> str:
    if not text:
        return ""

    normalized_text = text.lower().translate(_PUNCT_TRANSLATION)
    tokens = wordpunct_tokenize(normalized_text)
    stop_words = _english_stopwords()

    cleaned_tokens = [token for token in tokens if token and token not in stop_words and token.isalnum()]
    return " ".join(cleaned_tokens)