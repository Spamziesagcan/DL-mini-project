from __future__ import annotations

import re
from typing import Sequence

import numpy as np


class Tokenizer:
    def __init__(self, num_words: int | None = None, oov_token: str = "<unk>") -> None:
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index: dict[str, int] = {}

    def fit_on_texts(self, texts: Sequence[str]) -> None:
        frequency: dict[str, int] = {}
        for text in texts:
            for token in _tokenize(text):
                frequency[token] = frequency.get(token, 0) + 1

        self.word_index = {}
        if self.oov_token:
            self.word_index[self.oov_token] = 1

        sorted_tokens = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
        next_index = 2 if self.oov_token else 1
        for token, _ in sorted_tokens:
            if self.num_words is not None and next_index > self.num_words:
                break
            if token == self.oov_token:
                continue
            self.word_index[token] = next_index
            next_index += 1

    def texts_to_sequences(self, texts: Sequence[str]) -> list[list[int]]:
        sequences: list[list[int]] = []
        oov_index = self.word_index.get(self.oov_token, 1 if self.oov_token else 0)
        max_index = self.num_words if self.num_words is not None else None

        for text in texts:
            sequence: list[int] = []
            for token in _tokenize(text):
                index = self.word_index.get(token, oov_index)
                if max_index is not None and index > max_index:
                    index = oov_index
                sequence.append(index)
            sequences.append(sequence)
        return sequences


def _tokenize(text: str) -> list[str]:
    normalized_text = (text or "").strip().lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", normalized_text)


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    maxlen: int,
    padding: str = "pre",
    truncating: str = "pre",
) -> np.ndarray:
    padded = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for row_index, sequence in enumerate(sequences):
        values = list(sequence)
        if len(values) > maxlen:
            if truncating == "post":
                values = values[:maxlen]
            else:
                values = values[-maxlen:]

        if padding == "post":
            padded[row_index, : len(values)] = values
        else:
            padded[row_index, maxlen - len(values) :] = values
    return padded


def fit_tokenizer(texts: Sequence[str], num_words: int | None = None) -> Tokenizer:
    tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>")
    normalized_texts = [(text or "").strip() for text in texts]
    tokenizer.fit_on_texts(normalized_texts)
    return tokenizer


def texts_to_padded_sequences(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    max_length: int = 200,
) -> np.ndarray:
    normalized_texts = [(text or "").strip() for text in texts]
    sequences = tokenizer.texts_to_sequences(normalized_texts)
    return pad_sequences(
        sequences,
        maxlen=max_length,
        padding="post",
        truncating="post",
    )


def create_tokenizer_pipeline(
    resume_text: str,
    jd_text: str,
    max_length: int = 200,
) -> tuple[tuple[np.ndarray, np.ndarray], Tokenizer]:
    """Fit a Keras tokenizer on two texts and return padded sequences."""
    tokenizer = fit_tokenizer([resume_text, jd_text])
    padded_sequences = texts_to_padded_sequences(tokenizer, [resume_text, jd_text], max_length=max_length)

    return (padded_sequences[0], padded_sequences[1]), tokenizer