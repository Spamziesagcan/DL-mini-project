from __future__ import annotations

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def create_tokenizer_pipeline(
    resume_text: str,
    jd_text: str,
    max_length: int = 200,
) -> tuple[tuple[np.ndarray, np.ndarray], Tokenizer]:
    """Fit a Keras tokenizer on two texts and return padded sequences."""
    tokenizer = Tokenizer()
    texts = [(resume_text or "").strip(), (jd_text or "").strip()]

    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(
        sequences,
        maxlen=max_length,
        padding="post",
        truncating="post",
    )

    return (padded_sequences[0], padded_sequences[1]), tokenizer