from __future__ import annotations

from tensorflow.keras.layers import Concatenate, Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model


def _build_text_encoder(max_length: int, vocab_size: int, embedding_dim: int) -> Model:
    text_input = Input(shape=(max_length,), name="text_input")
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(text_input)
    x = LSTM(64)(x)
    return Model(text_input, x, name="text_encoder")


def build_lstm_similarity_model(
    vocab_size: int,
    max_length: int = 200,
    embedding_dim: int = 64,
) -> Model:
    resume_input = Input(shape=(max_length,), name="resume_input")
    jd_input = Input(shape=(max_length,), name="jd_input")

    text_encoder = _build_text_encoder(max_length, vocab_size, embedding_dim)
    resume_vector = text_encoder(resume_input)
    jd_vector = text_encoder(jd_input)

    merged = Concatenate()([resume_vector, jd_vector])
    hidden = Dense(64, activation="relu")(merged)
    output = Dense(1, activation="sigmoid", name="similarity_score")(hidden)

    model = Model(inputs=[resume_input, jd_input], outputs=output, name="lstm_similarity_model")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model