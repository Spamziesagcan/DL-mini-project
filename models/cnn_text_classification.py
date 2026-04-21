from __future__ import annotations

from utils.keras_backend import configure_keras_backend

configure_keras_backend()

from keras.layers import Concatenate, Conv1D, Dense, Embedding, GlobalMaxPooling1D, Input
from keras.models import Model


def _build_cnn_text_encoder(max_length: int, vocab_size: int, embedding_dim: int) -> Model:
    text_input = Input(shape=(max_length,), name="text_input")
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)
    x = Conv1D(filters=64, kernel_size=5, activation="relu", padding="same")(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation="relu")(x)
    return Model(text_input, x, name="cnn_text_encoder")


def build_cnn_similarity_model(
    vocab_size: int,
    max_length: int = 200,
    embedding_dim: int = 64,
) -> Model:
    resume_input = Input(shape=(max_length,), name="resume_input")
    jd_input = Input(shape=(max_length,), name="jd_input")

    text_encoder = _build_cnn_text_encoder(max_length, vocab_size, embedding_dim)
    resume_vector = text_encoder(resume_input)
    jd_vector = text_encoder(jd_input)

    merged = Concatenate()([resume_vector, jd_vector])
    hidden = Dense(64, activation="relu")(merged)
    outputs = Dense(1, activation="sigmoid", name="similarity_score")(hidden)

    model = Model(inputs=[resume_input, jd_input], outputs=outputs, name="cnn_similarity_model")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_cnn_text_classification_model(
    vocab_size: int,
    max_length: int = 200,
    embedding_dim: int = 64,
) -> Model:
    return build_cnn_similarity_model(vocab_size=vocab_size, max_length=max_length, embedding_dim=embedding_dim)