from __future__ import annotations

from tensorflow.keras.layers import Conv1D, Dense, Embedding, GlobalMaxPooling1D, Input
from tensorflow.keras.models import Model


def build_cnn_text_classification_model(
    vocab_size: int,
    max_length: int = 200,
    embedding_dim: int = 64,
) -> Model:
    inputs = Input(shape=(max_length,), name="text_input")
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs)
    x = Conv1D(filters=64, kernel_size=5, activation="relu")(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid", name="class_probability")(x)

    model = Model(inputs=inputs, outputs=outputs, name="cnn_text_classification_model")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model