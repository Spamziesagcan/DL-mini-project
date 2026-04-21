from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from utils.keras_backend import configure_keras_backend

configure_keras_backend()

from keras.models import Model

from preprocessing.text_cleaning import clean_text
from preprocessing.tokenizer_pipeline import Tokenizer, fit_tokenizer, texts_to_padded_sequences
from utils.skill_extraction import ROLE_SKILL_HINTS
from utils.synthetic_data import generate_synthetic_training_data

from .cnn_text_classification import build_cnn_similarity_model
from .lstm_similarity import build_lstm_similarity_model


MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 64
TRAINING_EPOCHS = 8
TRAINING_BATCH_SIZE = 8
NEGATIVE_RATIO = 1.5


@dataclass(frozen=True)
class SimilarityArtifacts:
    tokenizer: Tokenizer
    lstm_model: Model
    cnn_model: Model


_BASE_ROLE_PROFILES: dict[str, tuple[str, ...]] = {
    "web developer": ("html", "css", "javascript", "react", "frontend", "git"),
    "frontend developer": ("html", "css", "javascript", "react", "typescript", "frontend"),
    "backend developer": ("python", "sql", "django", "flask", "backend", "docker"),
    "full stack developer": ("html", "css", "javascript", "react", "node.js", "python", "sql"),
    "data scientist": ("python", "pandas", "numpy", "scikit-learn", "machine learning", "tableau"),
    "machine learning engineer": ("python", "tensorflow", "keras", "pytorch", "machine learning", "docker"),
    "data analyst": ("sql", "excel", "pandas", "tableau", "statistics", "python"),
    "devops engineer": ("docker", "git", "linux", "aws", "spark", "hadoop"),
}

_ROLE_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "experienced {role} with hands-on work in {skills}. built products, automated workflows, and collaborated with cross-functional teams.",
        "hiring {role} with strong experience in {skills}. the role needs clean code, reliable delivery, and clear communication.",
    ),
    (
        "{role} professional focused on {skills}. delivered production-ready solutions and improved user experience.",
        "seeking {role} who can apply {skills} to solve real business problems and work closely with product teams.",
    ),
    (
        "results-driven {role} using {skills} to ship software, analyze feedback, and improve reliability.",
        "looking for {role} who brings {skills} and can contribute to architecture, implementation, and testing.",
    ),
)


def _role_skill_profiles() -> dict[str, tuple[str, ...]]:
    profiles = dict(_BASE_ROLE_PROFILES)
    for role, skills in ROLE_SKILL_HINTS.items():
        profiles.setdefault(role, tuple(skills))
    return profiles


def _build_training_texts() -> tuple[list[str], list[str]]:
    resume_texts: list[str] = []
    jd_texts: list[str] = []

    for role, skills in _role_skill_profiles().items():
        skill_phrase = ", ".join(skills)
        for resume_template, jd_template in _ROLE_TEMPLATES:
            resume_texts.append(clean_text(resume_template.format(role=role, skills=skill_phrase)))
            jd_texts.append(clean_text(jd_template.format(role=role, skills=skill_phrase)))

    return resume_texts, jd_texts


def _prepare_sequence_pairs(
    tokenizer: Tokenizer,
    text_pairs: Sequence[tuple[str, str]],
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    resume_texts = [clean_text(resume_text) for resume_text, _ in text_pairs]
    jd_texts = [clean_text(jd_text) for _, jd_text in text_pairs]
    resume_sequences = texts_to_padded_sequences(tokenizer, resume_texts, max_length=max_length)
    jd_sequences = texts_to_padded_sequences(tokenizer, jd_texts, max_length=max_length)
    return resume_sequences.astype("int32"), jd_sequences.astype("int32")


@lru_cache(maxsize=1)
def get_similarity_artifacts(
    max_length: int = MAX_SEQUENCE_LENGTH,
    embedding_dim: int = EMBEDDING_DIM,
) -> SimilarityArtifacts:
    resume_texts, jd_texts = _build_training_texts()
    training_pairs, labels = generate_synthetic_training_data(
        resume_texts,
        jd_texts,
        negative_ratio=NEGATIVE_RATIO,
        random_state=42,
    )

    if not training_pairs or not labels:
        raise RuntimeError("Unable to build synthetic training data for DL similarity models.")

    tokenizer = fit_tokenizer([text for pair in training_pairs for text in pair])
    resume_sequences, jd_sequences = _prepare_sequence_pairs(tokenizer, training_pairs, max_length)
    targets = np.asarray(labels, dtype="float32")

    permutation = np.random.default_rng(42).permutation(len(targets))
    resume_sequences = resume_sequences[permutation]
    jd_sequences = jd_sequences[permutation]
    targets = targets[permutation]

    vocab_size = max(len(tokenizer.word_index) + 1, 2)
    lstm_model = build_lstm_similarity_model(
        vocab_size=vocab_size,
        max_length=max_length,
        embedding_dim=embedding_dim,
    )
    cnn_model = build_cnn_similarity_model(
        vocab_size=vocab_size,
        max_length=max_length,
        embedding_dim=embedding_dim,
    )

    batch_size = min(TRAINING_BATCH_SIZE, len(targets))
    lstm_model.fit(
        [resume_sequences, jd_sequences],
        targets,
        epochs=TRAINING_EPOCHS,
        batch_size=batch_size,
        verbose=0,
        shuffle=True,
    )
    cnn_model.fit(
        [resume_sequences, jd_sequences],
        targets,
        epochs=TRAINING_EPOCHS,
        batch_size=batch_size,
        verbose=0,
        shuffle=True,
    )

    return SimilarityArtifacts(tokenizer=tokenizer, lstm_model=lstm_model, cnn_model=cnn_model)


def predict_similarity_scores(
    resume_text: str,
    jd_text: str,
    max_length: int = MAX_SEQUENCE_LENGTH,
    embedding_dim: int = EMBEDDING_DIM,
) -> dict[str, float]:
    artifacts = get_similarity_artifacts(max_length=max_length, embedding_dim=embedding_dim)
    resume_sequences, jd_sequences = _prepare_sequence_pairs(
        artifacts.tokenizer,
        [(resume_text, jd_text)],
        max_length,
    )

    lstm_score = float(artifacts.lstm_model.predict([resume_sequences, jd_sequences], verbose=0)[0][0])
    cnn_score = float(artifacts.cnn_model.predict([resume_sequences, jd_sequences], verbose=0)[0][0])

    return {
        "lstm_score": max(0.0, min(1.0, lstm_score)),
        "cnn_score": max(0.0, min(1.0, cnn_score)),
    }