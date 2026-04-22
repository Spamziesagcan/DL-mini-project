from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from utils.keras_backend import configure_keras_backend

configure_keras_backend()

from keras.models import Model, load_model

from preprocessing.text_cleaning import clean_text
from preprocessing.tokenizer_pipeline import Tokenizer, fit_tokenizer, texts_to_padded_sequences
from utils.skill_extraction import ROLE_SKILL_HINTS
from utils.synthetic_data import generate_synthetic_training_data

from .cnn_text_classification import build_cnn_similarity_model
from .lstm_similarity import build_lstm_similarity_model


MAX_SEQUENCE_LENGTH = 60
EMBEDDING_DIM = 16
TRAINING_EPOCHS = 1
TRAINING_BATCH_SIZE = 16
NEGATIVE_RATIO = 1.0
BOOTSTRAP_ROLE_LIMIT = 6

ARTIFACTS_DIR = Path("artifacts")
RUNTIME_TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.pkl"
RUNTIME_CNN_MODEL_PATH = ARTIFACTS_DIR / "cnn_similarity_model.h5"
RUNTIME_LSTM_MODEL_PATH = ARTIFACTS_DIR / "lstm_similarity_model.h5"
LEGACY_CNN_MODEL_PATH = Path("cnn_expanded_model.h5")
LEGACY_CNN_TOKENIZER_PATH = Path("tokenizer.pkl")
LEGACY_LSTM_MODEL_PATH = Path("lstm_expanded_model.h5")
LEGACY_LSTM_TOKENIZER_PATH = Path("lstm_tokenizer.pkl")


@dataclass(frozen=True)
class SimilarityArtifacts:
    cnn_tokenizer: Any
    lstm_tokenizer: Any
    cnn_model: Model
    lstm_model: Model


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
)


def _role_skill_profiles() -> dict[str, tuple[str, ...]]:
    profiles = dict(_BASE_ROLE_PROFILES)
    for role, skills in ROLE_SKILL_HINTS.items():
        profiles.setdefault(role, tuple(skills))
    return profiles


def _build_training_texts() -> tuple[list[str], list[str]]:
    resume_texts: list[str] = []
    jd_texts: list[str] = []

    role_profiles = list(_role_skill_profiles().items())[:BOOTSTRAP_ROLE_LIMIT]
    for role, skills in role_profiles:
        skill_phrase = ", ".join(skills)
        for resume_template, jd_template in _ROLE_TEMPLATES:
            resume_texts.append(clean_text(resume_template.format(role=role, skills=skill_phrase)))
            jd_texts.append(clean_text(jd_template.format(role=role, skills=skill_phrase)))

    return resume_texts, jd_texts


def _pickle_load(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _pickle_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(value, file)


def _keras_texts_to_padded_sequences(
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences([(text or "").strip() for text in texts])
    padded = np.zeros((len(sequences), max_length), dtype=np.int32)
    for row_index, sequence in enumerate(sequences):
        values = list(sequence)[:max_length]
        padded[row_index, : len(values)] = values
    return padded


def _to_padded_sequences(
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
) -> np.ndarray:
    if isinstance(tokenizer, Tokenizer):
        return texts_to_padded_sequences(tokenizer, texts, max_length=max_length).astype("int32")
    return _keras_texts_to_padded_sequences(tokenizer, texts, max_length=max_length)


def _prepare_sequence_pairs(
    tokenizer: Any,
    text_pairs: Sequence[tuple[str, str]],
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    resume_texts = [clean_text(resume_text) for resume_text, _ in text_pairs]
    jd_texts = [clean_text(jd_text) for _, jd_text in text_pairs]
    resume_sequences = _to_padded_sequences(tokenizer, resume_texts, max_length)
    jd_sequences = _to_padded_sequences(tokenizer, jd_texts, max_length)
    return resume_sequences.astype("int32"), jd_sequences.astype("int32")


def _load_runtime_artifacts() -> SimilarityArtifacts | None:
    if not (RUNTIME_TOKENIZER_PATH.exists() and RUNTIME_CNN_MODEL_PATH.exists() and RUNTIME_LSTM_MODEL_PATH.exists()):
        return None

    tokenizer = _pickle_load(RUNTIME_TOKENIZER_PATH)
    cnn_model = load_model(RUNTIME_CNN_MODEL_PATH, compile=False)
    lstm_model = load_model(RUNTIME_LSTM_MODEL_PATH, compile=False)
    return SimilarityArtifacts(
        cnn_tokenizer=tokenizer,
        lstm_tokenizer=tokenizer,
        cnn_model=cnn_model,
        lstm_model=lstm_model,
    )


def _load_legacy_artifacts() -> SimilarityArtifacts | None:
    if not (LEGACY_CNN_MODEL_PATH.exists() and LEGACY_CNN_TOKENIZER_PATH.exists()):
        return None

    cnn_model = load_model(LEGACY_CNN_MODEL_PATH, compile=False)
    cnn_tokenizer = _pickle_load(LEGACY_CNN_TOKENIZER_PATH)

    if RUNTIME_LSTM_MODEL_PATH.exists() and RUNTIME_TOKENIZER_PATH.exists():
        lstm_model = load_model(RUNTIME_LSTM_MODEL_PATH, compile=False)
        lstm_tokenizer = _pickle_load(RUNTIME_TOKENIZER_PATH)
    else:
        try:
            if not LEGACY_LSTM_TOKENIZER_PATH.exists():
                raise FileNotFoundError(LEGACY_LSTM_TOKENIZER_PATH)
            lstm_model = load_model(LEGACY_LSTM_MODEL_PATH, compile=False)
            lstm_tokenizer = _pickle_load(LEGACY_LSTM_TOKENIZER_PATH)
        except Exception:
            return None

    return SimilarityArtifacts(
        cnn_tokenizer=cnn_tokenizer,
        lstm_tokenizer=lstm_tokenizer,
        cnn_model=cnn_model,
        lstm_model=lstm_model,
    )


def _train_runtime_artifacts(
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

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    _pickle_dump(RUNTIME_TOKENIZER_PATH, tokenizer)
    cnn_model.save(RUNTIME_CNN_MODEL_PATH)
    lstm_model.save(RUNTIME_LSTM_MODEL_PATH)

    return SimilarityArtifacts(
        cnn_tokenizer=tokenizer,
        lstm_tokenizer=tokenizer,
        cnn_model=cnn_model,
        lstm_model=lstm_model,
    )


@lru_cache(maxsize=1)
def get_similarity_artifacts(
    max_length: int = MAX_SEQUENCE_LENGTH,
    embedding_dim: int = EMBEDDING_DIM,
) -> SimilarityArtifacts:
    runtime_artifacts = _load_runtime_artifacts()
    if runtime_artifacts is not None:
        return runtime_artifacts

    legacy_artifacts = _load_legacy_artifacts()
    if legacy_artifacts is not None:
        return legacy_artifacts

    return _train_runtime_artifacts(max_length=max_length, embedding_dim=embedding_dim)


def warmup_similarity_models(
    max_length: int = MAX_SEQUENCE_LENGTH,
    embedding_dim: int = EMBEDDING_DIM,
) -> SimilarityArtifacts:
    get_similarity_artifacts.cache_clear()
    return get_similarity_artifacts(max_length=max_length, embedding_dim=embedding_dim)


def predict_similarity_scores(
    resume_text: str,
    jd_text: str,
    max_length: int = MAX_SEQUENCE_LENGTH,
    embedding_dim: int = EMBEDDING_DIM,
) -> dict[str, float]:
    artifacts = get_similarity_artifacts(max_length=max_length, embedding_dim=embedding_dim)
    cnn_resume_sequences, cnn_jd_sequences = _prepare_sequence_pairs(
        artifacts.cnn_tokenizer,
        [(resume_text, jd_text)],
        max_length,
    )
    lstm_resume_sequences, lstm_jd_sequences = _prepare_sequence_pairs(
        artifacts.lstm_tokenizer,
        [(resume_text, jd_text)],
        max_length,
    )

    lstm_score = float(artifacts.lstm_model.predict([lstm_resume_sequences, lstm_jd_sequences], verbose=0)[0][0])
    cnn_score = float(artifacts.cnn_model.predict([cnn_resume_sequences, cnn_jd_sequences], verbose=0)[0][0])

    return {
        "lstm_score": max(0.0, min(1.0, lstm_score)),
        "cnn_score": max(0.0, min(1.0, cnn_score)),
    }
