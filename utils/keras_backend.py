from __future__ import annotations

import os


def configure_keras_backend() -> None:
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
