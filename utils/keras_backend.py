from __future__ import annotations

import os


def configure_keras_backend() -> None:
    os.environ.setdefault("KERAS_BACKEND", "torch")