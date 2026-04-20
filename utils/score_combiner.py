from __future__ import annotations


def combine_similarity_scores(tfidf_score: float, lstm_score: float, cnn_score: float) -> float:
    weighted_score = (
        0.4 * float(tfidf_score)
        + 0.3 * float(lstm_score)
        + 0.3 * float(cnn_score)
    )
    percentage = weighted_score * 100.0
    return max(0.0, min(100.0, percentage))