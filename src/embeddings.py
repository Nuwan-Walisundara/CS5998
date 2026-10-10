"""Transformer sentence embeddings, wrapped as a scikit-learn transformer.

Keeping the same interface as TfidfVectorizer lets the baseline and the
transformer approach share the classifier, folds and metrics; only the
text representation changes.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


@lru_cache(maxsize=2)
def load_encoder(name: str = config.EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def encode(texts, name: str = config.EMBEDDING_MODEL) -> np.ndarray:
    return load_encoder(name).encode(list(texts), normalize_embeddings=True,
                                     batch_size=32, show_progress_bar=False)


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return encode(X, self.model_name)
