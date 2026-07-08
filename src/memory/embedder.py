# DMS reproduction project — embedding function phi(.) for Dual-Factor Retrieval.
#
# Implements the embedding function `phi(.)` used in the paper's Dual-Factor
# Similarity Metric (Sec 3.2.2):
#
#   Score(p_hat, p) = sim(phi(p_hat_pre), phi(p_pre)) * sim(phi(p_hat_goal), phi(p_goal))
#
# We use a small, local, CPU-only sentence embedding model
# (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim, ~90MB) rather than an
# API-based embedding service, so retrieval works fully offline and never
# touches a GPU (the 8 GPUs on this shared server are reserved for the VLM
# / other users' workloads per project constraints).

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

# Force fully-local, no-network behavior BEFORE `sentence_transformers`/
# `huggingface_hub` are ever imported (those libraries snapshot some of
# these env vars as module-level constants at import time, so setting
# them any later has no effect). This shared server's ambient shell sets
# a SOCKS proxy (used for other tools) that `httpx` can't even construct
# a client for unless the `socksio` extra is installed -- and we don't
# need any proxy at all here once the embedding model has been downloaded
# once, so we simply strip proxy env vars for this process and pin
# offline mode.
for _proxy_var in (
    "all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY",
    "https_proxy", "HTTPS_PROXY",
):
  os.environ.pop(_proxy_var, None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "embeddings",
)

_lock = threading.Lock()
_model_cache: dict[str, "object"] = {}


def _get_model(model_name: str = _DEFAULT_MODEL_NAME):
  """Lazily loads (and caches) the sentence-transformers model on CPU."""
  with _lock:
    if model_name not in _model_cache:
      # Imported lazily so importing this module doesn't require torch
      # unless embeddings are actually requested.
      from sentence_transformers import SentenceTransformer  # pylint: disable=g-import-not-at-top

      _model_cache[model_name] = SentenceTransformer(
          model_name, cache_folder=_DEFAULT_CACHE_DIR, device="cpu"
      )
    return _model_cache[model_name]


class Embedder:
  """Wraps a local sentence embedding model: phi(text) -> dense vector."""

  def __init__(self, model_name: str = _DEFAULT_MODEL_NAME):
    self.model_name = model_name

  @property
  def dim(self) -> int:
    return _get_model(self.model_name).get_sentence_embedding_dimension()

  def embed(self, text: str) -> np.ndarray:
    """Embeds a single string into a unit-normalized dense vector."""
    return self.embed_batch([text])[0]

  def embed_batch(self, texts: list[str]) -> np.ndarray:
    """Embeds a batch of strings into unit-normalized dense vectors."""
    model = _get_model(self.model_name)
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
  """Cosine similarity, robust to non-unit-normalized inputs."""
  denom = float(np.linalg.norm(a) * np.linalg.norm(b))
  if denom == 0.0:
    return 0.0
  return float(np.dot(a, b) / denom)


_default_embedder: Optional[Embedder] = None


def get_default_embedder() -> Embedder:
  global _default_embedder
  if _default_embedder is None:
    _default_embedder = Embedder()
  return _default_embedder
