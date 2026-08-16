"""
On-disk embedding cache.

Stores computed embeddings keyed by (content, model, dimension,
cache-version) so re-running the indexer over a mostly-unchanged
corpus skips the expensive encode step for chunks it has already
seen. This is what makes iterating on the corpus cheap.

Design notes:
- The key is a SHA-256 of the *text* plus the model identity, so the
  same text embedded by a different model doesn't collide, and a
  model swap transparently invalidates the whole cache.
- Vectors are stored as JSON (list[float]) - simple, language-
  agnostic, and small at a few thousand chunks. If the corpus ever
  grows enough that JSON overhead matters, this is the one file to
  change; nothing else depends on the on-disk format.
- Cache misses and corrupt entries are never fatal: a bad cache
  should only ever cost a re-embed, never crash indexing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.config.constants import EMBEDDING_CACHE_VERSION
from app.config.paths import EMBEDDINGS_DIR
from app.utils.logger import get_logger

logger = get_logger("EmbeddingCache")


class EmbeddingCache:
    """
    Content-addressed cache of embeddings on disk.
    """

    def __init__(
        self,
        model_id: str,
        dimension: int,
        cache_dir: Path | None = None,
    ) -> None:
        self._model_id = model_id
        self._dimension = dimension
        self._cache_dir = cache_dir or EMBEDDINGS_DIR
        self._hits = 0
        self._misses = 0

    def get(self, text: str) -> list[float] | None:
        """
        Return the cached embedding for `text`, or None on a miss.
        A corrupt or wrong-dimension entry is treated as a miss (and
        removed) rather than trusted.
        """
        entry_path = self._entry_path(text)

        if not entry_path.exists():
            self._misses += 1
            return None

        try:
            payload = json.loads(entry_path.read_text(encoding="utf-8"))
            vector = payload["embedding"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "embedding_cache.entry_corrupt",
                path=str(entry_path),
                error=str(exc),
            )
            self._misses += 1
            self._safe_unlink(entry_path)
            return None

        if not isinstance(vector, list) or len(vector) != self._dimension:
            logger.warning(
                "embedding_cache.dimension_mismatch",
                path=str(entry_path),
                expected=self._dimension,
                got=len(vector) if isinstance(vector, list) else "n/a",
            )
            self._misses += 1
            self._safe_unlink(entry_path)
            return None

        self._hits += 1
        return vector

    def set(self, text: str, embedding: list[float]) -> None:
        """
        Store an embedding for `text`. Storage failures are logged,
        not raised - a cache that can't write should slow indexing,
        not break it.
        """
        if len(embedding) != self._dimension:
            logger.warning(
                "embedding_cache.refuse_wrong_dimension",
                expected=self._dimension,
                got=len(embedding),
            )
            return

        entry_path = self._entry_path(text)

        try:
            entry_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_id": self._model_id,
                "dimension": self._dimension,
                "version": EMBEDDING_CACHE_VERSION,
                "embedding": embedding,
            }
            entry_path.write_text(
                json.dumps(payload), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning(
                "embedding_cache.write_failed",
                path=str(entry_path),
                error=str(exc),
            )

    @property
    def stats(self) -> dict[str, int]:
        """Hit/miss counters for the current process."""
        return {"hits": self._hits, "misses": self._misses}

    def _key(self, text: str) -> str:
        """
        Content+identity address for a piece of text. Model id,
        dimension, and cache version are folded in so any of them
        changing produces a different key (i.e. a clean invalidation).
        """
        hasher = hashlib.sha256()
        hasher.update(EMBEDDING_CACHE_VERSION.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(self._model_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(str(self._dimension).encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()

    def _entry_path(self, text: str) -> Path:
        """
        Path for a cache entry. Sharded one level by the first two
        hex chars of the key so a large corpus doesn't put thousands
        of files in a single directory.
        """
        key = self._key(text)
        return self._cache_dir / "cache" / key[:2] / f"{key}.json"

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass