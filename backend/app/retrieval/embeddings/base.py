"""
Abstract embedder interface.

Every embedder turns text into fixed-width dense vectors. Two entry
points, deliberately distinct:

    embed_texts(list[str]) -> list[list[float]]   # corpus, batched, build-time
    embed_query(str)       -> list[float]         # a single user query, runtime

They're separate because some embedding models prepend different
instructions to documents vs queries (asymmetric models), and
because batching only ever applies to the corpus side. Keeping both
on the interface means a future model swap can honor that
distinction without changing any caller.

Concrete embedders subclass this; the vector store and retriever
depend only on this interface, never on a specific model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    Base class for all embedding backends.
    """

    #: Output dimension of this embedder's vectors. Subclasses set
    #: this so the vector store can size its arrays without having to
    #: run an embedding first.
    dimension: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts (the corpus path).

        Returns one vector per input text, in the same order. An empty
        input returns an empty list.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string (the runtime path).
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"{type(self).__name__}(dimension={getattr(self, 'dimension', '?')})"