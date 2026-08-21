"""
Shared type definitions used throughout Farm Pal.

This module centralizes reusable type aliases to improve
readability, consistency, and maintainability across the
application.

Avoid redefining common types elsewhere in the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias
from uuid import UUID

# =============================================================================
# Generic Types
# =============================================================================

#: Generic JSON-compatible dictionary.
JSON: TypeAlias = dict[str, Any]

#: Generic metadata dictionary.
Metadata: TypeAlias = dict[str, Any]

#: Generic configuration mapping.
Config: TypeAlias = dict[str, Any]

#: Generic headers mapping.
Headers: TypeAlias = dict[str, str]


# =============================================================================
# Identifier Types
# =============================================================================

#: Unique request identifier.
RequestID: TypeAlias = UUID

#: Document identifier.
DocumentID: TypeAlias = UUID

#: Chunk identifier.
ChunkID: TypeAlias = UUID

#: Session identifier.
SessionID: TypeAlias = str


# =============================================================================
# File System Types
# =============================================================================

#: Absolute or relative file path.
FilePath: TypeAlias = Path

#: Directory path.
DirectoryPath: TypeAlias = Path

#: File checksum/hash.
FileHash: TypeAlias = str


# =============================================================================
# Language Types
# =============================================================================

#: ISO language code (e.g. "en", "sw").
LanguageCode: TypeAlias = str

#: User input text.
Text: TypeAlias = str

#: Normalized query.
Query: TypeAlias = str


# =============================================================================
# Embedding & Vector Types
# =============================================================================

#: Single embedding vector.
Embedding: TypeAlias = list[float]

#: Dense vector.
Vector: TypeAlias = list[float]

#: Matrix of embeddings.
EmbeddingMatrix: TypeAlias = list[list[float]]

#: Similarity score.
SimilarityScore: TypeAlias = float


# =============================================================================
# Document Types
# =============================================================================

#: Raw document contents.
DocumentText: TypeAlias = str

#: Chunk contents.
ChunkText: TypeAlias = str

#: Collection of chunks.
ChunkCollection: TypeAlias = list[str]


# =============================================================================
# AI & Retrieval Types
# =============================================================================

#: Prompt sent to the LLM.
Prompt: TypeAlias = str

#: Generated model response.
LLMResponse: TypeAlias = str

#: Retrieved context passages.
RetrievedContext: TypeAlias = list[str]


# =============================================================================
# Vision Types
# =============================================================================

#: Image classifier label.
ClassificationLabel: TypeAlias = str

#: Image classifier confidence.
ConfidenceScore: TypeAlias = float

#: Predicted class names.
Predictions: TypeAlias = list[str]


# =============================================================================
# Timing Types
# =============================================================================

#: Processing duration in milliseconds.
Milliseconds: TypeAlias = float

#: Processing duration in seconds.
Seconds: TypeAlias = float