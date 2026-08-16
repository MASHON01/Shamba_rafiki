"""
Application-wide custom exceptions.

Every module should raise these exceptions instead of generic
Python exceptions whenever possible.
"""

from __future__ import annotations


class ShambaRafikiError(Exception):
    """
    Base application exception.

    Every custom exception should inherit from this class.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ConfigurationError(ShambaRafikiError):
    """Raised when configuration is invalid."""


class DependencyNotFoundError(ShambaRafikiError):
    """Raised when a dependency cannot be resolved."""


class RegistryError(ShambaRafikiError):
    """Raised when registry operations fail."""


class StartupError(ShambaRafikiError):
    """Raised during application startup."""


class ShutdownError(ShambaRafikiError):
    """Raised during graceful shutdown."""


class ValidationError(ShambaRafikiError):
    """Raised for internal validation failures."""


class ResourceNotFoundError(ShambaRafikiError):
    """Raised when an internal resource cannot be located."""


class InitializationError(ShambaRafikiError):
    """Raised when a service cannot initialize."""


# =============================================================================
# Knowledge Ingestion Pipeline (Output 3)
# =============================================================================


class IngestionError(ShambaRafikiError):
    """Base exception for the knowledge ingestion pipeline."""


class DocumentLoadError(IngestionError):
    """Raised when a document cannot be read or parsed from disk."""


class CorpusBuildError(IngestionError):
    """Raised when the corpus builder fails to persist processed output."""


class DocumentValidationError(IngestionError, ValidationError):
    """
    Raised when a document or its chunks fail validation.

    Inherits from both IngestionError and ValidationError so callers
    can catch it under either hierarchy.
    """


class UnsupportedFormatError(DocumentValidationError):
    """Raised when a document's file type is not in SUPPORTED_DOCUMENT_TYPES."""


class EmptyDocumentError(DocumentValidationError):
    """Raised when a document or chunk contains no usable text."""


class CorruptedDocumentError(DocumentValidationError):
    """Raised when a document's content is malformed or cannot be trusted."""


class DuplicateDocumentError(DocumentValidationError):
    """Raised when a document duplicates one already present in the corpus."""