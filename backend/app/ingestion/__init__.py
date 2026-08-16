"""
Knowledge Ingestion Pipeline.

Turns raw source documents (KALRO extension manuals, AFA crop
strategy docs, county-level context) into a validated, chunked,
metadata-tagged corpus ready for embedding in Output 4.

    Documents -> Clean Text -> Chunks -> Metadata -> Corpus

Public entry point: `app.ingestion.pipeline.IngestionPipeline`.
"""

from __future__ import annotations