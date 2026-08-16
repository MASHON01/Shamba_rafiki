"""
Versioned prompt template content (English + Swahili, per intent).

Just the strings, grouped by version. The template_registry selects and
assembles them; nothing here has behaviour. output_format.py holds the
per-intent response-shape guidance the v2 templates compose.
"""

from __future__ import annotations

from app.orchestration.prompts.templates.output_format import (
    OUTPUT_FORMAT_EN,
    OUTPUT_FORMAT_SW,
)
from app.orchestration.prompts.templates.system_en import (
    EN_V1_BASE,
    EN_V1_INTENT,
    EN_V2_BASE,
    EN_V2_INTENT,
)
from app.orchestration.prompts.templates.system_sw import (
    SW_V1_BASE,
    SW_V1_INTENT,
    SW_V2_BASE,
    SW_V2_INTENT,
)

__all__ = [
    "OUTPUT_FORMAT_EN",
    "OUTPUT_FORMAT_SW",
    "EN_V1_BASE",
    "EN_V1_INTENT",
    "EN_V2_BASE",
    "EN_V2_INTENT",
    "SW_V1_BASE",
    "SW_V1_INTENT",
    "SW_V2_BASE",
    "SW_V2_INTENT",
]
