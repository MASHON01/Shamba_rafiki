"""
Prompt assembly: system prompts, context formatting, final build.

    get_system_prompt   language + intent -> system instructions
    ContextBuilder      retrieved chunks -> numbered reference block
    PromptBuilder       all parts -> BuiltPrompt (for the LLM client)

The BuiltPrompt exposes both a flattened string and separated
system/user/history parts, so the LLM client works against either a
completion or a chat endpoint.
"""

from __future__ import annotations

from app.orchestration.prompts.builder import (
    BuiltPrompt,
    ConversationTurn,
    PromptBuilder,
)
from app.orchestration.prompts.context_builder import (
    BuiltContext,
    ContextBuilder,
)
from app.orchestration.prompts.system_prompts import get_system_prompt

__all__ = [
    "get_system_prompt",
    "ContextBuilder",
    "BuiltContext",
    "PromptBuilder",
    "BuiltPrompt",
    "ConversationTurn",
]