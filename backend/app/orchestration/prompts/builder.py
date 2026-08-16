"""
Prompt builder.

The single place that knows the full shape of a prompt. It combines,
in order:

    system prompt        (system_prompts.get_system_prompt)
    reference context    (ContextBuilder output)
    conversation history (optional, from memory.py in a later batch)
    the farmer's question

into a `BuiltPrompt` the LLM client can send. The builder is
deliberately endpoint-agnostic: it exposes both a single assembled
string (for llama-server's /completion endpoint) and the separated
system/user parts plus history (for a /v1/chat/completions endpoint),
so whichever the llama client uses, nothing here changes.

Keeping assembly here - not in the orchestrator - means the prompt's
structure is defined and testable in one spot, independent of how the
stages are wired together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.constants import DEFAULT_INTENT
from app.models.language import LanguageCode
from app.orchestration.prompts.context_builder import BuiltContext
from app.orchestration.prompts.system_prompts import get_system_prompt


@dataclass(slots=True)
class ConversationTurn:
    """A single prior exchange, oldest-relevant first when listed."""

    question: str
    answer: str


@dataclass(slots=True)
class BuiltPrompt:
    """
    The assembled prompt in two interchangeable forms.

    system_prompt / user_prompt / history support a chat-style
    endpoint; `full_prompt` is the flattened single string for a
    completion-style endpoint. The LLM client picks whichever matches
    the endpoint it calls.
    """

    system_prompt: str
    user_prompt: str
    full_prompt: str
    history: list[ConversationTurn] = field(default_factory=list)


class PromptBuilder:
    """
    Assembles the final prompt from its parts.
    """

    def build(
        self,
        question: str,
        context: BuiltContext,
        language: LanguageCode = LanguageCode.ENGLISH,
        intent: str = DEFAULT_INTENT,
        history: list[ConversationTurn] | None = None,
    ) -> BuiltPrompt:
        history = history or []

        system_prompt = get_system_prompt(language, intent)

        # The user-facing portion: reference material then the question.
        user_prompt = (
            f"Reference material:\n{context.text}\n\n"
            f"Question: {question.strip()}"
        )

        full_prompt = self._flatten(system_prompt, user_prompt, history)

        return BuiltPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            full_prompt=full_prompt,
            history=history,
        )

    def _flatten(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[ConversationTurn],
    ) -> str:
        """
        Flatten everything into one string for a completion endpoint.
        History is rendered as prior Q/A pairs so a single-string model
        still sees the conversation.
        """
        parts = [system_prompt, ""]

        for turn in history:
            parts.append(f"Question: {turn.question.strip()}")
            parts.append(f"Answer: {turn.answer.strip()}")
            parts.append("")

        parts.append(user_prompt)
        parts.append("Answer:")

        return "\n".join(parts)