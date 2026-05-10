from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any, Optional

PROMPT_PACKAGE = "doc_analyse.prompt"
SYSTEM_PROMPT_FILE = "system.md"
CLASSIFICATION_PROMPT_FILE = "classification.md"
CLASSIFIER_AGENT_PROMPT_FILE = "classifieragent.md"
TEXT_PLACEHOLDER = "{{ text }}"
METADATA_PLACEHOLDER = "{{ metadata }}"

# Single-pass substitution: matches the exact placeholder forms above. Doing
# both replacements in one pass means a value substituted in for one
# placeholder cannot accidentally introduce another placeholder for the
# next replacement to consume — i.e. an attacker-controlled metadata value
# of "{{ text }}" can no longer pull the chunk text into the metadata slot.
#
# The pattern matches the *literal* placeholder forms (one space inside the
# braces) so it stays in lockstep with ``resolve_prompt_text``'s literal-
# containment validation in ``required_placeholders``. Templates using a
# different whitespace shape would fail validation anyway; we don't want
# the substitution to be more permissive than the validator.
_PLACEHOLDER_PATTERN = re.compile(r"\{\{ (text|metadata) \}\}")


class PromptTemplateError(RuntimeError):
    pass


def load_default_system_prompt(prompt_text: Optional[str] = None) -> str:
    return resolve_prompt_text(prompt_text, SYSTEM_PROMPT_FILE)


def load_default_classification_prompt(prompt_text: Optional[str] = None) -> str:
    return resolve_prompt_text(
        prompt_text,
        CLASSIFICATION_PROMPT_FILE,
        required_placeholders=(TEXT_PLACEHOLDER, METADATA_PLACEHOLDER),
    )


def load_classifier_agent_prompt(prompt_text: Optional[str] = None) -> str:
    return resolve_prompt_text(prompt_text, CLASSIFIER_AGENT_PROMPT_FILE)


def render_classification_prompt(
    prompt_template: str,
    text: str,
    metadata: Mapping[str, Any],
) -> str:
    if not isinstance(text, str) or not text.strip():
        raise PromptTemplateError("Classifier input text must be a non-empty string.")

    values = {"text": text, "metadata": _format_metadata(metadata)}
    return _PLACEHOLDER_PATTERN.sub(
        lambda match: values[match.group(1)], prompt_template
    ).strip()


def resolve_prompt_text(
    prompt_text: Optional[str],
    default_filename: str,
    required_placeholders: Sequence[str] = (),
) -> str:
    if prompt_text is None:
        prompt_text = load_prompt_template(default_filename)

    prompt_text = require_prompt_text(default_filename, prompt_text)
    missing_placeholders = [
        placeholder for placeholder in required_placeholders if placeholder not in prompt_text
    ]
    if missing_placeholders:
        missing = ", ".join(missing_placeholders)
        raise PromptTemplateError(
            f"Prompt template '{default_filename}' is missing required placeholder(s): {missing}"
        )

    return prompt_text


def load_prompt_template(filename: str) -> str:
    try:
        # Prompt text lives in markdown resources so reviewers can change behavior
        # without editing provider or verifier code.
        prompt_text = files(PROMPT_PACKAGE).joinpath(filename).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise PromptTemplateError(f"Prompt template '{filename}' could not be loaded.") from exc

    return require_prompt_text(filename, prompt_text)


def require_prompt_text(prompt_name: str, prompt_text: Any) -> str:
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise PromptTemplateError(f"Prompt template '{prompt_name}' is empty.")

    return prompt_text.strip()


def _format_metadata(metadata: Mapping[str, Any]) -> str:
    if not metadata:
        return "none"

    return "\n".join(f"- {key}: {value}" for key, value in metadata.items())
