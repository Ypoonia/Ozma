from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1200


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    max_tokens: int


def resolve_generation_config(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> GenerationConfig:
    resolved_temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    resolved_max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    if not isinstance(resolved_temperature, (int, float)) or isinstance(resolved_temperature, bool):
        raise ValueError("temperature must be a number or None.")

    if not isinstance(resolved_max_tokens, int) or isinstance(resolved_max_tokens, bool):
        raise ValueError("max_tokens must be an integer or None.")

    if resolved_max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0.")

    return GenerationConfig(
        temperature=float(resolved_temperature),
        max_tokens=resolved_max_tokens,
    )
