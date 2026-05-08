/**
 * Generation config for LLM classifiers.
 * Ported from doc_analyse/classifiers/config.py
 */

export const DEFAULT_TEMPERATURE = 0.0;
export const DEFAULT_MAX_TOKENS = 1200;

export interface GenerationConfig {
  readonly temperature: number;
  readonly max_tokens: number;
}

export function resolve_generation_config({
  temperature,
  max_tokens,
}: {
  temperature?: number | null;
  max_tokens?: number | null;
}): GenerationConfig {
  const resolvedTemperature =
    temperature === undefined || temperature === null
      ? DEFAULT_TEMPERATURE
      : temperature;
  const resolvedMaxTokens =
    max_tokens === undefined || max_tokens === null
      ? DEFAULT_MAX_TOKENS
      : max_tokens;

  if (
    typeof resolvedTemperature !== "number" ||
    Number.isNaN(resolvedTemperature) ||
    typeof resolvedTemperature === "boolean"
  ) {
    throw new Error("temperature must be a number or null.");
  }

  if (
    !Number.isInteger(resolvedMaxTokens) ||
    typeof resolvedMaxTokens === "boolean"
  ) {
    throw new Error("max_tokens must be an integer or null.");
  }

  if (resolvedMaxTokens <= 0) {
    throw new Error("max_tokens must be greater than 0.");
  }

  return Object.freeze({
    temperature: resolvedTemperature,
    max_tokens: resolvedMaxTokens,
  });
}
