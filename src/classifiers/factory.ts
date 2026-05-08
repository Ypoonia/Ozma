/**
 * Classifier factory.
 * Ported from doc_analyse/classifiers/factory.py
 */

import { BaseClassifier } from "./base.js";
import { AnthropicClassifier } from "./anthropic.js";
import { OpenAIClassifier } from "./openai.js";
import { GeminiClassifier } from "./gemini.js";

const PROVIDERS: Readonly<Record<string, typeof BaseClassifier>> = {
  anthropic: AnthropicClassifier,
  claude: AnthropicClassifier,
  codex: OpenAIClassifier,
  gemini: GeminiClassifier,
  google: GeminiClassifier,
  openai: OpenAIClassifier,
};

const PROVIDER_API_KEY_ENV: Readonly<Record<string, string>> = {
  anthropic: "ANTHROPIC_API_KEY",
  claude: "ANTHROPIC_API_KEY",
  codex: "OPENAI_API_KEY",
  gemini: "GEMINI_API_KEY",
  google: "GOOGLE_API_KEY",
  openai: "OPENAI_API_KEY",
};

export function build_classifier(
  provider: string,
  kwargs: Record<string, unknown> = {},
): BaseClassifier {
  const key = provider.trim().toLowerCase();
  const ClassifierType = PROVIDERS[key];

  if (!ClassifierType) {
    const available = Object.keys(PROVIDERS).sort().join(", ");
    throw new Error(
      `Unknown classifier provider '${provider}'. Available providers: ${available}`,
    );
  }

  return new (ClassifierType as new (kwargs: Record<string, unknown>) => BaseClassifier)(kwargs);
}

export function classifier_from_env(
  prefix = "DOC_ANALYSE_LLM",
  kwargs: Record<string, unknown> = {},
): BaseClassifier {
  const provider = (process.env[`${prefix}_PROVIDER`] ?? "openai")
    .trim()
    .toLowerCase();
  const model = process.env[`${prefix}_MODEL`];
  const apiKey =
    process.env[`${prefix}_API_KEY`] ??
    _provider_api_key(provider);

  const classifierKwargs: Record<string, unknown> = { ...kwargs };
  if (model && !("model" in classifierKwargs)) {
    classifierKwargs["model"] = model;
  }
  if (apiKey && !("api_key" in classifierKwargs)) {
    classifierKwargs["api_key"] = apiKey;
  }

  return build_classifier(provider, classifierKwargs);
}

function _provider_api_key(provider: string): string {
  const envName = PROVIDER_API_KEY_ENV[provider];
  if (!envName) return "";

  if (provider === "gemini" || provider === "google") {
    return process.env["GEMINI_API_KEY"] ?? process.env["GOOGLE_API_KEY"] ?? "";
  }

  return process.env[envName] ?? "";
}
