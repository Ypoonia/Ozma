/**
 * Prompt template loading and rendering.
 * Ported from doc_analyse/prompt/loader.py
 */

import { readFileSync } from "fs";

export const SYSTEM_PROMPT_FILE = "system.md";
export const CLASSIFICATION_PROMPT_FILE = "classification.md";
export const CLASSIFIER_AGENT_PROMPT_FILE = "classifieragent.md";
export const TEXT_PLACEHOLDER = "{{ text }}";
export const METADATA_PLACEHOLDER = "{{ metadata }}";

export class PromptTemplateError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PromptTemplateError";
  }
}

export function load_default_system_prompt(promptText?: string): string {
  return resolve_prompt_text(promptText, SYSTEM_PROMPT_FILE);
}

export function load_default_classification_prompt(promptText?: string): string {
  return resolve_prompt_text(promptText, CLASSIFICATION_PROMPT_FILE, [
    TEXT_PLACEHOLDER,
    METADATA_PLACEHOLDER,
  ]);
}

export function load_classifier_agent_prompt(promptText?: string): string {
  return resolve_prompt_text(promptText, CLASSIFIER_AGENT_PROMPT_FILE);
}

export function render_classification_prompt(
  promptTemplate: string,
  text: string,
  metadata: Record<string, unknown>,
): string {
  if (!text || !text.trim()) {
    throw new PromptTemplateError(
      "Classifier input text must be a non-empty string.",
    );
  }

  return promptTemplate
    .replace(METADATA_PLACEHOLDER, _format_metadata(metadata))
    .replace(TEXT_PLACEHOLDER, text)
    .trim();
}

export function resolve_prompt_text(
  promptText: string | undefined,
  defaultFilename: string,
  requiredPlaceholders: string[] = [],
): string {
  let text = promptText;
  if (text === undefined) {
    text = _resolvePrompt(defaultFilename);
  }

  text = require_prompt_text(defaultFilename, text);

  const missingPlaceholders = requiredPlaceholders.filter(
    (p) => !text!.includes(p),
  );
  if (missingPlaceholders.length > 0) {
    throw new PromptTemplateError(
      `Prompt template '${defaultFilename}' is missing required placeholder(s): ${missingPlaceholders.join(", ")}`,
    );
  }

  return text;
}

function _load_embedded_prompt(filename: string): string {
  try {
    const fileUrl = new URL(`./${filename}`, import.meta.url);
    return readFileSync(fileUrl, "utf-8");
  } catch (err) {
    throw new PromptTemplateError(
      `Prompt template '${filename}' could not be loaded.`,
    );
  }
}

export function require_prompt_text(
  promptName: string,
  promptText: string,
): string {
  if (!promptText || !promptText.trim()) {
    throw new PromptTemplateError(
      `Prompt template '${promptName}' is empty.`,
    );
  }
  return promptText.trim();
}

function _format_metadata(
  metadata: Record<string, unknown>,
): string {
  if (!metadata || Object.keys(metadata).length === 0) {
    return "none";
  }

  return Object.entries(metadata)
    .map(([key, value]) => `- ${key}: ${value}`)
    .join("\n");
}

// ---------------------------------------------------------------------------
// Prompt registry — allows runtime prompt loading from the file system
// ---------------------------------------------------------------------------

type PromptLoader = (filename: string) => string;

let _promptLoader: PromptLoader | null = null;

export function registerPromptLoader(loader: PromptLoader): void {
  _promptLoader = loader;
}

export function _resolvePrompt(filename: string): string {
  if (_promptLoader) {
    return _promptLoader(filename);
  }
  return _load_embedded_prompt(filename);
}
