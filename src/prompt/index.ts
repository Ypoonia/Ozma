/**
 * Prompt loader public API.
 */

export {
  PromptTemplateError,
  load_default_system_prompt,
  load_default_classification_prompt,
  load_classifier_agent_prompt,
  render_classification_prompt,
  resolve_prompt_text,
  require_prompt_text,
  registerPromptLoader,
  SYSTEM_PROMPT_FILE,
  CLASSIFICATION_PROMPT_FILE,
  CLASSIFIER_AGENT_PROMPT_FILE,
  TEXT_PLACEHOLDER,
  METADATA_PLACEHOLDER,
} from "./loader.js";
