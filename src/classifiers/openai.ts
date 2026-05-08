/**
 * OpenAI LLM classifier.
 * Ported from doc_analyse/classifiers/openai.py
 */

import { BaseClassifier, ClassifierMessage } from "./base.js";

export class OpenAIClassifier extends BaseClassifier {
  readonly provider_name = "openai";
  readonly default_model = "gpt-4o-mini";

  private api_key: string | null;
  private _client: unknown | null = null;
  private readonly client_options: Record<string, unknown>;
  private readonly request_options: Record<string, unknown>;

  constructor({
    model,
    api_key,
    client,
    temperature,
    max_tokens,
    system_prompt,
    user_prompt_template,
    request_options,
    ...client_options
  }: {
    model?: string | null;
    api_key?: string | null;
    client?: unknown;
    temperature?: number | null;
    max_tokens?: number | null;
    system_prompt?: string | null;
    user_prompt_template?: string | null;
    request_options?: Record<string, unknown>;
    [key: string]: unknown;
  } = {}) {
    super({ model, temperature, max_tokens, system_prompt, user_prompt_template });
    this.api_key = api_key ?? null;
    this._client = client ?? null;
    this.client_options = client_options;
    this.request_options = request_options ?? {};
  }

  _complete(messages: readonly ClassifierMessage[]): string {
    const client = this._get_client();
    const payload = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const c = client as any;
    if (typeof c.responses !== "undefined") {
      const response = c.responses.create({
        model: this.model,
        input: payload,
        temperature: this.temperature,
        max_output_tokens: this.max_tokens,
        ...this.request_options,
      });
      return _openai_response_text(response);
    }

    const response = c.chat.completions.create({
      model: this.model,
      messages: payload,
      temperature: this.temperature,
      max_tokens: this.max_tokens,
      ...this.request_options,
    });
    return _openai_chat_response_text(response);
  }

  private _get_client(): unknown {
    if (this._client !== null) return this._client;

    const options: Record<string, unknown> = { ...this.client_options };
    const { ensure_api_key } = require("./base.js");
    ensure_api_key("OpenAI", ["OPENAI_API_KEY"], this.api_key, options);

    let OpenAI: unknown;
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      OpenAI = require("openai").OpenAI;
    } catch {
      throw new (require("./base.js").ClassifierDependencyError)(
        "OpenAIClassifier requires the 'openai' package. Install with: npm install openai",
      );
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this._client = new (OpenAI as any)(options);
    return this._client;
  }
}

function _openai_response_text(response: {
  output_text?: string;
  output?: readonly { content?: readonly { text?: string }[] }[];
}): string {
  const { require_text_response } = require("./base.js");

  if (response.output_text && response.output_text.trim()) {
    return response.output_text.trim();
  }

  const textParts: string[] = [];
  for (const item of response.output ?? []) {
    for (const content of item.content ?? []) {
      if (content.text && content.text.trim()) {
        textParts.push(content.text);
      }
    }
  }

  return require_text_response("OpenAI", textParts.join(""));
}

function _openai_chat_response_text(response: {
  choices?: readonly { message?: { content?: string } }[];
}): string {
  const { require_text_response } = require("./base.js");
  let text: string | null = null;

  try {
    text = response.choices?.[0]?.message?.content ?? null;
  } catch {
    text = null;
  }

  return require_text_response("OpenAI", text);
}
