/**
 * Anthropic LLM classifier.
 * Ported from doc_analyse/classifiers/anthropic.py
 */

import { BaseClassifier, ClassifierMessage } from "./base.js";

export class AnthropicClassifier extends BaseClassifier {
  readonly provider_name = "anthropic";
  readonly default_model = "claude-3-5-haiku-latest";

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
    const [systemPrompt, userMessages] = _split_anthropic_messages(messages);
    const params: Record<string, unknown> = {
      model: this.model,
      max_tokens: this.max_tokens,
      temperature: this.temperature,
      messages: userMessages,
      ...this.request_options,
    };
    if (systemPrompt) {
      params["system"] = systemPrompt;
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const response = (client as any).messages.create(params);
    return _anthropic_response_text(response);
  }

  private _get_client(): unknown {
    if (this._client !== null) return this._client;

    const options: Record<string, unknown> = { ...this.client_options };
    const { ensure_api_key } = require("./base.js");
    ensure_api_key("Anthropic", ["ANTHROPIC_API_KEY"], this.api_key, options);

    let Anthropic: unknown;
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      Anthropic = require("anthropic").Anthropic;
    } catch {
      throw new (require("./base.js").ClassifierDependencyError)(
        "AnthropicClassifier requires the '@anthropic/sdk' package. Install with: npm install @anthropic/sdk",
      );
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this._client = new (Anthropic as any)(options);
    return this._client;
  }
}

function _split_anthropic_messages(
  messages: readonly ClassifierMessage[],
): [string, readonly { role: string; content: string }[]] {
  const systemParts: string[] = [];
  const userMessages: { role: string; content: string }[] = [];

  for (const message of messages) {
    if (message.role === "system") {
      systemParts.push(message.content);
    } else if (message.role === "user" || message.role === "assistant") {
      userMessages.push({ role: message.role, content: message.content });
    } else {
      userMessages.push({ role: "user", content: message.content });
    }
  }

  return [systemParts.join("\n\n"), userMessages];
}

function _anthropic_response_text(response: {
  content?: readonly { text?: string }[];
}): string {
  const { require_text_response } = require("./base.js");
  const textParts: string[] = [];

  for (const block of response.content ?? []) {
    if (block.text && block.text.trim()) {
      textParts.push(block.text);
    }
  }

  return require_text_response("Anthropic", textParts.join(""));
}
