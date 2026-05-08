/**
 * Google Gemini LLM classifier.
 * Ported from doc_analyse/classifiers/gemini.py
 */

import { BaseClassifier, ClassifierMessage } from "./base.js";

export class GeminiClassifier extends BaseClassifier {
  readonly provider_name = "gemini";
  readonly default_model = "gemini-2.0-flash";

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
    const { render_messages_for_single_prompt } = require("./base.js");
    const prompt = render_messages_for_single_prompt(messages);
    const config = this._build_generation_config();

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const response = (client as any).models.generate_content({
      model: this.model,
      contents: prompt,
      config,
      ...this.request_options,
    });
    return _gemini_response_text(response);
  }

  private _get_client(): unknown {
    if (this._client !== null) return this._client;

    const options: Record<string, unknown> = { ...this.client_options };
    const { ensure_api_key } = require("./base.js");
    ensure_api_key("Gemini", ["GEMINI_API_KEY", "GOOGLE_API_KEY"], this.api_key, options);

    let genai: unknown;
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      genai = require("@google/generative-ai");
    } catch {
      throw new (require("./base.js").ClassifierDependencyError)(
        "GeminiClassifier requires the '@google/generative-ai' package. Install with: npm install @google/generative-ai",
      );
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this._client = new (genai as any).Client(options);
    return this._client;
  }

  private _build_generation_config(): unknown {
    const { ClassifierDependencyError } = require("./base.js");
    let types: unknown;
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const genai = require("@google/generative-ai");
      types = genai.types ?? genai.GenAIConfig;
    } catch {
      throw new ClassifierDependencyError(
        "GeminiClassifier requires the '@google/generative-ai' package. Install with: npm install @google/generative-ai",
      );
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const t = types as any;
    return new t.GenerateContentConfig({
      temperature: this.temperature,
      max_output_tokens: this.max_tokens,
      response_mime_type: "application/json",
    });
  }
}

function _gemini_response_text(response: { text?: string }): string {
  const { require_text_response } = require("./base.js");
  return require_text_response("Gemini", response.text ?? null);
}
