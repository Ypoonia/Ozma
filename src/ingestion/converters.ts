/**
 * Document converters for ingestion pipeline.
 * Ported from doc_analyse/ingestion/converters.py
 */

import { readFileSync } from "fs";
import { createRequire } from "module";
import { extname } from "path";
import {
  createExtractedDocument,
  createDocumentSegment,
  DocumentSegment,
  ExtractedDocument,
} from "./models.js";

const require = createRequire(import.meta.url);

export class DocumentConversionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DocumentConversionError";
  }
}

export class ConverterDependencyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConverterDependencyError";
  }
}

export class UnsupportedDocumentError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnsupportedDocumentError";
  }
}

export interface BaseDocumentConverter {
  supported_extensions: ReadonlySet<string>;
  supports_any_extension: boolean;
  supports(path: string): boolean;
  convert(path: string): ExtractedDocument;
}

export class TextDocumentConverter implements BaseDocumentConverter {
  readonly supported_extensions: ReadonlySet<string> = new Set([".txt", ".md", ".markdown"]);
  readonly supports_any_extension = false;
  private readonly encoding: string;

  constructor({ encoding = "utf-8" }: { encoding?: string } = {}) {
    this.encoding = encoding;
  }

  supports(path: string): boolean {
    return this.supports_any_extension || this.supported_extensions.has(extname(path).toLowerCase());
  }

  convert(path: string): ExtractedDocument {
    const text = readFileSync(path, { encoding: this.encoding as BufferEncoding });
    return _document_from_text(path, text, {
      converter: "text",
      extension: extname(path).toLowerCase(),
      normalized_format: "text",
    });
  }
}

export class MarkItDownDocumentConverter implements BaseDocumentConverter {
  readonly supported_extensions: ReadonlySet<string> = new Set();
  readonly supports_any_extension = true;
  private _markitdown: unknown | null = null;
  private readonly enable_plugins: boolean;
  private readonly options: Record<string, unknown>;

  constructor({
    enable_plugins = false,
    options = {},
  }: { enable_plugins?: boolean; options?: Record<string, unknown> } = {}) {
    this.enable_plugins = enable_plugins;
    this.options = options;
  }

  supports(_path: string): boolean {
    return true;
  }

  convert(path: string): ExtractedDocument {
    let client: unknown;
    try {
      client = this._getClient();
    } catch (err) {
      if (err instanceof ConverterDependencyError) throw err;
    }

    if (!client) throw new ConverterDependencyError("MarkItDown client unavailable.");

    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const result = (client as any).convert(path);
      return _document_from_text(path, _markitdown_text(result), {
        converter: "markitdown",
        extension: extname(path).toLowerCase(),
        normalized_format: "markdown",
      });
    } catch (err) {
      if (err instanceof DocumentConversionError) throw err;
      throw new DocumentConversionError(
        `Could not convert document with MarkItDown: ${path}: ${err}`,
      );
    }
  }

  private _getClient(): unknown {
    if (this._markitdown !== null) return this._markitdown;

    let MarkItDown: unknown;
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const mod = require("markitdown");
      MarkItDown = mod.MarkItDown ?? mod.default?.MarkItDown ?? mod;
    } catch {
      throw new ConverterDependencyError(
        "Missing document converter dependency. Install ozma[conversion] to convert rich document formats.",
      );
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this._markitdown = new (MarkItDown as any)({
      enable_plugins: this.enable_plugins,
      ...this.options,
    });
    return this._markitdown;
  }
}

export class ConverterRegistry {
  private readonly _converters: BaseDocumentConverter[] = [];

  constructor(converters?: BaseDocumentConverter[]) {
    if (converters) {
      this._converters.push(...converters);
    }
  }

  register(converter: BaseDocumentConverter): void {
    this._converters.push(converter);
  }

  convert(path: string): ExtractedDocument {
    const resolvedPath = _resolve_existing_file(path);
    const converter = this._find_converter(resolvedPath);
    if (!converter) {
      throw new UnsupportedDocumentError(
        `Unsupported document extension: ${extname(resolvedPath) || "<none>"}`,
      );
    }
    return converter.convert(resolvedPath);
  }

  private _find_converter(path: string): BaseDocumentConverter | null {
    for (const converter of this._converters) {
      if (converter.supports(path)) return converter;
    }
    return null;
  }
}

export function default_registry(): ConverterRegistry {
  return new ConverterRegistry([
    new TextDocumentConverter(),
    new MarkItDownDocumentConverter(),
  ]);
}

export function convert_document(
  path: string,
  registry?: ConverterRegistry,
): ExtractedDocument {
  return (registry ?? default_registry()).convert(path);
}

function _resolve_existing_file(path: string): string {
  const { existsSync, statSync } = require("fs");
  const { resolve, isAbsolute } = require("path");

  const resolvedPath = isAbsolute(path) ? path : resolve(process.cwd(), path);

  if (!existsSync(resolvedPath)) {
    throw new DocumentConversionError(`Document does not exist: ${resolvedPath}`);
  }
  if (!statSync(resolvedPath).isFile()) {
    throw new DocumentConversionError(`Document path is not a file: ${resolvedPath}`);
  }
  return resolvedPath;
}

function _document_from_text(
  path: string,
  text: string,
  metadata: Record<string, unknown>,
): ExtractedDocument {
  const normalizedText = _require_converted_text(path, text);

  const segment: DocumentSegment = createDocumentSegment(
    normalizedText,
    0,
    normalizedText.length,
    Object.freeze({ segment_type: "body" }),
  );

  return createExtractedDocument(
    normalizedText,
    path,
    _guess_mime_type(path),
    Object.freeze(metadata),
    [segment],
  );
}

function _markitdown_text(result: { text_content?: string; markdown?: string }): string {
  const text = result.text_content ?? result.markdown;
  if (!text) {
    throw new DocumentConversionError("MarkItDown returned no text content.");
  }
  return text;
}

function _guess_mime_type(path: string): string | null {
  // mime-types exports .types or .lookup
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const mime = require("mime-types");
  const lookup = (mime as Record<string, unknown>).lookup as ((path: string) => string | false) | undefined;
  if (lookup) {
    const result = lookup(path);
    return result === false ? null : result;
  }
  return null;
}

function _require_converted_text(path: string, text: string): string {
  if (!text || !text.trim()) {
    throw new DocumentConversionError(
      `Document contains no convertible text: ${path}`,
    );
  }
  return text.trim();
}
