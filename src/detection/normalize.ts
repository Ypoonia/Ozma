/**
 * Text normalization for Prompt Guard input.
 * Ported from doc_analyse/detection/normalize.py
 */

const ZERO_WIDTH_CHARS = [
  "\u200b", // zero width space
  "\u200c", // zero width non-joiner
  "\u200d", // zero width joiner
  "\ufeff", // zero width no-break space (BOM)
  "\u00ad", // soft hyphen
];

const WHITESPACE_COLLAPSE = /[ \t]+/g;
const MULTIPLE_NEWLINES = /\n{3,}/g;

export function normalize_for_detection(text: string): string {
  text = _normalize_unicode(text);
  text = _strip_zero_width_chars(text);
  text = _normalize_whitespace(text);
  return text.trim();
}

function _normalize_unicode(text: string): string {
  return text.normalize("NFKC");
}

function _strip_zero_width_chars(text: string): string {
  for (const char of ZERO_WIDTH_CHARS) {
    text = text.split(char).join("");
  }
  return text;
}

function _normalize_whitespace(text: string): string {
  text = text.replace(WHITESPACE_COLLAPSE, " ");
  text = text.replace(MULTIPLE_NEWLINES, "\n\n");
  return text;
}
