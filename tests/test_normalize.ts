import { describe, expect, it } from "vitest";
import { normalize_for_detection } from "../src/detection/normalize.js";

describe("normalize_for_detection", () => {
  it("strips zero-width characters and BOM variants", () => {
    expect(normalize_for_detection("a\u200bb\u200cc\u200dd\ufeffe\u00adf")).toBe("abcdef");
  });

  it("applies NFKC normalization", () => {
    expect(normalize_for_detection("km²")).toBe("km2");
    const fraction = normalize_for_detection("½");
    expect(fraction.includes("1")).toBe(true);
    expect(fraction.includes("2")).toBe(true);
  });

  it("collapses spaces and tabs without flattening single newlines", () => {
    expect(normalize_for_detection("hello    world")).toBe("hello world");
    expect(normalize_for_detection("hello\t\tworld")).toBe("hello world");
    expect(normalize_for_detection("line1\nline2\nline3")).toBe("line1\nline2\nline3");
  });

  it("collapses 3+ newlines to two", () => {
    expect(normalize_for_detection("hello\n\n\n\nworld")).toBe("hello\n\nworld");
  });

  it("trims leading and trailing whitespace", () => {
    expect(normalize_for_detection("  \u200bhello\u200c  ")).toBe("hello");
    expect(normalize_for_detection("")).toBe("");
  });

  it("preserves regular CJK text", () => {
    expect(normalize_for_detection("こんにちは世界")).toBe("こんにちは世界");
  });
});
