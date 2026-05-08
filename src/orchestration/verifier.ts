/**
 * Document verifier — simple wrapper around a classifier.
 * Ported from doc_analyse/verifier.py
 */

import { BaseClassifier, ClassificationResult } from "../classifiers/base.js";

export class DocumentVerifier {
  private readonly classifier: BaseClassifier;

  constructor(classifier: BaseClassifier) {
    this.classifier = classifier;
  }

  verify_text(
    text: string,
    metadata: Record<string, unknown> | null = null,
  ): ClassificationResult {
    return this.classifier.classify(text, metadata);
  }
}
