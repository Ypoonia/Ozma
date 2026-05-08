// Package ozma provides a Go implementation of Ozma's document ingestion,
// prompt-injection detection, routing, classifier, worker-pool, and
// orchestration APIs.
//
// The package mirrors the Python main-branch library with Go-style exported
// types and constructors. Start with ConvertDocument or IngestDocument for
// document preparation, NewYaraDetector and NewPromptGuardDetector for Layer 1
// detection, NewCheapRouter for signal fusion, and BuildOrchestrator for the
// full document-analysis pipeline.
package ozma
