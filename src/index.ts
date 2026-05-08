/**
 * Ozma — Prompt injection detection library.
 *
 * Two-layer detection:
 *   Layer 1 (cheap): YARA + Prompt Guard ML → SAFE / REVIEW / HOLD
 *   Layer 2 (LLM):   Only REVIEW/HOLD chunks → LLM classifier verdict
 */

// Ingestion
export {
  createDocumentSegment,
  createExtractedDocument,
  createTextChunk,
  createIngestedDocument,
  DocumentSegment,
  ExtractedDocument,
  TextChunk,
  IngestedDocument,
  TextChunker,
  ConverterRegistry,
  TextDocumentConverter,
  MarkItDownDocumentConverter,
  default_registry,
  convert_document,
  ingest_document,
  DocumentConversionError,
  ConverterDependencyError,
  UnsupportedDocumentError,
  DEFAULT_CHUNK_SIZE,
  DEFAULT_CHUNK_OVERLAP,
  chunk_document,
} from "./ingestion/index.js";

// Detection
export {
  DetectionFinding,
  createDetectionFinding,
  CheapRouter,
  CheapChunkDecision,
  YaraEvidence,
  createYaraEvidence,
  yaraEvidenceFromFinding,
  DECISION_SAFE,
  DECISION_REVIEW,
  DECISION_HOLD,
  normalize_for_detection,
  YaraDetector,
  YaraGlossaryError,
  PromptGuardDetector,
  HuggingFacePromptGuardClient,
  HeuristicPromptGuardClient,
  PromptGuardDependencyError,
  BaseDetector,
  ParallelDetector,
} from "./detection/index.js";

// Classifiers
export {
  BaseClassifier,
  ClassificationResult,
  PromptInjectionFinding,
  ClassifierMessage,
  ClassifierDependencyError,
  ClassifierResponseError,
  ClassifierPromptError,
  createClassificationResult,
  createPromptInjectionFinding,
  AnthropicClassifier,
  OpenAIClassifier,
  GeminiClassifier,
  build_classifier,
  classifier_from_env,
  GenerationConfig,
  resolve_generation_config,
  DEFAULT_TEMPERATURE,
  DEFAULT_MAX_TOKENS,
} from "./classifiers/index.js";

// Orchestration
export {
  DocumentOrchestrator,
  DocumentOrchestratorAsync,
  DocumentAnalysisResult,
  ChunkAnalysisResult,
  run_layer1,
  run_layer1_async,
  build_orchestrator,
  build_orchestrator_async,
  analyze_document_path,
  DocumentVerifier,
  VERDICT_SAFE,
  VERDICT_SUSPICIOUS,
  VERDICT_UNSAFE,
} from "./orchestration/index.js";

// Workers
export {
  ClassifierWorkerPool,
  StatelessClassifierWorker,
  WorkerResult,
  WorkerPoolError,
  build_classifier_worker_pool,
  build_stateless_classifier_factory,
  RETRY_DELAYS,
  CHUNK_TIMEOUT_MS,
} from "./workers/index.js";

// Prompt
export {
  PromptTemplateError,
  load_default_system_prompt,
  load_default_classification_prompt,
  load_classifier_agent_prompt,
  render_classification_prompt,
  registerPromptLoader,
} from "./prompt/index.js";
