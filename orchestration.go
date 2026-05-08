package ozma

import (
	"fmt"
	"strings"
)

const (
	VerdictSafe       = "safe"
	VerdictSuspicious = "suspicious"
	VerdictUnsafe     = "unsafe"
)

func RunLayer1(chunk TextChunk, yara *YaraDetector, pg *PromptGuardDetector, router CheapRouter) (CheapChunkDecision, []DetectionFinding, error) {
	yaraFindings, err := yara.Detect(chunk)
	if err != nil {
		return CheapChunkDecision{}, nil, err
	}
	pgScore := 0.0
	if pg != nil {
		if score, err := pg.RawScore(NormalizeForDetection(chunk.Text)); err == nil {
			pgScore = score.Malicious
		}
	}
	decision, err := router.Route(yaraFindings, pgScore)
	if err != nil {
		return CheapChunkDecision{}, nil, err
	}
	requiresLLM := decision.RequiresLayer2()
	findings := []DetectionFinding{}
	for _, e := range decision.Findings {
		var score *float64
		if e.Weight > 0 {
			v := e.Weight / 100
			if v > 1 {
				v = 1
			}
			score = &v
		}
		findings = append(findings, DetectionFinding{
			Span: e.Span, Category: e.Category, Severity: e.Severity,
			Reason:    fmt.Sprintf("[YARA] %s - %s (%s)", e.RuleID, e.Category, e.Severity),
			StartChar: e.StartChar, EndChar: e.EndChar, Source: chunk.Source, RuleID: e.RuleID,
			RequiresLLMValidation: requiresLLM, Score: score,
			Metadata: Metadata{"detector": "YaraDetector", "yara_rule": e.RuleID, "yara_weight": e.Weight, "route_hint": e.RouteHint},
		})
	}
	if pgScore > 0 && len(yaraFindings) == 0 && decision.RequiresLayer2() {
		v := pgScore
		strength := "moderate"
		if pgScore >= 0.75 {
			strength = "strong"
		}
		findings = append(findings, DetectionFinding{
			Span: "", Category: "prompt_guard_signal", Severity: "high",
			Reason:    fmt.Sprintf("[PG] score=%.3f - %s signal", pgScore, strength),
			StartChar: chunk.StartChar, EndChar: chunk.StartChar, Source: chunk.Source,
			RuleID: "prompt_guard", RequiresLLMValidation: requiresLLM, Score: &v,
			Metadata: Metadata{"detector": "PromptGuardDetector", "pg_malicious": pgScore},
		})
	}
	return decision, findings, nil
}

type ChunkAnalysisResult struct {
	ChunkIndex        int
	Chunk             TextChunk
	CheapFindings     []DetectionFinding
	CheapDecision     CheapChunkDecision
	RoutedToLLM       bool
	LLMClassification *ClassificationResult
	FinalVerdict      string
}

type DocumentAnalysisResult struct {
	IngestedDocument IngestedDocument
	ChunkResults     []ChunkAnalysisResult
	Verdict          string
	Reasons          []string
}

func (r DocumentAnalysisResult) ChunkResult(index int) ChunkAnalysisResult {
	return r.ChunkResults[index]
}

func (r DocumentAnalysisResult) ChunkText(index int) string {
	chunk := r.ChunkResults[index].Chunk
	runes := []rune(r.IngestedDocument.Document.Text)
	return string(runes[chunk.StartChar:chunk.EndChar])
}

type DocumentOrchestrator struct {
	Yara       *YaraDetector
	PG         *PromptGuardDetector
	Router     CheapRouter
	WorkerPool *ClassifierWorkerPool
	closed     bool
}

func (o *DocumentOrchestrator) Close() {
	if o.closed {
		return
	}
	if o.WorkerPool != nil {
		o.WorkerPool.Close()
	}
	o.closed = true
}

func (o *DocumentOrchestrator) AnalyzePath(path string, registry *ConverterRegistry, chunker *TextChunker) (DocumentAnalysisResult, error) {
	ingested, err := IngestDocument(path, registry, chunker, DefaultChunkSize, DefaultChunkOverlap)
	if err != nil {
		return DocumentAnalysisResult{}, err
	}
	return o.AnalyzeIngested(ingested)
}

func (o *DocumentOrchestrator) AnalyzeIngested(ingested IngestedDocument) (DocumentAnalysisResult, error) {
	chunkDecisions := make([]CheapChunkDecision, len(ingested.Chunks))
	chunkFindings := make([][]DetectionFinding, len(ingested.Chunks))
	routedIndices := []int{}
	for i, chunk := range ingested.Chunks {
		decision, findings, err := RunLayer1(chunk, o.Yara, o.PG, o.Router)
		if err != nil {
			return DocumentAnalysisResult{}, err
		}
		chunkDecisions[i] = decision
		chunkFindings[i] = findings
		if decision.RequiresLayer2() {
			routedIndices = append(routedIndices, i)
		}
	}
	llmResults := map[int]ClassificationResult{}
	if len(routedIndices) > 0 {
		if o.WorkerPool == nil {
			return DocumentAnalysisResult{}, fmt.Errorf("worker pool is required for REVIEW/HOLD chunks")
		}
		routedChunks := make([]TextChunk, len(routedIndices))
		for i, idx := range routedIndices {
			routedChunks[i] = ingested.Chunks[idx]
		}
		workerResults, err := o.WorkerPool.ClassifyChunks(routedChunks)
		if err != nil {
			return DocumentAnalysisResult{}, err
		}
		for i, idx := range routedIndices {
			llmResults[idx] = workerResults[i].Classification
		}
	}
	routed := map[int]bool{}
	for _, idx := range routedIndices {
		routed[idx] = true
	}
	chunkResults := make([]ChunkAnalysisResult, len(ingested.Chunks))
	for i, chunk := range ingested.Chunks {
		finalVerdict := VerdictSafe
		var llm *ClassificationResult
		if result, ok := llmResults[i]; ok {
			copy := result
			llm = &copy
			finalVerdict = normalizeVerdict(result.Verdict)
		} else if chunkDecisions[i].RequiresLayer2() {
			finalVerdict = VerdictSuspicious
		}
		chunkResults[i] = ChunkAnalysisResult{
			ChunkIndex: i, Chunk: chunk, CheapFindings: chunkFindings[i], CheapDecision: chunkDecisions[i],
			RoutedToLLM: routed[i], LLMClassification: llm, FinalVerdict: finalVerdict,
		}
	}
	verdict, reasons := aggregateDocumentVerdict(chunkResults)
	return DocumentAnalysisResult{IngestedDocument: ingested, ChunkResults: chunkResults, Verdict: verdict, Reasons: reasons}, nil
}

func BuildOrchestrator(yara *YaraDetector, pg *PromptGuardDetector, router CheapRouter, workerPool *ClassifierWorkerPool) (*DocumentOrchestrator, error) {
	var err error
	if yara == nil {
		yara, err = NewYaraDetector()
		if err != nil {
			return nil, err
		}
	}
	if router == (CheapRouter{}) {
		router = NewCheapRouter()
	}
	return &DocumentOrchestrator{Yara: yara, PG: pg, Router: router, WorkerPool: workerPool}, nil
}

func AnalyzeDocumentPath(path string, yara *YaraDetector, pg *PromptGuardDetector, router CheapRouter, workerPool *ClassifierWorkerPool, registry *ConverterRegistry, chunker *TextChunker, closeWorkerPool bool) (DocumentAnalysisResult, error) {
	orchestrator, err := BuildOrchestrator(yara, pg, router, workerPool)
	if err != nil {
		return DocumentAnalysisResult{}, err
	}
	if closeWorkerPool {
		defer orchestrator.Close()
	}
	return orchestrator.AnalyzePath(path, registry, chunker)
}

func aggregateDocumentVerdict(results []ChunkAnalysisResult) (string, []string) {
	unsafe := []string{}
	suspicious := []string{}
	for _, result := range results {
		if result.FinalVerdict == VerdictUnsafe {
			unsafe = append(unsafe, fmt.Sprint(result.ChunkIndex))
		}
		if result.FinalVerdict == VerdictSuspicious {
			suspicious = append(suspicious, fmt.Sprint(result.ChunkIndex))
		}
	}
	if len(unsafe) > 0 {
		return VerdictUnsafe, []string{"Unsafe chunk indices: [" + strings.Join(unsafe, ", ") + "]"}
	}
	if len(suspicious) > 0 {
		return VerdictSuspicious, []string{"Suspicious chunk indices: [" + strings.Join(suspicious, ", ") + "]"}
	}
	return VerdictSafe, []string{"No suspicious or unsafe chunks detected."}
}

func normalizeVerdict(raw string) string {
	verdict := strings.ToLower(strings.TrimSpace(raw))
	if verdict == VerdictSafe || verdict == VerdictSuspicious || verdict == VerdictUnsafe {
		return verdict
	}
	return VerdictSuspicious
}
