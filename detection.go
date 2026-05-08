package ozma

import (
	"fmt"
	"sort"
	"strings"
)

const (
	DecisionSafe   = "safe"
	DecisionReview = "review"
	DecisionHold   = "hold"
)

type DetectionFinding struct {
	Span                  string
	Category              string
	Severity              string
	Reason                string
	StartChar             int
	EndChar               int
	Source                string
	RuleID                string
	RequiresLLMValidation bool
	Score                 *float64
	Metadata              Metadata
}

func (f DetectionFinding) Length() int { return f.EndChar - f.StartChar }

type Detector interface {
	Detect(chunk TextChunk) ([]DetectionFinding, error)
}

type ParallelDetector struct {
	Detectors []Detector
}

func (d ParallelDetector) Detect(chunk TextChunk) ([]DetectionFinding, error) {
	return d.DetectMany([]TextChunk{chunk})
}

func (d ParallelDetector) DetectMany(chunks []TextChunk) ([]DetectionFinding, error) {
	findings := []DetectionFinding{}
	for _, chunk := range chunks {
		for _, detector := range d.Detectors {
			got, err := detector.Detect(chunk)
			if err != nil {
				findings = append(findings, detectorErrorFinding(chunk, detector, err))
				continue
			}
			findings = append(findings, got...)
		}
	}
	return finalizeFindings(findings), nil
}

func BuildFinding(chunk TextChunk, span, category, severity, reason, ruleID string, startChar, endChar int, score *float64, requiresLLM bool, metadata Metadata) DetectionFinding {
	resolved := cloneMetadata(chunk.Metadata)
	for k, v := range metadata {
		resolved[k] = v
	}
	return DetectionFinding{
		Span:                  span,
		Category:              category,
		Severity:              severity,
		Reason:                reason,
		StartChar:             startChar,
		EndChar:               endChar,
		Source:                chunk.Source,
		RuleID:                ruleID,
		RequiresLLMValidation: requiresLLM,
		Score:                 score,
		Metadata:              resolved,
	}
}

func finalizeFindings(findings []DetectionFinding) []DetectionFinding {
	seen := map[string]bool{}
	out := []DetectionFinding{}
	for _, f := range findings {
		key := fmt.Sprintf("%s|%s|%d|%d", f.RuleID, f.Source, f.StartChar, f.EndChar)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, f)
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.StartChar != b.StartChar {
			return a.StartChar < b.StartChar
		}
		if a.EndChar != b.EndChar {
			return a.EndChar < b.EndChar
		}
		return a.RuleID < b.RuleID
	})
	return out
}

func detectorErrorFinding(chunk TextChunk, detector Detector, err error) DetectionFinding {
	name := fmt.Sprintf("%T", detector)
	return BuildFinding(chunk, chunk.Text, "detector_error", "medium", name+" failed; send this chunk to LLM validation.", strings.ToLower(name)+"_error", chunk.StartChar, chunk.EndChar, nil, true, Metadata{
		"detector": name,
		"error":    err.Error(),
	})
}

type YaraEvidence struct {
	RuleID    string
	Category  string
	Severity  string
	Span      string
	StartChar int
	EndChar   int
	Weight    float64
	RouteHint string
}

func YaraEvidenceFromFinding(f DetectionFinding) YaraEvidence {
	weight, _ := f.Metadata["yara_weight"].(float64)
	if n, ok := f.Metadata["yara_weight"].(int); ok {
		weight = float64(n)
	}
	routeHint, _ := f.Metadata["route_hint"].(string)
	if routeHint == "" {
		routeHint = "evidence"
	}
	return YaraEvidence{
		RuleID: f.RuleID, Category: f.Category, Severity: f.Severity, Span: f.Span,
		StartChar: f.StartChar, EndChar: f.EndChar, Weight: weight, RouteHint: routeHint,
	}
}

type CheapChunkDecision struct {
	Decision  string
	RiskScore float64
	PGScore   float64
	YaraScore float64
	Findings  []YaraEvidence
	Reason    string
}

func (d CheapChunkDecision) RequiresLayer2() bool {
	return d.Decision == DecisionReview || d.Decision == DecisionHold
}

type CheapRouter struct {
	YaraReviewThreshold float64
	YaraHoldThreshold   float64
	PGReviewThreshold   float64
	PGHoldThreshold     float64
	YaraWeight          float64
	PGWeight            float64
}

func NewCheapRouter() CheapRouter {
	return CheapRouter{15, 40, 0.40, 0.75, 0.50, 0.50}
}

func (r CheapRouter) validate() error {
	if r.YaraReviewThreshold == 0 && r.YaraHoldThreshold == 0 && r.PGReviewThreshold == 0 && r.PGHoldThreshold == 0 && r.YaraWeight == 0 && r.PGWeight == 0 {
		return nil
	}
	if r.YaraWeight < 0 || r.YaraWeight > 1 || r.PGWeight < 0 || r.PGWeight > 1 {
		return fmt.Errorf("Weights must be between 0 and 1")
	}
	if r.YaraWeight == 0 && r.PGWeight == 0 {
		return fmt.Errorf("At least one of yara_weight or pg_weight must be non-zero")
	}
	if r.YaraReviewThreshold < 0 || r.YaraReviewThreshold > r.YaraHoldThreshold {
		return fmt.Errorf("yara_review_threshold must be <= yara_hold_threshold")
	}
	if r.PGReviewThreshold < 0 || r.PGReviewThreshold > r.PGHoldThreshold {
		return fmt.Errorf("pg_review_threshold must be <= pg_hold_threshold")
	}
	if r.YaraHoldThreshold > 100 {
		return fmt.Errorf("yara_hold_threshold cannot exceed 100.0 (YARA score max)")
	}
	if r.PGHoldThreshold > 1 {
		return fmt.Errorf("pg_hold_threshold cannot exceed 1.0 (PG score max)")
	}
	return nil
}

func (r CheapRouter) Route(yaraFindings []DetectionFinding, pgScore float64) (CheapChunkDecision, error) {
	if r == (CheapRouter{}) {
		r = NewCheapRouter()
	}
	if err := r.validate(); err != nil {
		return CheapChunkDecision{}, err
	}
	evidence := make([]YaraEvidence, 0, len(yaraFindings))
	for _, f := range yaraFindings {
		evidence = append(evidence, YaraEvidenceFromFinding(f))
	}
	if pgScore < 0 {
		pgScore = 0
	}
	if pgScore > 1 {
		pgScore = 1
	}
	yaraScore := computeYaraScore(evidence)
	riskScore := yaraScore*r.YaraWeight + pgScore*100*r.PGWeight
	if combo := checkCategoryCombinationRules(evidence, pgScore); combo != "" {
		return CheapChunkDecision{combo, riskScore, pgScore, yaraScore, evidence, buildReason(yaraScore, pgScore, evidence, false, false)}, nil
	}
	hasHoldHint, hasReviewHint := false, false
	for _, e := range evidence {
		hasHoldHint = hasHoldHint || e.RouteHint == "hold"
		hasReviewHint = hasReviewHint || e.RouteHint == "review"
	}
	yaraStrong := r.YaraWeight > 0 && yaraScore >= r.YaraHoldThreshold
	yaraModerate := r.YaraWeight > 0 && yaraScore >= r.YaraReviewThreshold
	pgStrong := r.PGWeight > 0 && pgScore >= r.PGHoldThreshold
	pgModerate := r.PGWeight > 0 && pgScore >= r.PGReviewThreshold

	decision := DecisionSafe
	reason := fmt.Sprintf("YARA=%.0f, PG=%.2f - both signals weak.", yaraScore, pgScore)
	switch {
	case yaraStrong || pgStrong:
		decision = DecisionHold
		reason = buildReason(yaraScore, pgScore, evidence, yaraStrong, pgStrong)
	case yaraModerate || pgModerate || riskScore >= 20 || hasHoldHint || hasReviewHint:
		decision = DecisionReview
		reason = buildReason(yaraScore, pgScore, evidence, yaraModerate, pgModerate)
	case riskScore >= 10:
		decision = DecisionReview
		reason = buildReason(yaraScore, pgScore, evidence, false, false)
	}
	return CheapChunkDecision{decision, riskScore, pgScore, yaraScore, evidence, reason}, nil
}

func computeYaraScore(evidence []YaraEvidence) float64 {
	weights := map[string]float64{"critical": 40, "high": 25, "medium": 10, "low": 5}
	best := map[string]float64{}
	for _, e := range evidence {
		w := e.Weight
		if w <= 0 {
			w = weights[strings.ToLower(strings.TrimSpace(e.Severity))]
			if w == 0 {
				w = 10
			}
		}
		if w > best[e.Category] {
			best[e.Category] = w
		}
	}
	total := 0.0
	for _, w := range best {
		total += w
	}
	if total > 100 {
		return 100
	}
	return total
}

func checkCategoryCombinationRules(evidence []YaraEvidence, pgScore float64) string {
	cats := map[string]bool{}
	for _, e := range evidence {
		cats[e.Category] = true
	}
	has := func(names ...string) bool {
		for _, n := range names {
			if !cats[n] {
				return false
			}
		}
		return true
	}
	switch {
	case has("tool_hijack", "instruction_override"):
		return DecisionHold
	case has("secret_exfiltration", "instruction_override"):
		return DecisionHold
	case has("secret_exfiltration", "tool_hijack"):
		return DecisionHold
	case len(cats) == 1 && cats["topic_mention"] && pgScore < 0.10:
		return DecisionSafe
	default:
		return ""
	}
}

func buildReason(yaraScore, pgScore float64, evidence []YaraEvidence, yaraSignal, pgSignal bool) string {
	parts := []string{fmt.Sprintf("YARA=%.0f, PG=%.2f", yaraScore, pgScore)}
	if len(evidence) > 0 {
		idsMap := map[string]bool{}
		for _, e := range evidence {
			idsMap[e.RuleID] = true
		}
		ids := make([]string, 0, len(idsMap))
		for id := range idsMap {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		parts = append(parts, "YARA hits: "+strings.Join(ids, ", "))
	}
	if yaraSignal {
		parts = append(parts, "YARA strong")
	}
	if pgSignal {
		parts = append(parts, "PG strong")
	}
	return strings.Join(parts, " | ")
}
