package ozma

import (
	"fmt"
	"regexp"
)

const (
	DefaultPromptGuardModel   = "meta-llama/Llama-Prompt-Guard-2-86M"
	DefaultMaliciousThreshold = 0.80
	DefaultUncertainThreshold = 0.50
)

type PromptGuardDependencyError struct{ Message string }

func (e PromptGuardDependencyError) Error() string { return e.Message }

type PromptGuardClient interface {
	Score(text string) (PromptGuardScore, error)
}

type PromptGuardScore struct {
	Malicious float64
	Benign    float64
}

type PromptGuardDetector struct {
	Model              string
	Client             PromptGuardClient
	MaliciousThreshold float64
	UncertainThreshold float64
}

func NewPromptGuardDetector(client PromptGuardClient) (*PromptGuardDetector, error) {
	return NewPromptGuardDetectorWithOptions(client, DefaultPromptGuardModel, DefaultMaliciousThreshold, DefaultUncertainThreshold)
}

func NewPromptGuardDetectorWithOptions(client PromptGuardClient, model string, maliciousThreshold, uncertainThreshold float64) (*PromptGuardDetector, error) {
	if uncertainThreshold < 0 || maliciousThreshold < uncertainThreshold || maliciousThreshold > 1 {
		return nil, fmt.Errorf("thresholds must satisfy 0 <= uncertain_threshold <= malicious_threshold <= 1.")
	}
	if model == "" {
		model = DefaultPromptGuardModel
	}
	return &PromptGuardDetector{Model: model, Client: client, MaliciousThreshold: maliciousThreshold, UncertainThreshold: uncertainThreshold}, nil
}

func (d *PromptGuardDetector) Detect(chunk TextChunk) ([]DetectionFinding, error) {
	if chunk.Text == "" {
		return nil, nil
	}
	score, err := d.RawScore(chunk.Text)
	if err != nil {
		return nil, err
	}
	meta := Metadata{"detector": "PromptGuardDetector", "model": d.Model, "pg_malicious": score.Malicious, "pg_benign": score.Benign}
	if score.Malicious >= d.MaliciousThreshold {
		v := score.Malicious
		return []DetectionFinding{BuildFinding(chunk, chunk.Text, "prompt_guard_malicious", "high", "Prompt Guard classified this chunk as malicious.", "prompt_guard", chunk.StartChar, chunk.EndChar, &v, true, meta)}, nil
	}
	if score.Malicious >= d.UncertainThreshold {
		v := score.Malicious
		return []DetectionFinding{BuildFinding(chunk, chunk.Text, "prompt_guard_uncertain", "medium", "Prompt Guard score is uncertain enough to require LLM validation.", "prompt_guard", chunk.StartChar, chunk.EndChar, &v, true, meta)}, nil
	}
	return nil, nil
}

func (d *PromptGuardDetector) RawScore(text string) (PromptGuardScore, error) {
	if d == nil || d.Client == nil {
		return PromptGuardScore{}, PromptGuardDependencyError{Message: "Prompt Guard client not configured. Provide a PromptGuardClient."}
	}
	score, err := d.Client.Score(text)
	if err != nil {
		return PromptGuardScore{}, PromptGuardDependencyError{Message: "Prompt Guard scoring failed: " + err.Error()}
	}
	score.Malicious = clamp(score.Malicious)
	score.Benign = clamp(score.Benign)
	return score, nil
}

type HeuristicPromptGuardClient struct{}

var injectionKeywords = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\b(ignore|disregard|forget)\s+(all\s+)?previous\s+instructions?\b`),
	regexp.MustCompile(`(?i)\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\b`),
	regexp.MustCompile(`(?i)\b(safe|benign|harmless)\s+(to\s+)?(forward|process|execute)\b`),
	regexp.MustCompile(`(?i)\{[\s\S]*system\s*[\s\S]*instruction[\s\S]*\}`),
	regexp.MustCompile(`(?i)\<\|.*?\|>?\s*system\s*\<`),
}

func (HeuristicPromptGuardClient) Score(text string) (PromptGuardScore, error) {
	normalized := NormalizeForDetection(text)
	score := 0.0
	for _, re := range injectionKeywords {
		if re.MatchString(normalized) {
			score += 0.15
		}
	}
	score = clamp(score)
	return PromptGuardScore{Malicious: score, Benign: 1 - score}, nil
}

func clamp(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
