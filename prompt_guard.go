package ozma

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
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
	if client == nil {
		client = DefaultPromptGuardClient()
	}
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

type HuggingFacePromptGuardClient struct {
	APIKey     string
	ModelURL   string
	HTTPClient *http.Client
}

func NewHuggingFacePromptGuardClient(apiKey string) *HuggingFacePromptGuardClient {
	return &HuggingFacePromptGuardClient{
		APIKey:   apiKey,
		ModelURL: "https://api-inference.huggingface.co/models/meta-llama/Llama-Prompt-Guard-2-86M",
	}
}

func DefaultPromptGuardClient() PromptGuardClient {
	apiKey := os.Getenv("HUGGINGFACE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("HF_TOKEN")
	}
	if apiKey != "" {
		return NewHuggingFacePromptGuardClient(apiKey)
	}
	return HeuristicPromptGuardClient{}
}

func (c *HuggingFacePromptGuardClient) Score(text string) (PromptGuardScore, error) {
	if c.APIKey == "" {
		return PromptGuardScore{}, PromptGuardDependencyError{Message: "Missing Hugging Face API key. Set HUGGINGFACE_API_KEY or HF_TOKEN."}
	}
	data, err := postJSON(c.HTTPClient, c.ModelURL, map[string]string{
		"Authorization": "Bearer " + c.APIKey,
	}, map[string]any{"inputs": text})
	if err != nil {
		return PromptGuardScore{}, err
	}
	return coercePromptGuardScores(data)
}

func coercePromptGuardScores(data []byte) (PromptGuardScore, error) {
	var direct PromptGuardScore
	if err := json.Unmarshal(data, &direct); err == nil && (direct.Malicious != 0 || direct.Benign != 0) {
		return PromptGuardScore{Malicious: clamp(direct.Malicious), Benign: clamp(direct.Benign)}, nil
	}
	var rows [][]map[string]any
	if err := json.Unmarshal(data, &rows); err == nil {
		score := PromptGuardScore{Benign: 1}
		for _, group := range rows {
			for _, row := range group {
				label := fmt.Sprint(row["label"])
				value := toFloat(row["score"])
				switch label {
				case "malicious", "MALICIOUS", "jailbreak", "injection", "LABEL_1":
					if value > score.Malicious {
						score.Malicious = value
					}
				case "benign", "BENIGN", "safe", "LABEL_0":
					if value > score.Benign || score.Benign == 1 {
						score.Benign = value
					}
				}
			}
		}
		return PromptGuardScore{Malicious: clamp(score.Malicious), Benign: clamp(score.Benign)}, nil
	}
	var flat []map[string]any
	if err := json.Unmarshal(data, &flat); err != nil {
		return PromptGuardScore{}, err
	}
	score := PromptGuardScore{Benign: 1}
	for _, row := range flat {
		label := fmt.Sprint(row["label"])
		value := toFloat(row["score"])
		switch label {
		case "malicious", "MALICIOUS", "jailbreak", "injection", "LABEL_1":
			if value > score.Malicious {
				score.Malicious = value
			}
		case "benign", "BENIGN", "safe", "LABEL_0":
			score.Benign = value
		}
	}
	return PromptGuardScore{Malicious: clamp(score.Malicious), Benign: clamp(score.Benign)}, nil
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
