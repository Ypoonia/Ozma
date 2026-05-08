package ozma

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const (
	DefaultTemperature = 0.0
	DefaultMaxTokens   = 1200
)

type GenerationConfig struct {
	Temperature float64
	MaxTokens   int
}

func ResolveGenerationConfig(temperature *float64, maxTokens *int) (GenerationConfig, error) {
	t := DefaultTemperature
	if temperature != nil {
		t = *temperature
	}
	mt := DefaultMaxTokens
	if maxTokens != nil {
		mt = *maxTokens
	}
	if mt <= 0 {
		return GenerationConfig{}, fmt.Errorf("max_tokens must be greater than 0.")
	}
	return GenerationConfig{Temperature: t, MaxTokens: mt}, nil
}

type ClassifierMessage struct {
	Role    string
	Content string
}

type PromptInjectionFinding struct {
	Span       string
	AttackType string
	Severity   string
	Reason     string
	StartChar  *int
	EndChar    *int
}

type ClassificationResult struct {
	Verdict     string
	Confidence  float64
	Reasons     []string
	Findings    []PromptInjectionFinding
	RawResponse *string
}

type Classifier interface {
	Provider() string
	Classify(text string, metadata Metadata) (ClassificationResult, error)
	BuildMessages(text string, metadata Metadata) ([]ClassifierMessage, error)
	ParseResponse(rawResponse string) (ClassificationResult, error)
}

type ClassifierDependencyError struct{ Message string }

func (e ClassifierDependencyError) Error() string { return e.Message }

type ClassifierResponseError struct{ Message string }

func (e ClassifierResponseError) Error() string { return e.Message }

type CompletionFunc func(messages []ClassifierMessage, model string, config GenerationConfig) (string, error)

type BaseClassifier struct {
	ProviderName       string
	DefaultModel       string
	Model              string
	Temperature        float64
	MaxTokens          int
	SystemPrompt       string
	UserPromptTemplate string
	Complete           CompletionFunc
}

func NewBaseClassifier(providerName, defaultModel string, opts ClassifierOptions, complete CompletionFunc) (*BaseClassifier, error) {
	if complete == nil {
		complete = opts.Complete
	}
	model := opts.Model
	if model == "" {
		model = defaultModel
	}
	if model == "" {
		return nil, fmt.Errorf("A model is required for this classifier.")
	}
	config, err := ResolveGenerationConfig(opts.Temperature, opts.MaxTokens)
	if err != nil {
		return nil, err
	}
	systemPrompt, err := LoadDefaultSystemPrompt(opts.SystemPrompt)
	if err != nil {
		return nil, err
	}
	userPrompt, err := LoadDefaultClassificationPrompt(opts.UserPromptTemplate)
	if err != nil {
		return nil, err
	}
	return &BaseClassifier{
		ProviderName: providerName, DefaultModel: defaultModel, Model: model,
		Temperature: config.Temperature, MaxTokens: config.MaxTokens,
		SystemPrompt: systemPrompt, UserPromptTemplate: userPrompt, Complete: complete,
	}, nil
}

type ClassifierOptions struct {
	Model              string
	APIKey             string
	BaseURL            string
	HTTPClient         *http.Client
	Temperature        *float64
	MaxTokens          *int
	SystemPrompt       string
	UserPromptTemplate string
	Complete           CompletionFunc
}

func (c *BaseClassifier) Provider() string { return c.ProviderName }

func (c *BaseClassifier) BuildMessages(text string, metadata Metadata) ([]ClassifierMessage, error) {
	userPrompt, err := RenderClassificationPrompt(c.UserPromptTemplate, text, metadata)
	if err != nil {
		return nil, err
	}
	return []ClassifierMessage{
		{Role: "system", Content: c.SystemPrompt},
		{Role: "user", Content: userPrompt},
	}, nil
}

func (c *BaseClassifier) Classify(text string, metadata Metadata) (ClassificationResult, error) {
	messages, err := c.BuildMessages(text, metadata)
	if err != nil {
		return ClassificationResult{}, err
	}
	if c.Complete == nil {
		return ClassificationResult{}, ClassifierDependencyError{Message: "Classifier completion function is not configured."}
	}
	raw, err := c.Complete(messages, c.Model, GenerationConfig{Temperature: c.Temperature, MaxTokens: c.MaxTokens})
	if err != nil {
		return ClassificationResult{}, err
	}
	return c.ParseResponse(raw)
}

func (c *BaseClassifier) ParseResponse(rawResponse string) (ClassificationResult, error) {
	jsonText := ExtractJSONObject(rawResponse)
	var data map[string]any
	if err := json.Unmarshal([]byte(jsonText), &data); err != nil {
		return ClassificationResult{}, ClassifierResponseError{Message: "Classifier returned invalid JSON."}
	}
	result := ClassificationResultFromMap(data, rawResponse)
	return result, nil
}

func ClassificationResultFromMap(data map[string]any, rawResponse string) ClassificationResult {
	verdict := strings.ToLower(fmt.Sprint(data["verdict"]))
	if verdict != "safe" && verdict != "suspicious" && verdict != "unsafe" {
		verdict = "suspicious"
	}
	reasons := []string{}
	if rawReasons, ok := data["reasons"].([]any); ok {
		for _, item := range rawReasons {
			if s := strings.TrimSpace(fmt.Sprint(item)); s != "" {
				reasons = append(reasons, s)
			}
		}
	}
	findings := []PromptInjectionFinding{}
	if rawFindings, ok := data["findings"].([]any); ok {
		for _, item := range rawFindings {
			if m, ok := item.(map[string]any); ok {
				findings = append(findings, PromptInjectionFindingFromMap(m))
			}
		}
	}
	raw := rawResponse
	return ClassificationResult{
		Verdict: verdict, Confidence: clamp(toFloat(data["confidence"])),
		Reasons: reasons, Findings: findings, RawResponse: &raw,
	}
}

func PromptInjectionFindingFromMap(data map[string]any) PromptInjectionFinding {
	severity := strings.ToLower(fmt.Sprint(data["severity"]))
	if severity != "low" && severity != "medium" && severity != "high" && severity != "critical" {
		severity = "medium"
	}
	return PromptInjectionFinding{
		Span:       fmt.Sprint(data["span"]),
		AttackType: defaultString(data["attack_type"], "other"),
		Severity:   severity,
		Reason:     fmt.Sprint(data["reason"]),
		StartChar:  optionalInt(data["start_char"]),
		EndChar:    optionalInt(data["end_char"]),
	}
}

func ExtractJSONObject(rawResponse string) string {
	text := strings.TrimSpace(rawResponse)
	re := regexp.MustCompile("(?is)```(?:json)?\\s*(.*?)```")
	if m := re.FindStringSubmatch(text); len(m) > 1 {
		text = strings.TrimSpace(m[1])
	}
	if strings.HasPrefix(text, "{") && strings.HasSuffix(text, "}") {
		return text
	}
	start, end := strings.Index(text, "{"), strings.LastIndex(text, "}")
	if start == -1 || end == -1 || end <= start {
		return text
	}
	return text[start : end+1]
}

func RenderMessagesForSinglePrompt(messages []ClassifierMessage) string {
	parts := make([]string, len(messages))
	for i, m := range messages {
		parts[i] = strings.ToUpper(m.Role) + ":\n" + m.Content
	}
	return strings.Join(parts, "\n\n")
}

func EnsureAPIKey(providerName string, envNames []string, apiKey string) (string, error) {
	if apiKey != "" {
		return apiKey, nil
	}
	for _, name := range envNames {
		if value := os.Getenv(name); value != "" {
			return value, nil
		}
	}
	return "", ClassifierDependencyError{Message: fmt.Sprintf("Missing %s API key. Set %s or pass api_key=...", providerName, strings.Join(envNames, " or "))}
}

func RequireTextResponse(providerName, text string) (string, error) {
	if strings.TrimSpace(text) == "" {
		return "", ClassifierResponseError{Message: fmt.Sprintf("%s returned no text content.", providerName)}
	}
	return strings.TrimSpace(text), nil
}

func httpClient(client *http.Client) *http.Client {
	if client != nil {
		return client
	}
	return http.DefaultClient
}

func postJSON(client *http.Client, url string, headers map[string]string, body any) ([]byte, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	resp, err := httpClient(client).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return nil, readErr
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, ClassifierResponseError{Message: fmt.Sprintf("provider API returned %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))}
	}
	return data, nil
}

func BuildClassifier(provider string, opts ClassifierOptions) (Classifier, error) {
	key := strings.ToLower(strings.TrimSpace(provider))
	switch key {
	case "anthropic", "claude":
		return NewAnthropicClassifier(opts)
	case "codex", "openai":
		return NewOpenAIClassifier(opts)
	case "gemini", "google":
		return NewGeminiClassifier(opts)
	default:
		keys := []string{"anthropic", "claude", "codex", "gemini", "google", "openai"}
		sort.Strings(keys)
		return nil, fmt.Errorf("Unknown classifier provider '%s'. Available providers: %s", provider, strings.Join(keys, ", "))
	}
}

func ClassifierFromEnv(prefix string, opts ClassifierOptions) (Classifier, error) {
	if prefix == "" {
		prefix = "DOC_ANALYSE_LLM"
	}
	provider := os.Getenv(prefix + "_PROVIDER")
	if provider == "" {
		provider = "openai"
	}
	if opts.Model == "" {
		opts.Model = os.Getenv(prefix + "_MODEL")
	}
	if opts.APIKey == "" {
		opts.APIKey = os.Getenv(prefix + "_API_KEY")
		if opts.APIKey == "" {
			opts.APIKey = providerAPIKey(provider)
		}
	}
	return BuildClassifier(provider, opts)
}

func providerAPIKey(provider string) string {
	switch strings.ToLower(provider) {
	case "anthropic", "claude":
		return os.Getenv("ANTHROPIC_API_KEY")
	case "gemini", "google":
		if v := os.Getenv("GEMINI_API_KEY"); v != "" {
			return v
		}
		return os.Getenv("GOOGLE_API_KEY")
	default:
		return os.Getenv("OPENAI_API_KEY")
	}
}

func defaultString(value any, fallback string) string {
	if value == nil {
		return fallback
	}
	return fmt.Sprint(value)
}

func toFloat(value any) float64 {
	switch v := value.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case json.Number:
		f, _ := v.Float64()
		return f
	case string:
		f, _ := strconv.ParseFloat(v, 64)
		return f
	default:
		return 0
	}
}

func optionalInt(value any) *int {
	if value == nil {
		return nil
	}
	i := int(toFloat(value))
	return &i
}
