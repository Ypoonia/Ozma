package ozma

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
)

type OpenAIClassifier struct {
	*BaseClassifier
	APIKey string
}

func NewOpenAIClassifier(opts ClassifierOptions) (*OpenAIClassifier, error) {
	complete := opts.Complete
	if complete == nil {
		complete = openAICompletion(opts)
	}
	base, err := NewBaseClassifier("openai", "gpt-4o-mini", opts, complete)
	if err != nil {
		return nil, err
	}
	return &OpenAIClassifier{BaseClassifier: base, APIKey: opts.APIKey}, nil
}

type AnthropicClassifier struct {
	*BaseClassifier
	APIKey string
}

func NewAnthropicClassifier(opts ClassifierOptions) (*AnthropicClassifier, error) {
	complete := opts.Complete
	if complete == nil {
		complete = anthropicCompletion(opts)
	}
	base, err := NewBaseClassifier("anthropic", "claude-3-5-haiku-latest", opts, complete)
	if err != nil {
		return nil, err
	}
	return &AnthropicClassifier{BaseClassifier: base, APIKey: opts.APIKey}, nil
}

type GeminiClassifier struct {
	*BaseClassifier
	APIKey string
}

func NewGeminiClassifier(opts ClassifierOptions) (*GeminiClassifier, error) {
	complete := opts.Complete
	if complete == nil {
		complete = geminiCompletion(opts)
	}
	base, err := NewBaseClassifier("gemini", "gemini-2.0-flash", opts, complete)
	if err != nil {
		return nil, err
	}
	return &GeminiClassifier{BaseClassifier: base, APIKey: opts.APIKey}, nil
}

type DocumentVerifier struct {
	Classifier Classifier
}

func (v DocumentVerifier) VerifyText(text string, metadata Metadata) (ClassificationResult, error) {
	return v.Classifier.Classify(text, metadata)
}

func openAICompletion(opts ClassifierOptions) CompletionFunc {
	return func(messages []ClassifierMessage, model string, config GenerationConfig) (string, error) {
		apiKey, err := EnsureAPIKey("OpenAI", []string{"OPENAI_API_KEY"}, opts.APIKey)
		if err != nil {
			return "", err
		}
		baseURL := strings.TrimRight(opts.BaseURL, "/")
		if baseURL == "" {
			baseURL = "https://api.openai.com/v1"
		}
		payloadMessages := make([]map[string]string, len(messages))
		for i, message := range messages {
			payloadMessages[i] = map[string]string{"role": message.Role, "content": message.Content}
		}
		data, err := postJSON(opts.HTTPClient, baseURL+"/chat/completions", map[string]string{
			"Authorization": "Bearer " + apiKey,
		}, map[string]any{
			"model":       model,
			"messages":    payloadMessages,
			"temperature": config.Temperature,
			"max_tokens":  config.MaxTokens,
		})
		if err != nil {
			return "", err
		}
		var response struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(data, &response); err != nil {
			return "", err
		}
		if len(response.Choices) == 0 {
			return RequireTextResponse("OpenAI", "")
		}
		return RequireTextResponse("OpenAI", response.Choices[0].Message.Content)
	}
}

func anthropicCompletion(opts ClassifierOptions) CompletionFunc {
	return func(messages []ClassifierMessage, model string, config GenerationConfig) (string, error) {
		apiKey, err := EnsureAPIKey("Anthropic", []string{"ANTHROPIC_API_KEY"}, opts.APIKey)
		if err != nil {
			return "", err
		}
		baseURL := strings.TrimRight(opts.BaseURL, "/")
		if baseURL == "" {
			baseURL = "https://api.anthropic.com"
		}
		systemParts := []string{}
		userMessages := []map[string]string{}
		for _, message := range messages {
			if message.Role == "system" {
				systemParts = append(systemParts, message.Content)
				continue
			}
			role := message.Role
			if role != "user" && role != "assistant" {
				role = "user"
			}
			userMessages = append(userMessages, map[string]string{"role": role, "content": message.Content})
		}
		body := map[string]any{
			"model":       model,
			"max_tokens":  config.MaxTokens,
			"temperature": config.Temperature,
			"messages":    userMessages,
		}
		if len(systemParts) > 0 {
			body["system"] = strings.Join(systemParts, "\n\n")
		}
		data, err := postJSON(opts.HTTPClient, baseURL+"/v1/messages", map[string]string{
			"x-api-key":         apiKey,
			"anthropic-version": "2023-06-01",
		}, body)
		if err != nil {
			return "", err
		}
		var response struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		}
		if err := json.Unmarshal(data, &response); err != nil {
			return "", err
		}
		parts := []string{}
		for _, block := range response.Content {
			if strings.TrimSpace(block.Text) != "" {
				parts = append(parts, block.Text)
			}
		}
		return RequireTextResponse("Anthropic", strings.Join(parts, ""))
	}
}

func geminiCompletion(opts ClassifierOptions) CompletionFunc {
	return func(messages []ClassifierMessage, model string, config GenerationConfig) (string, error) {
		apiKey, err := EnsureAPIKey("Gemini", []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"}, opts.APIKey)
		if err != nil {
			return "", err
		}
		baseURL := strings.TrimRight(opts.BaseURL, "/")
		if baseURL == "" {
			baseURL = "https://generativelanguage.googleapis.com"
		}
		endpoint := fmt.Sprintf("%s/v1beta/models/%s:generateContent?key=%s", baseURL, url.PathEscape(model), url.QueryEscape(apiKey))
		data, err := postJSON(opts.HTTPClient, endpoint, nil, map[string]any{
			"contents": []map[string]any{{
				"parts": []map[string]string{{"text": RenderMessagesForSinglePrompt(messages)}},
			}},
			"generationConfig": map[string]any{
				"temperature":      config.Temperature,
				"maxOutputTokens":  config.MaxTokens,
				"responseMimeType": "application/json",
			},
		})
		if err != nil {
			return "", err
		}
		var response struct {
			Candidates []struct {
				Content struct {
					Parts []struct {
						Text string `json:"text"`
					} `json:"parts"`
				} `json:"content"`
			} `json:"candidates"`
		}
		if err := json.Unmarshal(data, &response); err != nil {
			return "", err
		}
		if len(response.Candidates) == 0 || len(response.Candidates[0].Content.Parts) == 0 {
			return RequireTextResponse("Gemini", "")
		}
		return RequireTextResponse("Gemini", response.Candidates[0].Content.Parts[0].Text)
	}
}
