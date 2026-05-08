package ozma

type OpenAIClassifier struct {
	*BaseClassifier
	APIKey string
}

func NewOpenAIClassifier(opts ClassifierOptions) (*OpenAIClassifier, error) {
	complete := opts.Complete
	if complete == nil {
		complete = func([]ClassifierMessage, string, GenerationConfig) (string, error) {
			_, err := EnsureAPIKey("OpenAI", []string{"OPENAI_API_KEY"}, opts.APIKey)
			return "", err
		}
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
		complete = func([]ClassifierMessage, string, GenerationConfig) (string, error) {
			_, err := EnsureAPIKey("Anthropic", []string{"ANTHROPIC_API_KEY"}, opts.APIKey)
			return "", err
		}
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
		complete = func([]ClassifierMessage, string, GenerationConfig) (string, error) {
			_, err := EnsureAPIKey("Gemini", []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"}, opts.APIKey)
			return "", err
		}
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
