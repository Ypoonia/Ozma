package ozma

import (
	"embed"
	"fmt"
	"strings"
)

//go:embed src/doc_analyse/prompt/system.md src/doc_analyse/prompt/classification.md src/doc_analyse/prompt/classifieragent.md
var promptAssets embed.FS

const (
	SystemPromptFile          = "system.md"
	ClassificationPromptFile  = "classification.md"
	ClassifierAgentPromptFile = "classifieragent.md"
	TextPlaceholder           = "{{ text }}"
	MetadataPlaceholder       = "{{ metadata }}"
)

type PromptTemplateError struct{ Message string }

func (e PromptTemplateError) Error() string { return e.Message }

func LoadDefaultSystemPrompt(promptText string) (string, error) {
	return resolvePromptText(promptText, SystemPromptFile, nil)
}

func LoadDefaultClassificationPrompt(promptText string) (string, error) {
	return resolvePromptText(promptText, ClassificationPromptFile, []string{TextPlaceholder, MetadataPlaceholder})
}

func LoadClassifierAgentPrompt(promptText string) (string, error) {
	return resolvePromptText(promptText, ClassifierAgentPromptFile, nil)
}

func RenderClassificationPrompt(template, text string, metadata Metadata) (string, error) {
	if strings.TrimSpace(text) == "" {
		return "", PromptTemplateError{Message: "Classifier input text must be a non-empty string."}
	}
	out := strings.ReplaceAll(template, MetadataPlaceholder, formatMetadata(metadata))
	out = strings.ReplaceAll(out, TextPlaceholder, text)
	return strings.TrimSpace(out), nil
}

func resolvePromptText(promptText, filename string, required []string) (string, error) {
	if promptText == "" {
		b, err := promptAssets.ReadFile("src/doc_analyse/prompt/" + filename)
		if err != nil {
			return "", PromptTemplateError{Message: fmt.Sprintf("Prompt template '%s' could not be loaded.", filename)}
		}
		promptText = string(b)
	}
	promptText = strings.TrimSpace(promptText)
	if promptText == "" {
		return "", PromptTemplateError{Message: fmt.Sprintf("Prompt template '%s' is empty.", filename)}
	}
	missing := []string{}
	for _, placeholder := range required {
		if !strings.Contains(promptText, placeholder) {
			missing = append(missing, placeholder)
		}
	}
	if len(missing) > 0 {
		return "", PromptTemplateError{Message: fmt.Sprintf("Prompt template '%s' is missing required placeholder(s): %s", filename, strings.Join(missing, ", "))}
	}
	return promptText, nil
}

func formatMetadata(metadata Metadata) string {
	if len(metadata) == 0 {
		return "none"
	}
	lines := []string{}
	for k, v := range metadata {
		lines = append(lines, fmt.Sprintf("- %s: %v", k, v))
	}
	return strings.Join(lines, "\n")
}
