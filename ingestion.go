package ozma

import (
	"errors"
	"fmt"
	"mime"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const (
	DefaultChunkSize    = 1500
	DefaultChunkOverlap = 200
)

type TextChunker struct {
	ChunkSize    int
	ChunkOverlap int
}

func NewTextChunker(chunkSize, chunkOverlap int) (*TextChunker, error) {
	if chunkSize == 0 {
		chunkSize = DefaultChunkSize
	}
	if chunkOverlap < 0 {
		return nil, errors.New("chunk_overlap must be greater than or equal to 0.")
	}
	if chunkSize <= 0 {
		return nil, errors.New("chunk_size must be greater than 0.")
	}
	if chunkOverlap >= chunkSize {
		return nil, errors.New("chunk_overlap must be smaller than chunk_size.")
	}
	return &TextChunker{ChunkSize: chunkSize, ChunkOverlap: chunkOverlap}, nil
}

func (c TextChunker) Chunk(document ExtractedDocument) ([]TextChunk, error) {
	if strings.TrimSpace(document.Text) == "" {
		return nil, errors.New("document text must be a non-empty string.")
	}
	runes := []rune(document.Text)
	chunks := []TextChunk{}
	start := 0
	for start < len(runes) {
		end := start + c.ChunkSize
		if end > len(runes) {
			end = len(runes)
		}
		end = moveEndToBoundary(runes, start, end)
		chunkText := string(runes[start:end])
		if strings.TrimSpace(chunkText) != "" {
			meta := cloneMetadata(document.Metadata)
			meta["chunk_index"] = len(chunks)
			meta["byte_to_char"] = BuildByteToChar(chunkText)
			chunks = append(chunks, TextChunk{
				Text:      chunkText,
				Source:    document.Source,
				StartChar: start,
				EndChar:   end,
				Metadata:  meta,
			})
		}
		if end >= len(runes) {
			break
		}
		next := end - c.ChunkOverlap
		if next < start+1 {
			next = start + 1
		}
		start = moveStartToBoundary(runes, next)
	}
	return chunks, nil
}

func ChunkDocument(document ExtractedDocument, chunkSize, chunkOverlap int) ([]TextChunk, error) {
	chunker, err := NewTextChunker(chunkSize, chunkOverlap)
	if err != nil {
		return nil, err
	}
	return chunker.Chunk(document)
}

func moveEndToBoundary(text []rune, start, end int) int {
	if end >= len(text) {
		return end
	}
	for i := end - 1; i > start; i-- {
		if text[i] == '\n' || text[i] == ' ' {
			return i
		}
	}
	return end
}

func moveStartToBoundary(text []rune, start int) int {
	for start < len(text) && strings.TrimSpace(string(text[start])) == "" {
		start++
	}
	return start
}

func BuildByteToChar(text string) []int {
	bytes := []byte(text)
	mapping := make([]int, len(bytes)+1)
	charIdx := 0
	for byteIdx := 0; byteIdx < len(bytes); {
		r := bytes[byteIdx]
		seqLen := 1
		switch {
		case r < 0x80:
			seqLen = 1
		case r < 0xE0:
			seqLen = 2
		case r < 0xF0:
			seqLen = 3
		default:
			seqLen = 4
		}
		for i := 0; i < seqLen && byteIdx+i < len(mapping); i++ {
			mapping[byteIdx+i] = charIdx
		}
		byteIdx += seqLen
		charIdx++
	}
	mapping[len(bytes)] = charIdx
	return mapping
}

type DocumentConverter interface {
	Supports(path string) bool
	Convert(path string) (ExtractedDocument, error)
}

type UnsupportedDocumentError struct{ Message string }

func (e UnsupportedDocumentError) Error() string { return e.Message }

type DocumentConversionError struct{ Message string }

func (e DocumentConversionError) Error() string { return e.Message }

type ConverterDependencyError struct{ Message string }

func (e ConverterDependencyError) Error() string { return e.Message }

type TextDocumentConverter struct {
	Encoding string
}

func (c TextDocumentConverter) Supports(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".txt" || ext == ".md" || ext == ".markdown"
}

func (c TextDocumentConverter) Convert(path string) (ExtractedDocument, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return ExtractedDocument{}, err
	}
	return documentFromText(path, string(b), Metadata{
		"converter":         "text",
		"extension":         strings.ToLower(filepath.Ext(path)),
		"normalized_format": "text",
	})
}

type MarkItDownDocumentConverter struct {
	ConvertFunc func(path string) (string, error)
	Command     string
}

func (c MarkItDownDocumentConverter) Supports(string) bool { return true }

func (c MarkItDownDocumentConverter) Convert(path string) (ExtractedDocument, error) {
	if c.ConvertFunc == nil {
		command := c.Command
		if command == "" {
			command = "markitdown"
		}
		if _, err := exec.LookPath(command); err != nil {
			return ExtractedDocument{}, ConverterDependencyError{
				Message: "Missing document converter dependency. Install the markitdown CLI or configure MarkItDownDocumentConverter.ConvertFunc to convert rich document formats.",
			}
		}
		output, err := exec.Command(command, path).Output()
		if err != nil {
			return ExtractedDocument{}, DocumentConversionError{Message: fmt.Sprintf("Could not convert document with MarkItDown: %s: %v", path, err)}
		}
		text := string(output)
		return documentFromText(path, text, Metadata{
			"converter":         "markitdown",
			"extension":         strings.ToLower(filepath.Ext(path)),
			"normalized_format": "markdown",
		})
	}
	text, err := c.ConvertFunc(path)
	if err != nil {
		return ExtractedDocument{}, DocumentConversionError{Message: fmt.Sprintf("Could not convert document with MarkItDown: %s: %v", path, err)}
	}
	return documentFromText(path, text, Metadata{
		"converter":         "markitdown",
		"extension":         strings.ToLower(filepath.Ext(path)),
		"normalized_format": "markdown",
	})
}

type ConverterRegistry struct {
	converters []DocumentConverter
}

func NewConverterRegistry(converters ...DocumentConverter) *ConverterRegistry {
	return &ConverterRegistry{converters: converters}
}

func DefaultRegistry() *ConverterRegistry {
	return NewConverterRegistry(TextDocumentConverter{}, MarkItDownDocumentConverter{})
}

func (r *ConverterRegistry) Register(converter DocumentConverter) {
	r.converters = append(r.converters, converter)
}

func (r *ConverterRegistry) Convert(path string) (ExtractedDocument, error) {
	resolved, err := resolveExistingFile(path)
	if err != nil {
		return ExtractedDocument{}, err
	}
	for _, converter := range r.converters {
		if converter.Supports(resolved) {
			return converter.Convert(resolved)
		}
	}
	ext := filepath.Ext(resolved)
	if ext == "" {
		ext = "<none>"
	}
	return ExtractedDocument{}, UnsupportedDocumentError{Message: "Unsupported document extension: " + ext}
}

func ConvertDocument(path string, registry *ConverterRegistry) (ExtractedDocument, error) {
	if registry == nil {
		registry = DefaultRegistry()
	}
	return registry.Convert(path)
}

func IngestDocument(path string, registry *ConverterRegistry, chunker *TextChunker, chunkSize, chunkOverlap int) (IngestedDocument, error) {
	document, err := ConvertDocument(path, registry)
	if err != nil {
		return IngestedDocument{}, err
	}
	if chunker == nil {
		chunker, err = NewTextChunker(chunkSize, chunkOverlap)
		if err != nil {
			return IngestedDocument{}, err
		}
	}
	chunks, err := chunker.Chunk(document)
	if err != nil {
		return IngestedDocument{}, err
	}
	return IngestedDocument{Document: document, Chunks: chunks}, nil
}

func resolveExistingFile(path string) (string, error) {
	resolved, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(resolved)
	if err != nil {
		if os.IsNotExist(err) {
			return "", os.ErrNotExist
		}
		return "", err
	}
	if info.IsDir() {
		return "", fmt.Errorf("Document path is not a file: %s", resolved)
	}
	return resolved, nil
}

func documentFromText(path, text string, metadata Metadata) (ExtractedDocument, error) {
	if strings.TrimSpace(text) == "" {
		return ExtractedDocument{}, DocumentConversionError{Message: "Document contains no convertible text: " + path}
	}
	return ExtractedDocument{
		Text:     text,
		Source:   path,
		MIMEType: mime.TypeByExtension(strings.ToLower(filepath.Ext(path))),
		Metadata: metadata,
		Segments: []DocumentSegment{{
			Text:      text,
			StartChar: 0,
			EndChar:   len([]rune(text)),
			Metadata:  Metadata{"segment_type": "body"},
		}},
	}, nil
}
