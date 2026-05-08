package ozma

import "path/filepath"

type Metadata map[string]any

type DocumentSegment struct {
	Text      string
	StartChar int
	EndChar   int
	Metadata  Metadata
}

type ExtractedDocument struct {
	Text     string
	Source   string
	MIMEType string
	Metadata Metadata
	Segments []DocumentSegment
}

func (d ExtractedDocument) SourcePath() string {
	return filepath.Clean(d.Source)
}

type TextChunk struct {
	Text      string
	Source    string
	StartChar int
	EndChar   int
	Metadata  Metadata
}

func (c TextChunk) Length() int {
	return c.EndChar - c.StartChar
}

type IngestedDocument struct {
	Document ExtractedDocument
	Chunks   []TextChunk
}

func (d IngestedDocument) Text() string {
	return d.Document.Text
}

func (d IngestedDocument) Source() string {
	return d.Document.Source
}

func cloneMetadata(m Metadata) Metadata {
	out := Metadata{}
	for k, v := range m {
		out[k] = v
	}
	return out
}
