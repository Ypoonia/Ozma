package ozma

import (
	"context"
	"fmt"
	"sync"
	"time"
)

var RetryDelays = []time.Duration{time.Second, 2 * time.Second, 4 * time.Second}
var ChunkTimeout = 120 * time.Second

type WorkerPoolError struct{ Message string }

func (e WorkerPoolError) Error() string { return e.Message }

type WorkerResult struct {
	Chunk          TextChunk
	Classification ClassificationResult
}

type StatelessClassifierWorker struct {
	ClassifierFactory func() (Classifier, error)
	pool              sync.Pool
}

func NewStatelessClassifierWorker(factory func() (Classifier, error)) *StatelessClassifierWorker {
	worker := &StatelessClassifierWorker{ClassifierFactory: factory}
	worker.pool.New = func() any {
		classifier, err := worker.ClassifierFactory()
		if err != nil {
			return classifierPoolItem{err: err}
		}
		return classifierPoolItem{classifier: classifier}
	}
	return worker
}

func (w *StatelessClassifierWorker) ClassifyChunk(chunk TextChunk) (WorkerResult, error) {
	if chunk.Text == "" {
		return WorkerResult{}, fmt.Errorf("Worker chunk text must be a non-empty string.")
	}
	item := w.pool.Get().(classifierPoolItem)
	if item.err != nil {
		return WorkerResult{}, item.err
	}
	if item.classifier == nil {
		return WorkerResult{}, fmt.Errorf("classifier factory returned nil")
	}
	defer w.pool.Put(item)
	classification, err := item.classifier.Classify(chunk.Text, chunkClassificationMetadata(chunk))
	if err != nil {
		return WorkerResult{}, err
	}
	return WorkerResult{Chunk: chunk, Classification: classification}, nil
}

type classifierPoolItem struct {
	classifier Classifier
	err        error
}

type ClassifierWorkerPool struct {
	Worker     *StatelessClassifierWorker
	MaxWorkers int
	closed     bool
	mu         sync.Mutex
}

func NewClassifierWorkerPool(worker *StatelessClassifierWorker, maxWorkers int) *ClassifierWorkerPool {
	if maxWorkers <= 0 {
		maxWorkers = 16
	}
	return &ClassifierWorkerPool{Worker: worker, MaxWorkers: maxWorkers}
}

func (p *ClassifierWorkerPool) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closed = true
}

func (p *ClassifierWorkerPool) ClassifyChunks(chunks []TextChunk) ([]WorkerResult, error) {
	p.mu.Lock()
	closed := p.closed
	p.mu.Unlock()
	if closed {
		return nil, WorkerPoolError{Message: "Pool is closed."}
	}
	if len(chunks) == 0 {
		return []WorkerResult{}, nil
	}
	type indexedResult struct {
		index  int
		result WorkerResult
		err    error
	}
	sem := make(chan struct{}, p.MaxWorkers)
	out := make(chan indexedResult, len(chunks))
	for i, chunk := range chunks {
		sem <- struct{}{}
		go func(index int, c TextChunk) {
			defer func() { <-sem }()
			ctx, cancel := context.WithTimeout(context.Background(), ChunkTimeout)
			defer cancel()
			result, err := classifyWithRetry(ctx, p.Worker, c)
			out <- indexedResult{index: index, result: result, err: err}
		}(i, chunk)
	}
	results := make([]WorkerResult, len(chunks))
	for range chunks {
		item := <-out
		if item.err != nil {
			return nil, WorkerPoolError{Message: fmt.Sprintf("Worker failed for chunk index %d after initial attempt + %d retries: %v", item.index, len(RetryDelays), item.err)}
		}
		results[item.index] = item.result
	}
	return results, nil
}

func classifyWithRetry(ctx context.Context, worker *StatelessClassifierWorker, chunk TextChunk) (WorkerResult, error) {
	var lastErr error
	for attempt := 0; attempt <= len(RetryDelays); attempt++ {
		if attempt > 0 {
			select {
			case <-time.After(RetryDelays[attempt-1]):
			case <-ctx.Done():
				return WorkerResult{}, fmt.Errorf("timed out after %s", ChunkTimeout)
			}
		}
		resultCh := make(chan struct {
			result WorkerResult
			err    error
		}, 1)
		go func() {
			result, err := worker.ClassifyChunk(chunk)
			resultCh <- struct {
				result WorkerResult
				err    error
			}{result, err}
		}()
		select {
		case got := <-resultCh:
			if got.err == nil {
				return got.result, nil
			}
			lastErr = got.err
			if !isRetryable(got.err) {
				return WorkerResult{}, got.err
			}
		case <-ctx.Done():
			return WorkerResult{}, fmt.Errorf("timed out after %s", ChunkTimeout)
		}
	}
	return WorkerResult{}, lastErr
}

func isRetryable(err error) bool {
	_, nonRetryable := err.(PromptTemplateError)
	return !nonRetryable
}

func BuildStatelessClassifierFactory(provider, prefix, systemPrompt string, opts ClassifierOptions) func() (Classifier, error) {
	return func() (Classifier, error) {
		workerPrompt, err := LoadClassifierAgentPrompt(systemPrompt)
		if err != nil {
			return nil, err
		}
		opts.SystemPrompt = workerPrompt
		if provider != "" {
			return BuildClassifier(provider, opts)
		}
		return ClassifierFromEnv(prefix, opts)
	}
}

func BuildClassifierWorkerPool(provider, prefix, systemPrompt string, maxWorkers int, opts ClassifierOptions) (*ClassifierWorkerPool, error) {
	factory := BuildStatelessClassifierFactory(provider, prefix, systemPrompt, opts)
	return NewClassifierWorkerPool(NewStatelessClassifierWorker(factory), maxWorkers), nil
}

func chunkClassificationMetadata(chunk TextChunk) Metadata {
	metadata := cloneMetadata(chunk.Metadata)
	if _, ok := metadata["source"]; !ok {
		metadata["source"] = chunk.Source
	}
	if _, ok := metadata["start_char"]; !ok {
		metadata["start_char"] = chunk.StartChar
	}
	if _, ok := metadata["end_char"]; !ok {
		metadata["end_char"] = chunk.EndChar
	}
	return metadata
}
