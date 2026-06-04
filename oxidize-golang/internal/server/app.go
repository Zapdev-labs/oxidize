package server

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"path/filepath"
	"strings"
	"sync"

	"github.com/Zapdev-labs/oxidize/golang/internal/auth"
	"github.com/Zapdev-labs/oxidize/golang/internal/generate"
	"github.com/Zapdev-labs/oxidize/golang/internal/serviceinfo"
)

type application struct {
	models           []serviceinfo.ModelInfo
	modelID          map[string]struct{}
	defaultModelPath string
	loader           generate.LoaderConfig
	defaultMaxTokens int
	defaultTemp      float32
	defaultTopP      float32
	defaultTopK      int
	mu               sync.Mutex
	metrics          serviceinfo.MetricsData
}

func NewHandler(cfg Config) (http.Handler, error) {
	cfg = cfg.withDefaults()
	models, err := serviceinfo.DiscoverModels(cfg.ModelsDir)
	if err != nil {
		return nil, fmt.Errorf("discover models: %w", err)
	}
	app := &application{
		models:           models,
		modelID:          make(map[string]struct{}, len(models)),
		defaultModelPath: cfg.DefaultModel,
		loader:           cfg.Loader,
		defaultMaxTokens: cfg.MaxTokens,
		defaultTemp:      cfg.Temperature,
		defaultTopP:      cfg.TopP,
		defaultTopK:      cfg.TopK,
	}
	for _, model := range models {
		app.modelID[model.ID] = struct{}{}
	}
	if cfg.DefaultModel != "" {
		id := modelIDFromPath(cfg.DefaultModel)
		if id != "" {
			app.modelID[id] = struct{}{}
			found := false
			for _, m := range models {
				if m.Path == cfg.DefaultModel {
					found = true
					break
				}
			}
			if !found {
				app.models = append(app.models, serviceinfo.ModelInfo{ID: id, Path: cfg.DefaultModel})
			}
		}
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", app.health)
	mux.HandleFunc("/livez", app.health)
	mux.HandleFunc("/readyz", app.health)
	mux.HandleFunc("/metrics", app.metricsHandler)
	mux.HandleFunc("/openapi.json", app.openapi)
	mux.HandleFunc("/v1/models", app.modelsHandler)
	mux.HandleFunc("/v1/chat/completions", app.chatCompletions)
	mux.HandleFunc("/v1/completions", app.completions)
	mux.HandleFunc("/v1/embeddings", app.embeddings)
	mux.HandleFunc("/v1/realtime", app.realtime)
	mux.HandleFunc("/v1/mesh/chat/completions", app.meshChatCompletions)
	return app.instrument(auth.Middleware(mux)), nil
}

func Listen(ctx context.Context, cfg Config) error {
	handler, err := NewHandler(cfg)
	if err != nil {
		return err
	}
	cfg = cfg.withDefaults()
	listener, err := net.Listen("tcp", fmt.Sprintf("%s:%d", cfg.Host, cfg.Port))
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	defer func() { _ = listener.Close() }()

	server := &http.Server{Handler: handler}
	shutdownDone := make(chan error, 1)
	go func() {
		<-ctx.Done()
		shutdownDone <- server.Shutdown(context.Background())
	}()

	serveErr := server.Serve(listener)
	if serveErr == http.ErrServerClosed {
		return <-shutdownDone
	}
	return serveErr
}

func modelIDFromPath(path string) string {
	base := filepath.Base(path)
	return strings.TrimSuffix(base, filepath.Ext(base))
}

func (a *application) instrument(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		a.mu.Lock()
		a.metrics.RequestsTotal++
		a.metrics.RequestsInflight++
		a.mu.Unlock()
		defer func() {
			a.mu.Lock()
			a.metrics.RequestsInflight--
			a.mu.Unlock()
		}()
		next.ServeHTTP(w, r)
	})
}
