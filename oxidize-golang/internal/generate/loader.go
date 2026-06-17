package generate

import (
	"fmt"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/backends"
	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
	"github.com/Zapdev-labs/oxidize/golang/core/model"
)

// LoaderConfig mirrors CLI / server loader options applied when opening GGUF weights.
type LoaderConfig struct {
	Backend       string
	Threads       int
	ContextSize   int
	NGPULayers    int
	GPUs          int
	Parallelism   string
	HFFilename    string
	HFCacheDir    string
	AllowFallback bool
}

// LoaderResult contains a loaded model and optional backend selection notes.
type LoaderResult struct {
	Model    model.Model
	Warning  string
	Backend  string
	FellBack bool
}

// SelectBackend resolves the requested backend name.
func SelectBackend(name string, allowFallback bool) (string, string, bool, error) {
	res, err := backends.NewComputeBackend(name, allowFallback)
	if err != nil {
		return "", "", false, err
	}
	return res.Effective.String(), res.Warning, res.FellBack, nil
}

// LoadModelFromPath loads a GGUF model with loader options and backend selection.
func LoadModelFromPath(path string, cfg LoaderConfig) (LoaderResult, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return LoaderResult{}, fmt.Errorf("generate: empty model path")
	}
	allow := cfg.AllowFallback
	if !allow {
		allow = true
	}
	effective, warn, fellBack, err := SelectBackend(cfg.Backend, allow)
	if err != nil {
		return LoaderResult{}, err
	}
	loaderCfg := model.NewLoaderConfig()
	loaderCfg.HFFilename = cfg.HFFilename
	loaderCfg.HFCacheDir = cfg.HFCacheDir
	loaderCfg.Backend = effective
	loaderCfg.Threads = cfg.Threads
	loaderCfg.ContextSize = cfg.ContextSize
	loaderCfg.NGPULayers = cfg.NGPULayers
	loaderCfg.GPUs = cfg.GPUs
	loaderCfg.Parallelism = cfg.Parallelism
	loaderCfg.AllowFallback = allow

	loaded, err := model.LoadGGUFModelFromPath(path, loaderCfg)
	if err != nil {
		return LoaderResult{}, err
	}
	return LoaderResult{Model: loaded, Warning: warn, Backend: effective, FellBack: fellBack}, nil
}

// LoadDraftFromPath loads a draft model (DFlash GGUF or smaller inference checkpoint).
// When the draft hidden size mismatches the target, callers should fall back to target-only.
func LoadDraftFromPath(path string, cfg LoaderConfig, targetHidden int) (model.Model, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, fmt.Errorf("generate: empty draft model path")
	}
	mapped, err := ggufcore.LoadMapped(path)
	if err != nil {
		return nil, err
	}
	arch := strings.ToLower(ggufcore.Architecture(mapped.Parsed))
	if strings.Contains(arch, "dflash") {
		dcfg := model.DFlashConfigFromGGUF(mapped.Parsed)
		if targetHidden > 0 && dcfg.HiddenSize > 0 && dcfg.HiddenSize != targetHidden {
			return nil, fmt.Errorf("generate: draft hidden_size %d != target %d", dcfg.HiddenSize, targetHidden)
		}
		return model.LoadDFlashFromGGUF(mapped, dcfg)
	}
	inf, err := model.LoadInferenceFromGGUF(mapped)
	if err != nil {
		return nil, err
	}
	if targetHidden > 0 && inf.Config.HiddenSize > 0 && inf.Config.HiddenSize != targetHidden {
		return nil, fmt.Errorf("generate: draft hidden_size %d != target %d", inf.Config.HiddenSize, targetHidden)
	}
	return inf, nil
}
