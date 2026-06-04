package cli

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/internal/generate"
)

// genOptions holds flags shared by run, chat, bench, and serve.
type genOptions struct {
	Prompt         string
	MaxTokens      int
	Temperature    float64
	TopP           float64
	TopK           int
	Threads        int
	CtxSize        int
	Backend        string
	NGPULayers     int
	GPUs           int
	Parallelism    string
	DraftModel     string
	DraftTokens    int
	TokenizerModel string
	HFFile         string
	UsePaged       bool
	DFlashFusion   bool
	Mesh           bool
	MeshPort       int
	PipeHead       bool
	PipeTail       bool
	PipePeer       string
	PipeListen     string
	Profile        bool
	Vision         bool
	ImagePath      string
}

func registerGenFlags(fs *flag.FlagSet, opts *genOptions) {
	fs.StringVar(&opts.Prompt, "prompt", "", "prompt text")
	fs.IntVar(&opts.MaxTokens, "max-tokens", 128, "maximum new tokens to generate")
	fs.Float64Var(&opts.Temperature, "temperature", 0.8, "sampling temperature")
	fs.Float64Var(&opts.TopP, "top-p", 0.9, "nucleus sampling top-p")
	fs.IntVar(&opts.TopK, "top-k", 0, "top-k sampling (0 = disabled)")
	fs.IntVar(&opts.Threads, "threads", 0, "CPU thread count (0 = default)")
	fs.IntVar(&opts.CtxSize, "ctx-size", 0, "context window override (0 = model default)")
	fs.StringVar(&opts.Backend, "backend", "cpu", "compute backend: cpu, cuda, metal, vulkan, webgpu, mlx")
	fs.IntVar(&opts.NGPULayers, "n-gpu-layers", 0, "layers to offload to GPU")
	fs.IntVar(&opts.GPUs, "gpus", 1, "GPU device count for offload")
	fs.StringVar(&opts.Parallelism, "parallelism", "pipeline", "parallelism strategy: pipeline, tensor, layer")
	fs.StringVar(&opts.DraftModel, "draft-model", "", "DFlash or draft GGUF path for speculative decoding")
	fs.IntVar(&opts.DraftTokens, "draft-tokens", 4, "draft tokens per speculative step")
	fs.StringVar(&opts.TokenizerModel, "tokenizer-model", "", "external GGUF with tokenizer metadata")
	fs.StringVar(&opts.HFFile, "file", "", "GGUF filename when model is a Hugging Face repo")
	fs.BoolVar(&opts.UsePaged, "paged", false, "use PagedAttention scheduler for generation")
	fs.BoolVar(&opts.DFlashFusion, "dflash-fusion", false, "use SpeculativeDecoder fusion (heuristic or --draft-model)")
	fs.BoolVar(&opts.Mesh, "mesh", false, "start mesh node (chat REPL broadcasts prompts)")
	fs.IntVar(&opts.MeshPort, "mesh-port", 0, "mesh listen port (0 = ephemeral)")
	fs.BoolVar(&opts.PipeHead, "pipe-head", false, "pipeline head stage")
	fs.BoolVar(&opts.PipeTail, "pipe-tail", false, "pipeline tail stage")
	fs.StringVar(&opts.PipePeer, "pipe-peer", "", "pipeline next stage address")
	fs.StringVar(&opts.PipeListen, "pipe-listen", "", "pipeline listen address")
	fs.BoolVar(&opts.Profile, "profile", false, "print generation profile stats after run")
	fs.BoolVar(&opts.Vision, "vision", false, "enable vision/multimodal path")
	fs.StringVar(&opts.ImagePath, "image", "", "image file for vision mode")
}

func parseGenFlags(name string, args []string) (*flag.FlagSet, genOptions, []string, error) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	var opts genOptions
	registerGenFlags(fs, &opts)
	if err := fs.Parse(args); err != nil {
		return nil, genOptions{}, nil, err
	}
	rest := fs.Args()
	if strings.TrimSpace(opts.Prompt) == "" && len(rest) > 1 && !strings.HasPrefix(rest[1], "-") {
		opts.Prompt = strings.Join(rest[1:], " ")
		rest = rest[:1]
	}
	return fs, opts, rest, nil
}

func (o genOptions) runConfig(modelPath string) generate.RunConfig {
	cfg := generate.DefaultRunConfig()
	cfg.ModelPath = modelPath
	cfg.Prompt = strings.TrimSpace(o.Prompt)
	if o.MaxTokens > 0 {
		cfg.MaxNewTokens = o.MaxTokens
	}
	cfg.Temperature = float32(o.Temperature)
	cfg.TopP = float32(o.TopP)
	cfg.TopK = o.TopK
	cfg.Threads = o.Threads
	cfg.ContextSize = o.CtxSize
	cfg.Backend = strings.TrimSpace(o.Backend)
	cfg.NGPULayers = o.NGPULayers
	cfg.GPUs = o.GPUs
	cfg.Parallelism = strings.TrimSpace(o.Parallelism)
	cfg.DraftModel = strings.TrimSpace(o.DraftModel)
	cfg.DraftTokens = o.DraftTokens
	cfg.TokenizerModel = strings.TrimSpace(o.TokenizerModel)
	cfg.HFFile = strings.TrimSpace(o.HFFile)
	cfg.UsePaged = o.UsePaged
	cfg.UseDFlashFusion = o.DFlashFusion
	cfg.Vision = o.Vision
	cfg.ImagePath = strings.TrimSpace(o.ImagePath)
	return cfg
}

func (o genOptions) loaderConfig() generate.LoaderConfig {
	return generate.LoaderConfig{
		Backend:        strings.TrimSpace(o.Backend),
		Threads:        o.Threads,
		ContextSize:    o.CtxSize,
		NGPULayers:     o.NGPULayers,
		GPUs:           o.GPUs,
		Parallelism:    strings.TrimSpace(o.Parallelism),
		HFFilename:     strings.TrimSpace(o.HFFile),
		AllowFallback:  true,
	}
}

func parseBenchEngine(args []string) (engine string, rest []string, err error) {
	engine = "inference"
	rest = append([]string(nil), args...)
	for i := 0; i < len(rest); i++ {
		if rest[i] != "--engine" {
			continue
		}
		if i+1 >= len(rest) {
			return "", nil, fmt.Errorf("missing value for --engine")
		}
		engine = strings.ToLower(strings.TrimSpace(rest[i+1]))
		rest = append(append(rest[:i], rest[i+2:]...))
		break
	}
	return engine, rest, nil
}

func validateBenchEngine(engine string) error {
	switch engine {
	case "inference", "dflash":
		return nil
	default:
		return fmt.Errorf("unknown bench engine %q (use inference or dflash)", engine)
	}
}
