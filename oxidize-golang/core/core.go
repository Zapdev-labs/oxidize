// Package core is the Go port of oxidize-core (Rust). It contains the entire
// inference engine, format parsers (GGUF, SafeTensors), tokenizer families,
// compute kernels, KV cache, paged-attention scheduler, mesh/distributed
// primitives, vision encoder, and supporting utilities.
//
// The package layout mirrors the Rust crate's top-level modules; subpackages
// are importable individually or re-exported through this top-level file.
package core

import (
	"github.com/Zapdev-labs/oxidize/golang/core/backend"
	"github.com/Zapdev-labs/oxidize/golang/core/conversion"
	"github.com/Zapdev-labs/oxidize/golang/core/cpu_kernels"
	"github.com/Zapdev-labs/oxidize/golang/core/flash_attention"
	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
	"github.com/Zapdev-labs/oxidize/golang/core/kv_cache"
	"github.com/Zapdev-labs/oxidize/golang/core/mesh"
	"github.com/Zapdev-labs/oxidize/golang/core/model"
	"github.com/Zapdev-labs/oxidize/golang/core/paged"
	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
	"github.com/Zapdev-labs/oxidize/golang/core/safetensors"
	"github.com/Zapdev-labs/oxidize/golang/core/simd"
	"github.com/Zapdev-labs/oxidize/golang/core/tensor"
	"github.com/Zapdev-labs/oxidize/golang/core/tokenizer"
	"github.com/Zapdev-labs/oxidize/golang/core/turboquant"
	"github.com/Zapdev-labs/oxidize/golang/core/util"
	"github.com/Zapdev-labs/oxidize/golang/core/validation"
	"github.com/Zapdev-labs/oxidize/golang/core/vision"
	"github.com/Zapdev-labs/oxidize/golang/core/workspace"
)

// WorkspaceHealth mirrors oxidize_core::WorkspaceHealth from the Rust crate.
type WorkspaceHealth = workspace.WorkspaceHealth

// WorkspaceHealthReady returns the constant workspace readiness signal.
func WorkspaceHealthReady() WorkspaceHealth { return workspace.Health() }

// BenchmarkInput mirrors oxidize_core::benchmark_input from the Rust crate.
func BenchmarkInput() WorkspaceHealth { return workspace.BenchmarkInput() }

// WasmWorkspaceStatus mirrors the wasm_bindgen export from the Rust crate.
func WasmWorkspaceStatus() string { return workspace.WasmStatus() }

// ImplementedValidationSuites lists the cross-validation suites supported by
// the validation package, mirroring oxidize_core::implemented_validation_suites.
func ImplementedValidationSuites() []validation.Suite { return validation.ImplementedSuites() }

// ImplementedCpuKernels lists the fused CPU kernels available in this build.
func ImplementedCpuKernels() []cpu_kernels.Kernel { return cpu_kernels.ImplementedKernels() }

// ImplementedLoraFeatures lists the LoRA features available in this build.
func ImplementedLoraFeatures() []string { return []string{"alpha-scaling", "merge-strategies", "rank-budget"} }

// ImplementedDFlashFeatures lists the DFlash draft-model features available.
func ImplementedDFlashFeatures() []string {
	features := model.ImplementedDFlashFeatures()
	out := make([]string, len(features))
	for i, f := range features {
		out[i] = string(f)
	}
	return out
}

// Re-exports for terse top-level access.
type (
	Backend                 = backend.Backend
	ComputeBackendInterface = backend.ComputeBackend

	DType                 = tensor.DType
	Tensor                = tensor.Tensor
	ActivationFn          = tensor.ActivationFn
	GemvError             = tensor.GemvError
	GemmError             = tensor.GemmError
	AttentionError        = tensor.AttentionError
	RopeError             = tensor.RopeError
	SwiGluError           = tensor.SwiGluError
	RmsNormError          = tensor.RmsNormError
	LayerNormError        = tensor.LayerNormError
	SoftmaxError          = tensor.SoftmaxError
	LinearActivationError = tensor.LinearActivationError

	QuantizationError    = quantization.Error
	QuantizationType     = quantization.Type
	IMatrix              = quantization.IMatrix
	MixedLayerPlan       = quantization.MixedLayerPlan
	QuantizedLayer       = quantization.QuantizedLayer

	SafeTensorsError     = safetensors.Error
	MappedSafeTensors    = safetensors.MappedFile
	SafeTensorsTensor    = safetensors.TensorInfo
	SafeTensorsDType     = safetensors.DType

	GgufFile             = ggufcore.File
	GgufTensorInfo       = ggufcore.TensorInfo
	GgufQuantizationType = ggufcore.QuantizationType
	GgufMetadataValue    = ggufcore.MetadataValue
	GgufMetadataArray    = ggufcore.MetadataArray
	GgufMetadataType     = ggufcore.MetadataType
	GgufParseError       = ggufcore.ParseError
	MappedGgufFile       = ggufcore.MappedFile

	Tokenizer               = tokenizer.Tokenizer
	BpeTokenizer            = tokenizer.BpeTokenizer
	SentencePieceTokenizer  = tokenizer.SentencePieceUnigramTokenizer
	WordPieceTokenizer      = tokenizer.WordPieceTokenizer
	TiktokenTokenizer       = tokenizer.TiktokenTokenizer
	TokenizerError          = tokenizer.Error
	TokenizerLoadError      = tokenizer.LoadError
	EncodeOptions           = tokenizer.EncodeOptions
	SpecialTokens           = tokenizer.SpecialTokens
	StreamingDetokenizer    = tokenizer.StreamingDetokenizer
	ChatMessage             = tokenizer.ChatMessage

	KvCache                 = kv_cache.Cache
	KvCacheConfig           = kv_cache.Config
	KvQuantization          = kv_cache.Quantization
	KvCacheEvictionStrategy = kv_cache.EvictionStrategy
	KvCacheError            = kv_cache.Error
	KvCachePersistenceError = kv_cache.PersistenceError
	ContinuousBatchKvCache  = kv_cache.ContinuousBatchCache
	ContinuousBatchError    = kv_cache.ContinuousBatchError

	SimdBackend             = simd.Backend

	FlashAttentionError     = flash_attention.Error

	TurboQuantType          = turboquant.Type
	TurboQuantData          = turboquant.Data
	TurboQuantBlock         = turboquant.Block

	CpuWorkspace            = cpu_kernels.Workspace
	CpuKernel               = cpu_kernels.Kernel
	FusedRmsNormGemv        = cpu_kernels.FusedRmsNormGemv
	FusedCpuError           = cpu_kernels.FusedError

	ModelArchitecture       = model.Architecture
	ModelTrait              = model.Model
	Session                 = model.Session
	Token                   = model.Token
	Logits                  = model.Logits
	ModelError              = model.Error
	BoxedModel              = model.Boxed

	InferenceConfig         = model.InferenceConfig
	InferenceModel          = model.InferenceModel
	Workspace               = model.Workspace
	WeightStorage           = model.WeightStorage

	LlamaModel              = model.LlamaModel
	LlamaArchitecture       = model.LlamaArchitecture
	LlamaConfig             = model.LlamaConfig

	LayerWiseModel          = model.LayerWiseModel

	GenerationConfig        = model.GenerationConfig
	GenerationError         = model.GenerationError
	GenerationStream        = model.GenerationStream
	SpeculativeGenConfig    = model.SpeculativeGenerationConfig
	SpeculativeGenStream    = model.SpeculativeGenerationStream

	SamplingConfig          = model.SamplingConfig
	SamplingError           = model.SamplingError
	NewlinePenalty          = model.NewlinePenalty
	RepetitionPenaltyConfig = model.RepetitionPenaltyConfig
	MirostatConfig          = model.MirostatConfig
	GrammarConstraint       = model.GrammarConstraint
	GrammarSymbol           = model.GrammarSymbol
	SpeculativeDecodeResult = model.SpeculativeDecodeResult
	BeamSearchResult        = model.BeamSearchResult

	SpeculativeConfig       = model.SpeculativeConfig
	SpeculativeStats        = model.SpeculativeStats
	SpeculativeDecoder      = model.SpeculativeDecoder
	SpeculativeError        = model.SpeculativeError

	DFlashConfig            = model.DFlashConfig
	DFlashDraftModel        = model.DFlashDraftModel

	LoraLayer               = model.LoraLayer
	LoraAdapter             = model.LoraAdapter
	LoraPlan                = model.LoraPlan
	LoraError               = model.LoraError
	RankBudgetExceededError = model.RankBudgetExceededError
	LoraScalingConfig       = model.LoraScalingConfig

	ModelLoaderTrait        = model.ModelLoader
	GgufModelLoader         = model.GgufModelLoader
	BaselineGgufModel       = model.BaselineGgufModel
	ModelSource             = model.ModelSource
	FileSource              = model.FileSource
	MemorySource            = model.MemorySource
	HFSource                = model.HFSource
	ModelType               = model.ModelType
	LoaderConfig            = model.LoaderConfig
	LoadMetrics             = model.LoadMetrics

	LayerOffloadPlan        = model.LayerOffloadPlan
	LayerAssignment         = model.LayerAssignment
	MultiGpuOffloadPlan     = model.MultiGpuOffloadPlan
	PipelineStage           = model.PipelineStage
	DeviceMemory            = model.DeviceMemory
	LayerOffloadPlanner     = model.LayerOffloadPlanner
	OffloadPolicy           = model.OffloadPolicy
	OffloadMetrics          = model.OffloadMetrics
	GpuOffloadConfig        = model.GpuOffloadConfig

	PrefixHash              = model.PrefixHash
	CachedPrefix            = model.CachedPrefix
	PrefixCache             = model.PrefixCache
	PrefixMatcher           = model.PrefixMatcher
	PrefixCacheConfig       = model.PrefixCacheConfig

	SamplerStep             = model.SamplerStep
	SamplerChain            = model.SamplerChain
	ToolFunction            = model.ToolFunction
	ToolCall                = model.ToolCall
	ToolRegistry            = model.ToolRegistry
	EngineConfig            = model.EngineConfig

	ConversionPlan          = conversion.Plan
)

// ConvArch returns the detected model architecture from a metadata map.
func ConvArch(metadata map[string]GgufMetadataValue) conversion.Architecture {
	return conversion.DetectArchitecture(metadata)
}

// ConvMapHFTensorName converts a HuggingFace-style tensor name to the GGUF
// naming convention used by the Llama.cpp family of models.
func ConvMapHFTensorName(name string) string { return conversion.MapHFTensorName(name) }

// ConvBuildPlan assembles a conversion plan from a parsed GGUF file.
func ConvBuildPlan(file GgufFile, target *QuantizationType) ConversionPlan {
	return conversion.BuildPlan(file, target)
}

// ConvParseSpecialTokenID extracts a special token id from metadata by key.
func ConvParseSpecialTokenID(metadata map[string]GgufMetadataValue, key string) (uint32, bool) {
	return conversion.ParseSpecialTokenID(metadata, key)
}

// Tensor type aliases for convenient construction.
func NewTensor(data []float32, shape []int) *Tensor         { return tensor.New(data, shape) }
func NewTensorFromBytes(data []byte, shape []int, dt DType) (*Tensor, error) {
	return tensor.FromBytes(data, shape, dt)
}

// Re-exported constants.
const (
	QK8_0        = quantization.QK8_0
	QK4_0        = quantization.QK4_0
	QK4_1        = quantization.QK4_1
	QK5_0        = quantization.QK5_0
	QK5_1        = quantization.QK5_1
	QK_K         = quantization.QK_K
	QK_NVFP4     = quantization.QK_NVFP4
	QK_NVFP4_SUB = quantization.QK_NVFP4_SUB
	BLOCK_Q4_0   = quantization.BLOCK_Q4_0_SIZE
	BLOCK_Q4_1   = quantization.BLOCK_Q4_1_SIZE
	BLOCK_Q5_0   = quantization.BLOCK_Q5_0_SIZE
	BLOCK_Q5_1   = quantization.BLOCK_Q5_1_SIZE
	BLOCK_Q8_0   = quantization.BLOCK_Q8_0_SIZE
	BLOCK_Q2_K   = quantization.BLOCK_Q2_K_SIZE
	BLOCK_Q3_K   = quantization.BLOCK_Q3_K_SIZE
	BLOCK_Q4_K   = quantization.BLOCK_Q4_K_SIZE
	BLOCK_Q5_K   = quantization.BLOCK_Q5_K_SIZE
	BLOCK_Q6_K   = quantization.BLOCK_Q6_K_SIZE
	BLOCK_Q8_K   = quantization.BLOCK_Q8_K_SIZE
	BLOCK_NVFP4  = quantization.BLOCK_NVFP4_SIZE
	BLOCK_IQ1_S  = quantization.BLOCK_IQ1_S_SIZE
	BLOCK_IQ1_M  = quantization.BLOCK_IQ1_M_SIZE
	BLOCK_IQ2_XXS= quantization.BLOCK_IQ2_XXS_SIZE
	BLOCK_IQ2_XS = quantization.BLOCK_IQ2_XS_SIZE
	BLOCK_IQ2_S  = quantization.BLOCK_IQ2_S_SIZE
	BLOCK_IQ3_XXS= quantization.BLOCK_IQ3_XXS_SIZE
	BLOCK_IQ3_S  = quantization.BLOCK_IQ3_S_SIZE
	BLOCK_IQ4_NL = quantization.BLOCK_IQ4_NL_SIZE
	BLOCK_IQ4_XS = quantization.BLOCK_IQ4_XS_SIZE

	FLASH_ATTENTION_BLOCK_TOKENS = tensor.FlashAttentionBlockTokens
	PARALLEL_GEMV_MIN_OPS        = tensor.ParallelGemvMinOps
	TRANSPOSED_GEMV_COL_CHUNK    = tensor.TransposedGemvColChunk
)

// Quantization type variants mirrored as constants.
const (
	QT_F32     = quantization.TypeF32
	QT_F16     = quantization.TypeF16
	QT_Q4_0    = quantization.TypeQ4_0
	QT_Q4_1    = quantization.TypeQ4_1
	QT_Q5_0    = quantization.TypeQ5_0
	QT_Q5_1    = quantization.TypeQ5_1
	QT_Q8_0    = quantization.TypeQ8_0
	QT_Q2_K    = quantization.TypeQ2_K
	QT_Q3_K_S  = quantization.TypeQ3_K_S
	QT_Q3_K_M  = quantization.TypeQ3_K_M
	QT_Q3_K_L  = quantization.TypeQ3_K_L
	QT_Q4_K_S  = quantization.TypeQ4_K_S
	QT_Q4_K_M  = quantization.TypeQ4_K_M
	QT_Q5_K_S  = quantization.TypeQ5_K_S
	QT_Q5_K_M  = quantization.TypeQ5_K_M
	QT_Q6_K    = quantization.TypeQ6_K
	QT_IQ2_XXS = quantization.TypeIQ2_XXS
	QT_IQ2_XS  = quantization.TypeIQ2_XS
	QT_IQ3_XXS = quantization.TypeIQ3_XXS
	QT_IQ1_S   = quantization.TypeIQ1_S
	QT_IQ4_NL  = quantization.TypeIQ4_NL
	QT_IQ3_S   = quantization.TypeIQ3_S
	QT_IQ2_S   = quantization.TypeIQ2_S
	QT_IQ4_XS  = quantization.TypeIQ4_XS
	QT_IQ1_M   = quantization.TypeIQ1_M
	QT_NVFP4   = quantization.TypeNVFP4
)

// DType variants.
const (
	DTypeF32 = tensor.DTypeF32
	DTypeF16 = tensor.DTypeF16
	DTypeI8  = tensor.DTypeI8
	DTypeI16 = tensor.DTypeI16
	DTypeI32 = tensor.DTypeI32
	DTypeI64 = tensor.DTypeI64
)

// Backend variants.
const (
	BackendCpu      = backend.BackendCpu
	BackendMetal    = backend.BackendMetal
	BackendCuda     = backend.BackendCuda
	BackendMlx      = backend.BackendMlx
	BackendVulkan   = backend.BackendVulkan
	BackendIntelArc = backend.BackendIntelArc
)

// KV cache eviction strategies.
const (
	KvEvictSlidingWindow = kv_cache.EvictSlidingWindow
	KvEvictStopAtCapacity= kv_cache.EvictStopAtCapacity
)

// SIMD variants.
const (
	SimdScalar   = simd.BackendScalar
	SimdSse2     = simd.BackendSse2
	SimdAvx      = simd.BackendAvx
	SimdAvx2     = simd.BackendAvx2
	SimdAvx512f  = simd.BackendAvx512f
	SimdNeon     = simd.BackendNeon
)

// Vision re-exports.
type (
	VisionConfig        = vision.Config
	VisionEncoder       = vision.Encoder
	VisionError         = vision.Error
	ImagePreprocessor   = vision.Preprocessor
	MultimodalPrompt    = vision.MultimodalPrompt
	VisionStubEncoder   = vision.StubEncoder
	VisionStubPreprocessor = vision.StubPreprocessor
)

// Paged re-exports.
type (
	Scheduler           = paged.Scheduler
	SchedulerConfig     = paged.SchedulerConfig
	SchedulerError      = paged.SchedulerError
	BlockPool           = paged.BlockPool
	PhysicalBlock       = paged.PhysicalBlock
	BlockTable          = paged.BlockTable
	BlockHash           = paged.BlockHash
	PagedRequest        = paged.Request
)

// Mesh re-exports.
type (
	MeshConfig           = mesh.MeshConfig
	MeshNode             = mesh.MeshNode
	GossipRouter         = mesh.GossipRouter
	RingTransport        = mesh.RingTransport
	ChannelTransport     = mesh.ChannelTransport
	TcpTransport         = mesh.TcpTransport
	ShardPlan            = mesh.ShardPlan
	MeshShard            = mesh.MeshShard
	DiscoveryService     = mesh.DiscoveryService
	BullyElection        = mesh.BullyElection
	TopologyGraph        = mesh.TopologyGraph
	MeshChatEngine       = mesh.MeshChatEngine
	LoadProgressReport   = mesh.LoadProgressReport
	MeshValidationReport = mesh.MeshValidationReport
)

// Util re-exports.
type (
	BenchmarkCase        = util.BenchmarkCase
	PerplexityDatasetCase= util.PerplexityDatasetCase
	BenchmarkResult      = util.Result
	BenchmarkSummary     = util.Summary
	WebWorkerRequest     = util.WebWorkerRequest
	WebWorkerResponse    = util.WebWorkerResponse
	PipelineStep         = util.PipelineStep
	Pipeline             = util.Pipeline
	PipelineError        = util.PipelineError
)

// Validation re-exports.
type (
	ValidationSuite = validation.Suite
	ValidationResult= validation.Result
	ParityReport    = validation.ParityReport
	ParityError     = validation.ParityError
)

// ModelConfigOpts is a passthrough for forwarders.
var _ = util.Summarise
