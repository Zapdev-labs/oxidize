// oxidize-cpp CLI / benchmark harness.
//
// Ported (in spirit) from oxidize-cli/src/main.rs (the prompt/bench entry point)
// and oxidize-core/src/model/model.rs (Model::forward / forward_many decode
// loop). This is a dependency-free benchmark driver: it loads a dense
// Llama-family GGUF via load_llama_gguf(), feeds a prefill of token ids, then
// greedily decodes max-tokens steps, timing prefill and decode separately.
//
// Tokenization note: model_llama.hpp exposes no detokenizer/encoder, so this
// CLI cannot turn arbitrary prompt text into faithful token ids. It therefore:
//   * uses --tokens "1,2,3" when the caller has pre-tokenized ids (preferred
//     for reproducible benchmarks);
//   * otherwise derives a trivial whitespace/byte fallback from --prompt,
//     clamped to the model vocab, seeded with the GGUF BOS id when present.
// Generation correctness still depends only on the model; the timing numbers
// are independent of which token-id source was used.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "oxidize/config.hpp"
#include "oxidize/gguf.hpp"
#include "oxidize/model.hpp"
#include "oxidize/model_llama.hpp"
#include "oxidize/sampler.hpp"
#include "oxidize/tokenizer.hpp"

namespace {

using oxidize::GgufModel;
using oxidize::LlamaModel;
using oxidize::Logits;
using oxidize::Model;
using oxidize::Session;
using oxidize::Token;

struct Args {
  std::string model;
  std::string prompt = "Hello";
  std::string tokens;  // raw "1,2,3" if provided
  size_t max_tokens = 64;
  bool cuda = false;
  uint64_t seed = 0;
  bool json = false;
  bool debug_logits = false;
  float temperature = 0.0f;  // <=0 => greedy
  size_t top_k = 0;          // 0 => disabled
  float top_p = 1.0f;
  bool no_bos = false;
  bool stream = false;
  std::string quantize;  // "q8_0" => on-the-fly F16/BF16->Q8_0 at load
};

[[noreturn]] void usage_and_exit(const char* prog, int code) {
  std::fprintf(
      stderr,
      "Usage: %s --model <path.gguf> [options]\n"
      "  --model <path.gguf>   GGUF model file (required)\n"
      "  --prompt <str>        Prompt text (default \"Hello\")\n"
      "  --tokens \"1,2,3\"      Pre-tokenized prefill ids (overrides --prompt)\n"
      "  --max-tokens <n>      Decode steps (default 64)\n"
      "  --cuda                Request the CUDA device\n"
      "  --seed <n>            RNG seed (default 0)\n"
      "  --temperature <f>     Sampling temperature (<=0 = greedy; default 0)\n"
      "  --top-k <n>           Top-k filter (0 = disabled)\n"
      "  --top-p <f>           Top-p/nucleus filter (default 1.0)\n"
      "  --no-bos              Do not prepend the BOS token\n"
      "  --stream              Stream decoded text to stdout as it generates\n"
      "  --quantize q8_0       Quantize F16/BF16 weights to Q8_0 at load (~1.3x, near-lossless)\n"
      "  --json                Emit timing JSON\n",
      prog);
  std::exit(code);
}

bool parse_size(std::string_view s, size_t& out) {
  size_t v = 0;
  auto* begin = s.data();
  auto* end = s.data() + s.size();
  auto res = std::from_chars(begin, end, v);
  if (res.ec != std::errc{} || res.ptr != end) return false;
  out = v;
  return true;
}

bool parse_u64(std::string_view s, uint64_t& out) {
  uint64_t v = 0;
  auto* begin = s.data();
  auto* end = s.data() + s.size();
  auto res = std::from_chars(begin, end, v);
  if (res.ec != std::errc{} || res.ptr != end) return false;
  out = v;
  return true;
}

// Require a value-bearing flag; advances i past the value.
std::string take_value(int argc, char** argv, int& i, const char* flag) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "error: %s requires a value\n", flag);
    usage_and_exit(argv[0], 2);
  }
  return std::string(argv[++i]);
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    if (arg == "--model") {
      a.model = take_value(argc, argv, i, "--model");
    } else if (arg == "--prompt") {
      a.prompt = take_value(argc, argv, i, "--prompt");
    } else if (arg == "--tokens") {
      a.tokens = take_value(argc, argv, i, "--tokens");
    } else if (arg == "--max-tokens") {
      std::string v = take_value(argc, argv, i, "--max-tokens");
      if (!parse_size(v, a.max_tokens)) {
        std::fprintf(stderr, "error: invalid --max-tokens '%s'\n", v.c_str());
        usage_and_exit(argv[0], 2);
      }
    } else if (arg == "--cuda") {
      a.cuda = true;
    } else if (arg == "--seed") {
      std::string v = take_value(argc, argv, i, "--seed");
      if (!parse_u64(v, a.seed)) {
        std::fprintf(stderr, "error: invalid --seed '%s'\n", v.c_str());
        usage_and_exit(argv[0], 2);
      }
    } else if (arg == "--json") {
      a.json = true;
    } else if (arg == "--debug-logits") {
      a.debug_logits = true;
    } else if (arg == "--quantize") {
      a.quantize = take_value(argc, argv, i, "--quantize");
    } else if (arg == "--no-bos") {
      a.no_bos = true;
    } else if (arg == "--stream") {
      a.stream = true;
    } else if (arg == "--temperature") {
      a.temperature = std::strtof(take_value(argc, argv, i, "--temperature").c_str(), nullptr);
    } else if (arg == "--top-p") {
      a.top_p = std::strtof(take_value(argc, argv, i, "--top-p").c_str(), nullptr);
    } else if (arg == "--top-k") {
      std::string v = take_value(argc, argv, i, "--top-k");
      if (!parse_size(v, a.top_k)) {
        std::fprintf(stderr, "error: invalid --top-k '%s'\n", v.c_str());
        usage_and_exit(argv[0], 2);
      }
    } else if (arg == "-h" || arg == "--help") {
      usage_and_exit(argv[0], 0);
    } else {
      std::fprintf(stderr, "error: unknown argument '%.*s'\n",
                   static_cast<int>(arg.size()), arg.data());
      usage_and_exit(argv[0], 2);
    }
  }
  if (a.model.empty()) {
    std::fprintf(stderr, "error: --model is required\n");
    usage_and_exit(argv[0], 2);
  }
  return a;
}

// Parse comma-separated token ids, clamped to vocab. Throws on malformed input.
std::vector<Token> parse_token_list(const std::string& csv, size_t vocab) {
  std::vector<Token> out;
  size_t pos = 0;
  while (pos < csv.size()) {
    size_t comma = csv.find(',', pos);
    std::string_view field(csv.data() + pos,
                           (comma == std::string::npos ? csv.size() : comma) - pos);
    // Trim surrounding whitespace.
    while (!field.empty() && (field.front() == ' ' || field.front() == '\t'))
      field.remove_prefix(1);
    while (!field.empty() && (field.back() == ' ' || field.back() == '\t'))
      field.remove_suffix(1);
    if (!field.empty()) {
      size_t id = 0;
      if (!parse_size(field, id)) {
        throw std::runtime_error("invalid token id in --tokens: '" +
                                 std::string(field) + "'");
      }
      if (vocab != 0 && id >= vocab) {
        throw std::runtime_error("token id " + std::to_string(id) +
                                 " out of range (vocab=" + std::to_string(vocab) +
                                 ")");
      }
      out.push_back(static_cast<Token>(id));
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  if (out.empty()) throw std::runtime_error("--tokens produced no ids");
  return out;
}

double secs(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double>(d).count();
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

#ifdef _OPENMP
  // Default OpenMP threads to physical-ish cores. On hyperthreaded machines,
  // using every logical core saturates memory bandwidth and oversubscribes the
  // memory-bound matmuls (measured: 8 threads = 44 tok/s, 16 = 12 tok/s on an
  // 8-core/16-thread box). Respect an explicit OMP_NUM_THREADS if the user set it.
  if (!std::getenv("OMP_NUM_THREADS")) {
    int logical = omp_get_max_threads();
    int want = (logical > 4 && logical % 2 == 0) ? logical / 2 : logical;
    omp_set_num_threads(want);
  }
#endif

  try {
    if (args.cuda) {
#ifndef OXIDIZE_CUDA
      std::fprintf(stderr,
                   "warning: --cuda requested but binary built without CUDA; "
                   "falling back to CPU\n");
      args.cuda = false;
#endif
    }
    // ---- load model ----
    oxidize::QuantType qto = oxidize::QuantType::F32;
    if (args.quantize == "q8_0" || args.quantize == "q8") {
      qto = oxidize::QuantType::Q8_0;
    } else if (!args.quantize.empty()) {
      std::fprintf(stderr, "error: --quantize supports only 'q8_0'\n");
      return 2;
    }
    auto t_load0 = std::chrono::steady_clock::now();
    std::unique_ptr<Model> model =
        oxidize::load_llama_gguf(args.model, args.cuda, qto);
    auto t_load1 = std::chrono::steady_clock::now();

    // Report the device actually in use: --cuda falls back to CPU when no
    // device is present, so query the model rather than trusting the request.
    bool cuda_active = false;
    if (auto* lm = dynamic_cast<LlamaModel*>(model.get()))
      cuda_active = lm->cuda_enabled();
    if (args.cuda && !cuda_active) {
      std::fprintf(stderr,
                   "warning: --cuda requested but no CUDA device active; "
                   "running on CPU\n");
    }
    const char* device = cuda_active ? "cuda" : "cpu";

    const size_t vocab = model->vocab_size();
    const size_t ctx = model->context_size();

    // Print derived config once the model has loaded (observable even if the
    // forward path is not yet implemented for this architecture, e.g. GLM MLA).
    if (auto* lm = dynamic_cast<LlamaModel*>(model.get())) {
      const auto& c = lm->config();
      std::fprintf(stderr,
                   "loaded: arch=%d layers=%zu hidden=%zu heads=%zu kv_heads=%zu "
                   "head_dim=%zu vocab=%zu ctx=%zu\n",
                   static_cast<int>(c.architecture), c.layer_count, c.hidden_size,
                   c.num_attention_heads, c.num_key_value_heads, c.head_dim(),
                   c.vocab_size, c.context_size);
      std::fprintf(stderr,
                   "  moe: experts=%zu used=%zu expert_inter=%zu shared=%zu "
                   "wscale=%.3f wnorm=%d gating_sigmoid=%d\n",
                   c.num_experts, c.num_experts_per_tok, c.expert_intermediate_size,
                   c.num_shared_experts, c.expert_weights_scale,
                   static_cast<int>(c.expert_weights_norm),
                   static_cast<int>(c.expert_gating_sigmoid));
      std::fprintf(stderr,
                   "  mla: q_lora=%zu kv_lora=%zu mla_key=%zu mla_val=%zu "
                   "rope_dim=%zu rope_theta=%.1f\n",
                   c.q_lora_rank, c.kv_lora_rank, c.mla_key_dim, c.mla_val_dim,
                   c.rope_dim, c.rope_theta);
    }

    // ---- tokenizer (from the model's GGUF) ----
    std::optional<oxidize::Tokenizer> tok;
    if (auto* lm = dynamic_cast<LlamaModel*>(model.get())) {
      try {
        tok = oxidize::Tokenizer::from_gguf(lm->gguf());
      } catch (const std::exception& e) {
        std::fprintf(stderr, "warning: tokenizer unavailable (%s); use --tokens\n",
                     e.what());
      }
    }

    // ---- determine prefill tokens ----
    std::vector<Token> prefill;
    if (!args.tokens.empty()) {
      prefill = parse_token_list(args.tokens, vocab);  // pre-tokenized (bench/parity)
    } else if (tok) {
      prefill = tok->encode(args.prompt, !args.no_bos);
    } else {
      std::fprintf(stderr,
                   "error: no usable tokenizer; pass --tokens \"id,id,...\"\n");
      return 2;
    }
    if (prefill.empty()) throw std::runtime_error("empty prefill after tokenize");

    // Respect context: prefill + decode must fit.
    if (ctx != 0 && prefill.size() >= ctx) {
      prefill.resize(ctx - 1);
      if (prefill.empty())
        prefill.push_back((tok && tok->has_bos()) ? tok->bos_id() : 0);
    }
    size_t room = (ctx == 0) ? args.max_tokens
                             : std::min(args.max_tokens, ctx - prefill.size());

    Session session;

    // ---- prefill ----
    auto t_pf0 = std::chrono::steady_clock::now();
    Logits logits = model->forward(prefill, session);
    auto t_pf1 = std::chrono::steady_clock::now();

    if (logits.size() != vocab) {
      throw std::runtime_error(
          "model returned logits of size " + std::to_string(logits.size()) +
          " but vocab_size=" + std::to_string(vocab));
    }

    // ---- debug: top-5 logits of the prefill's final position ----
    if (args.debug_logits) {
      std::vector<size_t> idx(logits.size());
      for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
      std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                        [&](size_t a, size_t b) { return logits[a] > logits[b]; });
      std::fprintf(stderr, "top5:");
      for (int i = 0; i < 5; ++i)
        std::fprintf(stderr, " %zu=%.4f", idx[i], logits[idx[i]]);
      std::fprintf(stderr, "\n");
    }

    // ---- decode ----
    oxidize::Rng rng(args.seed);
    oxidize::SamplerConfig scfg;
    scfg.temperature = args.temperature > 0.0f ? args.temperature : 1.0f;
    if (args.top_k > 0) scfg.top_k = args.top_k;
    if (args.top_p < 1.0f) scfg.top_p = args.top_p;
    const bool greedy = args.temperature <= 0.0f;
    auto pick = [&](const Logits& l) -> Token {
      return greedy ? oxidize::greedy(l)
                    : oxidize::sample(l, scfg, rng.next_unit());
    };

    std::vector<Token> generated;
    generated.reserve(room);

    auto emit = [&](Token t) {
      if (args.stream && tok) {
        std::fputs(tok->decode_token(t).c_str(), stdout);
        std::fflush(stdout);
      }
    };

    auto t_dec0 = std::chrono::steady_clock::now();
    size_t produced = 0;
    Token next = pick(logits);
    if (!(tok && tok->is_eog(next))) {
      generated.push_back(next);
      emit(next);
      ++produced;
      for (; produced < room; ++produced) {
        logits = model->forward({next}, session);
        next = pick(logits);
        if (tok && tok->is_eog(next)) break;
        generated.push_back(next);
        emit(next);
      }
    }
    auto t_dec1 = std::chrono::steady_clock::now();
    if (args.stream) std::fputc('\n', stdout);

    // ---- timings ----
    double load_s = secs(t_load1 - t_load0);
    double prefill_s = secs(t_pf1 - t_pf0);
    double decode_s = secs(t_dec1 - t_dec0);
    double total_s = load_s + prefill_s + decode_s;

    double prefill_tps = prefill_s > 0.0
                             ? static_cast<double>(prefill.size()) / prefill_s
                             : 0.0;
    double decode_tps =
        decode_s > 0.0 ? static_cast<double>(produced) / decode_s : 0.0;

    if (args.json) {
      // Compact, dependency-free JSON.
      std::printf(
          "{\"engine\":\"cpp\",\"device\":\"%s\",\"prefill_tps\":%.6f,"
          "\"decode_tps\":%.6f,\"total_s\":%.6f}\n",
          device, prefill_tps, decode_tps, total_s);
    } else {
      std::printf("oxidize-cpp benchmark\n");
      std::printf("  device:        %s\n", device);
      std::printf("  model:         %s\n", args.model.c_str());
      std::printf("  vocab_size:    %zu\n", vocab);
      std::printf("  context_size:  %zu\n", ctx);
      std::printf("  layer_count:   %zu\n", model->layer_count());
      std::printf("  prefill_tokens:%zu\n", prefill.size());
      std::printf("  decode_tokens: %zu\n", produced);
      std::printf("  ---\n");
      std::printf("  load:          %.4f s\n", load_s);
      std::printf("  prefill:       %.4f s  (%.2f tok/s)\n", prefill_s,
                  prefill_tps);
      std::printf("  decode:        %.4f s  (%.2f tok/s)\n", decode_s,
                  decode_tps);
      std::printf("  total:         %.4f s\n", total_s);
      std::printf("  first_token:   %u\n",
                  generated.empty() ? 0u : generated.front());
      std::printf("  last_token:    %u\n",
                  generated.empty() ? 0u : generated.back());
      std::printf("  gen_tokens:   ");
      for (Token t : generated) std::printf(" %u", t);
      std::printf("\n");
      if (tok && !args.stream) {
        std::printf("  ---\n  text: %s\n", tok->decode(generated).c_str());
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
