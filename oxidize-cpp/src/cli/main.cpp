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

// Trivial whitespace/byte fallback: hashes each prompt token (or byte) into the
// vocab range. Deterministic but NOT a faithful tokenizer; used only when no
// --tokens and no GGUF-side encoder are available, so prefill has real work.
std::vector<Token> fallback_tokens(const std::string& prompt, size_t vocab,
                                   std::optional<uint32_t> bos) {
  std::vector<Token> out;
  if (bos.has_value() && (vocab == 0 || *bos < vocab)) out.push_back(*bos);

  auto push_word = [&](std::string_view w) {
    if (w.empty()) return;
    uint64_t h = 1469598103934665603ull;  // FNV-1a 64
    for (unsigned char c : w) {
      h ^= c;
      h *= 1099511628211ull;
    }
    Token t = vocab ? static_cast<Token>(h % vocab) : static_cast<Token>(h);
    out.push_back(t);
  };

  size_t i = 0;
  while (i < prompt.size()) {
    while (i < prompt.size() &&
           (prompt[i] == ' ' || prompt[i] == '\t' || prompt[i] == '\n' ||
            prompt[i] == '\r'))
      ++i;
    size_t start = i;
    while (i < prompt.size() &&
           !(prompt[i] == ' ' || prompt[i] == '\t' || prompt[i] == '\n' ||
             prompt[i] == '\r'))
      ++i;
    if (i > start) push_word(std::string_view(prompt.data() + start, i - start));
  }

  if (out.empty()) {
    // Empty prompt: feed a single in-range token so prefill is non-trivial.
    out.push_back(bos.value_or(0) < (vocab ? vocab : SIZE_MAX)
                      ? static_cast<Token>(bos.value_or(0))
                      : static_cast<Token>(0));
  }
  return out;
}

std::optional<uint32_t> read_bos(const GgufModel& g) {
  if (auto v = g.get_u32("tokenizer.ggml.bos_token_id")) return v;
  return std::nullopt;
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
    auto t_load0 = std::chrono::steady_clock::now();
    std::unique_ptr<Model> model =
        oxidize::load_llama_gguf(args.model, args.cuda);
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

    // ---- determine prefill tokens ----
    std::optional<uint32_t> bos;
    {
      // Re-open just for BOS metadata (cheap header parse, no extra mmap kept).
      // The model already mmapped the file; this second parse only reads the
      // header region. If it fails we simply skip BOS seeding.
      try {
        GgufModel g = GgufModel::load(args.model);
        bos = read_bos(g);
      } catch (...) {
        bos = std::nullopt;
      }
    }

    std::vector<Token> prefill;
    if (!args.tokens.empty()) {
      prefill = parse_token_list(args.tokens, vocab);
    } else {
      prefill = fallback_tokens(args.prompt, vocab, bos);
    }

    // Respect context: prefill + decode must fit.
    if (ctx != 0 && prefill.size() >= ctx) {
      prefill.resize(ctx - 1);
      if (prefill.empty()) prefill.push_back(bos.value_or(0));
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
    oxidize::Rng rng(args.seed);  // reserved for non-greedy paths; greedy below.
    (void)rng;
    std::vector<Token> generated;
    generated.reserve(room);

    auto t_dec0 = std::chrono::steady_clock::now();
    size_t produced = 0;
    Token next = oxidize::greedy(logits);
    generated.push_back(next);
    ++produced;
    for (; produced < room; ++produced) {
      logits = model->forward({next}, session);
      next = oxidize::greedy(logits);
      generated.push_back(next);
    }
    auto t_dec1 = std::chrono::steady_clock::now();

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
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
