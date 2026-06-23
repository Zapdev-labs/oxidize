// oxidize-cpp-train CLI: LoRA + full fine-tuning for Qwen2.5 dense models.
//
// Usage:
//   oxidize-cpp-train --model path.gguf --data path.jsonl
//     [--mode lora|full] [--lr 1e-4] [--steps 200] [--rank 16] [--alpha 32]
//     [--seq-len 512] [--grad-accum 1] [--warmup 100] [--overfit-one-batch]
//     [--seed 42]

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "oxidize/autograd.hpp"
#include "oxidize/gguf.hpp"
#include "oxidize/model_llama.hpp"
#include "oxidize/tokenizer.hpp"
#include "oxidize/train_data.hpp"
#include "oxidize/train_forward.hpp"
#include "oxidize/train_loss.hpp"
#include "oxidize/train_optim.hpp"
#include "oxidize/train_types.hpp"

// Peak RSS in bytes (Linux).
static size_t peak_rss_bytes() {
#ifdef __linux__
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.substr(0, 6) == "VmPeak") {
      size_t kb = 0;
      sscanf(line.c_str(), "VmPeak: %zu kB", &kb);
      return kb * 1024;
    }
  }
#endif
  return 0;
}

static void usage(const char* prog) {
  fprintf(stderr,
    "Usage: %s --model <path.gguf> --data <path.jsonl> [options]\n"
    "  --mode lora|full       Training mode (default: lora)\n"
    "  --lr <float>           Learning rate (default: 1e-4)\n"
    "  --steps <int>          Max optimizer steps (default: 200)\n"
    "  --rank <int>           LoRA rank (default: 16)\n"
    "  --alpha <float>        LoRA alpha (default: 32)\n"
    "  --seq-len <int>        Max sequence length (default: 512)\n"
    "  --grad-accum <int>     Gradient accumulation steps (default: 1)\n"
    "  --warmup <int>         LR warmup steps (default: 10)\n"
    "  --seed <int>           Random seed (default: 42)\n"
    "  --overfit-one-batch    Overfit on a single sample to verify convergence\n",
    prog);
}

int main(int argc, char** argv) {
  oxidize::TrainConfig cfg;
  std::string model_path, data_path;
  bool overfit = false;

  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* flag) { return strcmp(argv[i], flag) == 0; };
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i]); exit(1); }
      return argv[++i];
    };

    if (arg("--model"))          { model_path = next(); }
    else if (arg("--data"))       { data_path  = next(); }
    else if (arg("--mode"))       { std::string m = next(); cfg.mode = (m == "full") ? oxidize::TrainMode::FullFT : oxidize::TrainMode::LoRA; }
    else if (arg("--lr"))         { cfg.adamw.lr = std::stof(next()); }
    else if (arg("--steps"))      { cfg.max_steps = std::stoul(next()); }
    else if (arg("--rank"))       { cfg.lora.rank = std::stoul(next()); }
    else if (arg("--alpha"))      { cfg.lora.alpha = std::stof(next()); }
    else if (arg("--seq-len"))    { cfg.seq_len = std::stoul(next()); }
    else if (arg("--grad-accum")) { cfg.grad_accum = std::stoul(next()); }
    else if (arg("--warmup"))     { cfg.adamw.warmup_steps = std::stoi(next()); }
    else if (arg("--seed"))       { cfg.seed = std::stoull(next()); }
    else if (arg("--grad-clip"))  { cfg.grad_clip = std::stof(next()); }
    else if (arg("--overfit-one-batch")) { overfit = true; }
    else if (arg("--help") || arg("-h")) { usage(argv[0]); return 0; }
    else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); usage(argv[0]); return 1; }
  }

  if (model_path.empty() || data_path.empty()) {
    fprintf(stderr, "Error: --model and --data are required\n");
    usage(argv[0]);
    return 1;
  }

  cfg.overfit_one_batch = overfit;
  cfg.adamw.total_steps = static_cast<int>(cfg.max_steps);
  cfg.model_path = model_path;
  cfg.data_path  = data_path;

  // --- Load model and tokenizer ---
  printf("[train] Loading model: %s\n", model_path.c_str());
  oxidize::GgufModel gguf = oxidize::GgufModel::load(model_path);
  oxidize::Tokenizer tok = oxidize::Tokenizer::from_gguf(gguf);
  oxidize::LlamaModel base(std::move(gguf));

  printf("[train] Model loaded. Mode: %s  seq_len=%zu  steps=%zu  grad_accum=%zu\n",
         cfg.mode == oxidize::TrainMode::LoRA ? "lora" : "full",
         cfg.seq_len, cfg.max_steps, cfg.grad_accum);

  // --- Load data ---
  printf("[train] Loading data: %s\n", data_path.c_str());
  auto samples = oxidize::load_jsonl_samples(data_path, tok, cfg.seq_len);
  if (samples.empty()) {
    fprintf(stderr, "Error: no samples loaded from %s\n", data_path.c_str());
    return 1;
  }
  printf("[train] Loaded %zu samples.\n", samples.size());

  // Overfit-one-batch: use only one sample (first with active loss tokens) repeatedly.
  if (overfit) {
    // Find first sample that has at least one active (assistant) token in window.
    size_t chosen = 0;
    for (size_t si = 0; si < samples.size(); ++si) {
      float active = 0.0f;
      for (size_t t = 1; t < samples[si].loss_mask.size(); ++t)
        active += samples[si].loss_mask[t];
      if (active > 0.0f) { chosen = si; break; }
    }
    printf("[train] --overfit-one-batch: training on sample[%zu] only "
           "(active_tokens=%.0f).\n",
           chosen,
           [&](){float a=0; for(size_t t=1;t<samples[chosen].loss_mask.size();t++) a+=samples[chosen].loss_mask[t]; return a;}());
    samples = {samples[chosen]};
  }

  oxidize::DataLoader loader(samples, 1, cfg.seed);

  // --- Build training model ---
  printf("[train] Building TrainModel...\n");
  oxidize::TrainModel model(&base, cfg, cfg.seed);
  printf("[train] TrainModel built. Activation storage: %.1f MB\n",
         model.activation_bytes() / 1e6);

  // --- Training loop ---
  int opt_step = 0;
  float loss_sum = 0.0f;
  int accum_count = 0;
  float last_loss = 0.0f;

  auto t_start = std::chrono::steady_clock::now();

  for (size_t step = 0; step < cfg.max_steps * cfg.grad_accum; ++step) {
    const oxidize::TrainSample& sample = loader.next_sample();
    const size_t T = sample.tokens.size();
    if (T < 2) continue;

    // Forward: tokens[0..T-1] -> logits[0..T-1].
    // Target at position t is tokens[t+1] (causal LM).
    // We compute loss only on positions 0..T-2 (there's no target for pos T-1).
    std::vector<oxidize::Token> input_tokens(sample.tokens.begin(),
                                              sample.tokens.begin() + T - 1);
    std::vector<oxidize::Token> target_tokens(sample.tokens.begin() + 1,
                                               sample.tokens.end());
    std::vector<float> mask(sample.loss_mask.begin() + 1, sample.loss_mask.end());

    // Forward.
    auto t_fwd0 = std::chrono::steady_clock::now();
    auto logits = model.forward(input_tokens);
    auto t_fwd1 = std::chrono::steady_clock::now();
    size_t T_in = input_tokens.size();
    size_t vocab = base.config().vocab_size;

    // Count active tokens.
    float active = 0.0f;
    for (float m : mask) active += m;
    if (active <= 0.0f) continue;

    // Compute loss and logits gradient.
    float loss = 0.0f;
    std::vector<float> dlogits(T_in * vocab, 0.0f);

    for (size_t t = 0; t < T_in; ++t) {
      if (mask[t] <= 0.0f) continue;
      const float* z = logits.data() + t * vocab;
      loss += oxidize::cross_entropy_forward(z, target_tokens[t], mask[t], vocab);
      oxidize::cross_entropy_backward(dlogits.data() + t * vocab,
                                       z, target_tokens[t],
                                       mask[t], active * static_cast<float>(cfg.grad_accum),
                                       vocab);
    }
    loss /= active;
    loss_sum += loss;
    ++accum_count;

    auto t_bwd0 = std::chrono::steady_clock::now();
    // Backward.
    model.backward(dlogits, input_tokens, mask);
    auto t_bwd1 = std::chrono::steady_clock::now();

    // Gradient clipping (applied once per grad_accum window, after last backward).
    if (accum_count >= static_cast<int>(cfg.grad_accum) && cfg.grad_clip > 0.0f) {
      model.clip_grad_norm(cfg.grad_clip);
    }
    if (opt_step == 0 && accum_count == 1) {
      printf("[timing] T=%zu fwd=%.2fs bwd=%.2fs\n", T_in,
             std::chrono::duration<double>(t_fwd1-t_fwd0).count(),
             std::chrono::duration<double>(t_bwd1-t_bwd0).count());
      fflush(stdout);
    }

    // Optimizer step every grad_accum batches.
    if (accum_count >= static_cast<int>(cfg.grad_accum)) {
      ++opt_step;
      float lr = oxidize::compute_lr(cfg.adamw.lr, opt_step,
                                      cfg.adamw.warmup_steps, cfg.adamw.total_steps);
      model.optimizer_step(lr, opt_step);
      model.zero_grads();

      last_loss = loss_sum / static_cast<float>(accum_count);
      loss_sum  = 0.0f;
      accum_count = 0;

      if (opt_step % 10 == 0 || opt_step <= 5) {
        size_t rss = peak_rss_bytes();
        printf("[train] step=%4d  loss=%.4f  lr=%.2e  RSS=%.1f MB\n",
               opt_step, last_loss, lr, rss / 1e6);
        fflush(stdout);
      }

      // Overfitting check: report every step.
      if (overfit && opt_step % 1 == 0) {
        printf("[overfit] step=%d  loss=%.6f\n", opt_step, last_loss);
        fflush(stdout);
      }
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();
  size_t rss = peak_rss_bytes();

  printf("\n[train] Done. Steps=%d  Final loss=%.4f  Time=%.1fs  Peak RSS=%.1f MB\n",
         opt_step, last_loss, elapsed, rss / 1e6);

  return 0;
}
