// llama_tokenize_ref.cpp — reference tokenizer driver against llama.cpp(-k2).
//
// Same stdin/stdout protocol as oc-tokenize (one hex-encoded text per line in,
// space-separated ids per line out) so scripts/tokenizer_parity.py can build a
// golden corpus. Loads the model with vocab_only (no weights).
//
// Build (on a host with a llama.cpp build):
//   g++ -O2 -std=c++17 -I$LLAMA/include -I$LLAMA/ggml/include \
//       scripts/llama_tokenize_ref.cpp -L$LLAMA/build-cpu/bin -lllama \
//       -Wl,-rpath,$LLAMA/build-cpu/bin -o llama-tokenize-ref
//
//   llama-tokenize-ref MODEL.gguf [--no-special] [--bos]
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [--no-special] [--bos]\n", argv[0]);
        return 2;
    }
    bool parse_special = true, add_special = false;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--no-special")) parse_special = false;
        else if (!strcmp(argv[i], "--bos")) add_special = true;
    }
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string line;
    std::vector<llama_token> toks;
    while (std::getline(std::cin, line)) {
        std::string text;
        for (size_t i = 0; i + 1 < line.size(); i += 2) {
            int hi = hexval(line[i]), lo = hexval(line[i + 1]);
            if (hi < 0 || lo < 0) break;
            text.push_back((char) (hi * 16 + lo));
        }
        toks.resize(text.size() + 16);
        int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(),
                               (int32_t) toks.size(), add_special, parse_special);
        if (n < 0) {
            toks.resize(-n);
            n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(),
                               (int32_t) toks.size(), add_special, parse_special);
        }
        for (int i = 0; i < n; ++i) printf(i ? " %d" : "%d", toks[i]);
        printf("\n");
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
