// GGUF tokenizer implementation: SentencePiece (llama) + byte-level BPE
// (gpt2/qwen). See tokenizer.hpp for the algorithm summary.

#include "oxidize/tokenizer.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <queue>
#include <stdexcept>

namespace oxidize {

namespace {

// UTF-8: byte length of the char starting at `c`.
inline size_t utf8_len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c >> 5) == 0x6) return 2;
  if ((c >> 4) == 0xe) return 3;
  if ((c >> 3) == 0x1e) return 4;
  return 1;  // invalid lead byte -> treat as 1
}

const std::string kSpace = "\xe2\x96\x81";  // U+2581 "▁"

// GPT-2 byte<->unicode reversible map. Bytes that are not printable ASCII/Latin
// get remapped to code points 256+n so every byte is a valid single char.
const std::array<std::string, 256>& byte_to_unicode() {
  static const std::array<std::string, 256> table = [] {
    std::array<std::string, 256> t;
    auto cp_to_utf8 = [](uint32_t cp) {
      std::string s;
      if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
      } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      } else {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      }
      return s;
    };
    std::array<bool, 256> printable{};
    auto mark = [&](int lo, int hi) {
      for (int b = lo; b <= hi; ++b) printable[b] = true;
    };
    mark('!', '~');      // 0x21..0x7E
    mark(0xA1, 0xAC);    // ¡..¬
    mark(0xAE, 0xFF);    // ®..ÿ
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if (printable[b]) {
        t[b] = cp_to_utf8(static_cast<uint32_t>(b));
      } else {
        t[b] = cp_to_utf8(static_cast<uint32_t>(256 + n));
        ++n;
      }
    }
    return t;
  }();
  return table;
}

// Reverse of byte_to_unicode: unicode char (utf8) -> original byte.
const std::unordered_map<std::string, uint8_t>& unicode_to_byte() {
  static const std::unordered_map<std::string, uint8_t> rev = [] {
    std::unordered_map<std::string, uint8_t> m;
    const auto& fwd = byte_to_unicode();
    for (int b = 0; b < 256; ++b) m[fwd[b]] = static_cast<uint8_t>(b);
    return m;
  }();
  return rev;
}

bool is_byte_token(const std::string& p, uint8_t& out) {
  // "<0xXX>"
  if (p.size() == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
    auto hex = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      return -1;
    };
    int hi = hex(p[3]), lo = hex(p[4]);
    if (hi >= 0 && lo >= 0) {
      out = static_cast<uint8_t>(hi * 16 + lo);
      return true;
    }
  }
  return false;
}

}  // namespace

int32_t Tokenizer::piece_id(const std::string& p) const {
  auto it = piece_to_id_.find(p);
  return it == piece_to_id_.end() ? -1 : it->second;
}

Tokenizer Tokenizer::from_gguf(const GgufModel& g) {
  Tokenizer t;
  std::string model = g.get_string("tokenizer.ggml.model").value_or("llama");
  if (model == "gpt2" || model == "bpe") {
    t.kind_ = Kind::BPE;
  } else if (model == "llama") {
    t.kind_ = Kind::SPM;
  } else {
    throw std::runtime_error("tokenizer: unsupported tokenizer.ggml.model '" +
                             model + "' (supported: llama, gpt2/bpe)");
  }

  const GgufMetadataValue* toks = g.get_array("tokenizer.ggml.tokens");
  if (!toks) throw std::runtime_error("tokenizer: missing tokenizer.ggml.tokens");
  t.id_to_piece_.reserve(toks->array.size());
  for (size_t i = 0; i < toks->array.size(); ++i) {
    const std::string& s = toks->array[i].str;
    t.id_to_piece_.push_back(s);
    t.piece_to_id_.emplace(s, static_cast<int32_t>(i));
  }
  if (const GgufMetadataValue* sc = g.get_array("tokenizer.ggml.scores")) {
    t.scores_.reserve(sc->array.size());
    for (const auto& e : sc->array) t.scores_.push_back(static_cast<float>(e.f));
  }
  t.scores_.resize(t.id_to_piece_.size(), 0.0f);
  if (const GgufMetadataValue* tt = g.get_array("tokenizer.ggml.token_type")) {
    t.token_types_.reserve(tt->array.size());
    for (const auto& e : tt->array)
      t.token_types_.push_back(static_cast<int32_t>(e.i));
  }
  t.token_types_.resize(t.id_to_piece_.size(), 1);

  if (t.kind_ == Kind::BPE) {
    if (const GgufMetadataValue* mg = g.get_array("tokenizer.ggml.merges")) {
      for (size_t r = 0; r < mg->array.size(); ++r)
        t.bpe_ranks_.emplace(mg->array[r].str, static_cast<int32_t>(r));
    }
  }

  auto id = [&](const char* key) -> int64_t {
    auto v = g.get_u32(key);
    return v ? static_cast<int64_t>(*v) : -1;
  };
  t.bos_id_ = id("tokenizer.ggml.bos_token_id");
  t.eos_id_ = id("tokenizer.ggml.eos_token_id");
  t.unk_id_ = id("tokenizer.ggml.unknown_token_id");
  t.eot_id_ = id("tokenizer.ggml.eot_token_id");
  // SPM default adds a leading space + BOS; gpt2/BPE does neither by default.
  t.add_space_prefix_ = (t.kind_ == Kind::SPM);
  t.wants_bos_ = (t.kind_ == Kind::SPM);
  // Honor an explicit tokenizer.ggml.add_bos_token if the loader exposes it
  // (stored as a bool; get_u32 returns it as 0/1 on backends that widen bools).
  if (auto v = g.get_u32("tokenizer.ggml.add_bos_token")) t.wants_bos_ = (*v != 0);
  return t;
}

bool Tokenizer::is_eog(Token id) const {
  int64_t i = static_cast<int64_t>(id);
  return i == eos_id_ || (eot_id_ >= 0 && i == eot_id_);
}

// --- SentencePiece (llama) -------------------------------------------------

std::vector<Token> Tokenizer::encode_spm(const std::string& text) const {
  // Normalize: optional leading space, then every ' ' -> "▁".
  std::string norm;
  norm.reserve(text.size() + 4);
  if (add_space_prefix_) norm += kSpace;
  for (char c : text) {
    if (c == ' ')
      norm += kSpace;
    else
      norm.push_back(c);
  }

  struct Sym {
    size_t off, len;
    int prev, next;
  };
  std::vector<Sym> syms;
  for (size_t i = 0; i < norm.size();) {
    size_t l = utf8_len(static_cast<unsigned char>(norm[i]));
    if (i + l > norm.size()) l = 1;
    int idx = static_cast<int>(syms.size());
    syms.push_back({i, l, idx - 1, -1});
    i += l;
  }
  for (size_t i = 0; i + 1 < syms.size(); ++i) syms[i].next = static_cast<int>(i + 1);

  struct Bigram {
    int left, right;
    float score;
    size_t size;
  };
  auto cmp = [](const Bigram& a, const Bigram& b) {
    return a.score < b.score || (a.score == b.score && a.left > b.left);
  };
  std::priority_queue<Bigram, std::vector<Bigram>, decltype(cmp)> q(cmp);

  auto piece_of = [&](int s) { return norm.substr(syms[s].off, syms[s].len); };
  auto try_add = [&](int left, int right) {
    if (left < 0 || right < 0) return;
    std::string merged = piece_of(left) + piece_of(right);
    int32_t pid = piece_id(merged);
    if (pid < 0) return;
    q.push({left, right, scores_[pid], merged.size()});
  };
  for (size_t i = 0; i + 1 < syms.size(); ++i)
    try_add(static_cast<int>(i), static_cast<int>(i + 1));

  while (!q.empty()) {
    Bigram b = q.top();
    q.pop();
    Sym& l = syms[b.left];
    if (l.len == 0) continue;
    if (b.right < 0 || syms[b.right].len == 0) continue;
    if (l.len + syms[b.right].len != b.size) continue;  // stale
    // Merge right into left.
    l.len += syms[b.right].len;
    syms[b.right].len = 0;
    l.next = syms[b.right].next;
    if (syms[b.right].next >= 0) syms[syms[b.right].next].prev = b.left;
    try_add(l.prev, b.left);
    try_add(b.left, l.next);
  }

  std::vector<Token> out;
  for (int s = 0; s >= 0 && s < static_cast<int>(syms.size()); s = syms[s].next) {
    if (syms[s].len == 0) continue;
    std::string piece = piece_of(s);
    int32_t pid = piece_id(piece);
    if (pid >= 0) {
      out.push_back(static_cast<Token>(pid));
    } else {
      // Byte fallback: emit <0xXX> per raw byte (or unk).
      for (unsigned char ch : piece) {
        char buf[7];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", ch);
        int32_t bid = piece_id(buf);
        if (bid >= 0)
          out.push_back(static_cast<Token>(bid));
        else if (unk_id_ >= 0)
          out.push_back(static_cast<Token>(unk_id_));
      }
    }
  }
  return out;
}

// --- byte-level BPE (gpt2/qwen) --------------------------------------------

namespace {

// Approximate the GPT-2 pretokenizer: contractions, then runs of letters /
// digits / other, each optionally led by a single space; whitespace runs.
// Codepoint classes use ASCII rules and treat >=0x80 as "letter".
enum class Cls { Space, Letter, Digit, Other };
Cls classify(uint32_t cp) {
  if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0b ||
      cp == 0x0c)
    return Cls::Space;
  if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || cp >= 0x80)
    return Cls::Letter;
  if (cp >= '0' && cp <= '9') return Cls::Digit;
  return Cls::Other;
}

uint32_t decode_cp(const std::string& s, size_t i, size_t& len) {
  unsigned char c = s[i];
  len = utf8_len(c);
  if (len == 1) return c;
  if (i + len > s.size()) {
    len = 1;
    return c;
  }
  uint32_t cp = 0;
  if (len == 2)
    cp = (c & 0x1f) << 6 | (s[i + 1] & 0x3f);
  else if (len == 3)
    cp = (c & 0x0f) << 12 | (s[i + 1] & 0x3f) << 6 | (s[i + 2] & 0x3f);
  else
    cp = (c & 0x07) << 18 | (s[i + 1] & 0x3f) << 12 | (s[i + 2] & 0x3f) << 6 |
         (s[i + 3] & 0x3f);
  return cp;
}

std::vector<std::string> pretokenize_gpt2(const std::string& text) {
  // GPT-2 pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+
  //                |\s+(?!\S)|\s+
  std::vector<std::string> words;
  size_t i = 0, n = text.size();
  auto starts_with = [&](size_t p, const char* s) {
    size_t l = std::strlen(s);
    return p + l <= n && text.compare(p, l, s) == 0;
  };
  while (i < n) {
    // Contractions.
    if (text[i] == '\'') {
      const char* ctr[] = {"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"};
      bool matched = false;
      for (const char* c : ctr) {
        if (starts_with(i, c)) {
          words.emplace_back(c);
          i += std::strlen(c);
          matched = true;
          break;
        }
      }
      if (matched) continue;
    }

    // " ?<class>+": at most one leading space, then a run of one class.
    size_t p = (text[i] == ' ') ? i + 1 : i;
    if (p < n) {
      size_t cl;
      Cls cls = classify(decode_cp(text, p, cl));
      if (cls != Cls::Space) {
        size_t start = i;
        i = p;
        while (i < n) {
          size_t l2;
          if (classify(decode_cp(text, i, l2)) != cls) break;
          i += l2;
        }
        words.emplace_back(text.substr(start, i - start));
        continue;
      }
    }

    // Whitespace run (the single-leading-space case is handled above).
    size_t k = i;
    while (k < n) {
      size_t l3;
      if (classify(decode_cp(text, k, l3)) != Cls::Space) break;
      k += l3;
    }
    words.emplace_back(text.substr(i, k - i));
    i = k;
  }
  return words;
}

}  // namespace

std::vector<Token> Tokenizer::encode_bpe(const std::string& text) const {
  const auto& b2u = byte_to_unicode();
  std::vector<Token> out;
  for (const std::string& word : pretokenize_gpt2(text)) {
    // Byte-level: each byte -> its unicode char (one BPE symbol).
    std::vector<std::string> syms;
    for (unsigned char c : word) syms.push_back(b2u[c]);
    if (syms.empty()) continue;

    // Merge by lowest rank until none applies.
    while (syms.size() > 1) {
      int best_rank = INT32_MAX;
      size_t best = SIZE_MAX;
      for (size_t k = 0; k + 1 < syms.size(); ++k) {
        auto it = bpe_ranks_.find(syms[k] + " " + syms[k + 1]);
        if (it != bpe_ranks_.end() && it->second < best_rank) {
          best_rank = it->second;
          best = k;
        }
      }
      if (best == SIZE_MAX) break;
      syms[best] += syms[best + 1];
      syms.erase(syms.begin() + best + 1);
    }
    for (const std::string& s : syms) {
      int32_t pid = piece_id(s);
      if (pid >= 0)
        out.push_back(static_cast<Token>(pid));
      else if (unk_id_ >= 0)
        out.push_back(static_cast<Token>(unk_id_));
    }
  }
  return out;
}

std::vector<Token> Tokenizer::encode(const std::string& text, bool add_bos) const {
  std::vector<Token> ids;
  if (add_bos && wants_bos_ && bos_id_ >= 0)
    ids.push_back(static_cast<Token>(bos_id_));
  std::vector<Token> body =
      kind_ == Kind::SPM ? encode_spm(text) : encode_bpe(text);
  ids.insert(ids.end(), body.begin(), body.end());
  return ids;
}

std::string Tokenizer::decode_token(Token id) const {
  if (id >= id_to_piece_.size()) return "";
  const std::string& p = id_to_piece_[id];
  if (kind_ == Kind::SPM) {
    uint8_t byte;
    if (is_byte_token(p, byte)) return std::string(1, static_cast<char>(byte));
    std::string out;
    for (size_t i = 0; i < p.size();) {
      if (p.compare(i, kSpace.size(), kSpace) == 0) {
        out.push_back(' ');
        i += kSpace.size();
      } else {
        out.push_back(p[i]);
        ++i;
      }
    }
    return out;
  }
  // BPE: map each byte-level-unicode char back to its raw byte.
  const auto& u2b = unicode_to_byte();
  std::string out;
  for (size_t i = 0; i < p.size();) {
    size_t l = utf8_len(static_cast<unsigned char>(p[i]));
    if (i + l > p.size()) l = 1;
    auto it = u2b.find(p.substr(i, l));
    if (it != u2b.end())
      out.push_back(static_cast<char>(it->second));
    else
      out += p.substr(i, l);
    i += l;
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<Token>& ids) const {
  std::string out;
  for (Token id : ids) out += decode_token(id);
  // SPM adds a leading space from the prefix; strip one for readability.
  if (kind_ == Kind::SPM && !out.empty() && out[0] == ' ') out.erase(0, 1);
  return out;
}

}  // namespace oxidize
