// Data pipeline: JSONL parse + Qwen2.5 chat template + tokenization + loss mask.

#include "oxidize/train_data.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace oxidize {

// Minimal JSON string parser (handles escape sequences).
static std::string parse_json_string(const std::string& s, size_t& pos) {
  if (pos >= s.size() || s[pos] != '"') {
    throw std::runtime_error("JSON: expected '\"' at pos " + std::to_string(pos));
  }
  ++pos;
  std::string out;
  while (pos < s.size() && s[pos] != '"') {
    if (s[pos] == '\\' && pos + 1 < s.size()) {
      ++pos;
      char esc = s[pos];
      switch (esc) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/':  out += '/';  break;
        case 'n':  out += '\n'; break;
        case 'r':  out += '\r'; break;
        case 't':  out += '\t'; break;
        default:   out += esc;  break;
      }
    } else {
      out += s[pos];
    }
    ++pos;
  }
  if (pos >= s.size()) throw std::runtime_error("JSON: unterminated string");
  ++pos;  // consume closing '"'
  return out;
}

static void skip_whitespace(const std::string& s, size_t& pos) {
  while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                             s[pos] == '\r' || s[pos] == '\n')) ++pos;
}

static bool consume(const std::string& s, size_t& pos, char c) {
  skip_whitespace(s, pos);
  if (pos < s.size() && s[pos] == c) { ++pos; return true; }
  return false;
}

static std::string find_string_key(const std::string& s, size_t& pos,
                                    const std::string& key) {
  // Find "key" : "value" pattern from current pos.
  size_t search = s.find("\"" + key + "\"", pos);
  if (search == std::string::npos) return "";
  pos = search + key.size() + 2;
  skip_whitespace(s, pos);
  if (pos >= s.size() || s[pos] != ':') return "";
  ++pos;
  skip_whitespace(s, pos);
  if (pos >= s.size() || s[pos] != '"') return "";
  return parse_json_string(s, pos);
}

// Parse one JSONL line into a list of ChatMessages.
static std::vector<ChatMessage> parse_jsonl_line(const std::string& line) {
  std::vector<ChatMessage> msgs;
  // Find "messages" array.
  size_t pos = 0;
  size_t msgs_start = line.find("\"messages\"");
  if (msgs_start == std::string::npos) return msgs;
  pos = msgs_start + 10;  // after "messages"
  skip_whitespace(line, pos);
  if (!consume(line, pos, ':')) return msgs;
  skip_whitespace(line, pos);
  if (!consume(line, pos, '[')) return msgs;

  // Iterate objects in array.
  while (true) {
    skip_whitespace(line, pos);
    if (pos >= line.size()) break;
    if (line[pos] == ']') break;
    if (!consume(line, pos, '{')) break;

    std::string role, content;
    // Parse key-value pairs until '}'.
    while (true) {
      skip_whitespace(line, pos);
      if (pos >= line.size() || line[pos] == '}') break;
      if (line[pos] == ',') { ++pos; continue; }
      if (line[pos] != '"') { ++pos; continue; }
      std::string key = parse_json_string(line, pos);
      skip_whitespace(line, pos);
      if (!consume(line, pos, ':')) break;
      skip_whitespace(line, pos);
      if (pos < line.size() && line[pos] == '"') {
        std::string val = parse_json_string(line, pos);
        if (key == "role") role = val;
        else if (key == "content") content = val;
      } else {
        // Skip non-string value.
        while (pos < line.size() && line[pos] != ',' && line[pos] != '}') ++pos;
      }
    }
    consume(line, pos, '}');

    if (!role.empty() && !content.empty()) {
      msgs.push_back({role, content});
    }

    skip_whitespace(line, pos);
    if (!consume(line, pos, ',')) {}  // ignore missing comma
  }
  return msgs;
}

// Qwen2.5 chat template tokens.
// Template: <|im_start|>{role}\n{content}<|im_end|>\n
// Loss mask 1 on tokens AFTER the \n following <|im_start|>assistant up to
// (but not including) <|im_end|>.
TrainSample build_chat_sample(const std::vector<ChatMessage>& messages,
                               const Tokenizer& tok,
                               size_t max_seq_len) {
  // Find special token ids.
  // Qwen2.5 uses BPE. We look up the pieces directly.
  auto encode_piece = [&](const std::string& text) -> std::vector<Token> {
    return tok.encode(text, /*add_bos=*/false);
  };

  // Build the full chat string with markers to track mask regions.
  TrainSample sample;

  for (const auto& msg : messages) {
    // Encode header: <|im_start|>{role}\n
    std::string header = "<|im_start|>" + msg.role + "\n";
    auto header_toks = encode_piece(header);
    for (Token t : header_toks) {
      sample.tokens.push_back(t);
      sample.loss_mask.push_back(0.0f);  // no loss on header
    }

    // Encode content.
    auto content_toks = encode_piece(msg.content);
    bool is_assistant = (msg.role == "assistant");
    for (Token t : content_toks) {
      sample.tokens.push_back(t);
      sample.loss_mask.push_back(is_assistant ? 1.0f : 0.0f);
    }

    // Encode footer: <|im_end|>\n
    auto footer_toks = encode_piece("<|im_end|>\n");
    for (Token t : footer_toks) {
      sample.tokens.push_back(t);
      sample.loss_mask.push_back(0.0f);
    }
  }

  // Truncate.
  if (sample.tokens.size() > max_seq_len) {
    sample.tokens.resize(max_seq_len);
    sample.loss_mask.resize(max_seq_len);
  }

  // For causal LM, the target at position i is tokens[i+1], so the effective
  // loss mask is shifted: we lose the last position.
  // We handle this in training by computing loss on tokens[1..T] with target=tokens[1..T].
  // The last token has no target, so we trim one from the mask tail implicitly
  // in the training loop.

  return sample;
}

std::vector<TrainSample> load_jsonl_samples(const std::string& path,
                                             const Tokenizer& tok,
                                             size_t max_seq_len) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Cannot open data file: " + path);

  std::vector<TrainSample> samples;
  std::string line;
  size_t line_no = 0;
  while (std::getline(f, line)) {
    ++line_no;
    if (line.empty() || line[0] == '#') continue;
    try {
      auto msgs = parse_jsonl_line(line);
      if (msgs.empty()) continue;
      auto s = build_chat_sample(msgs, tok, max_seq_len);
      if (s.tokens.size() < 2) continue;  // need at least one input+target pair
      samples.push_back(std::move(s));
    } catch (const std::exception& e) {
      // Skip malformed lines.
      (void)e;
    }
  }
  return samples;
}

DataLoader::DataLoader(std::vector<TrainSample> samples, size_t batch_size,
                       uint64_t seed)
    : samples_(std::move(samples)), batch_size_(batch_size), rng_(seed) {
  order_.resize(samples_.size());
  std::iota(order_.begin(), order_.end(), 0);
  reshuffle();
}

void DataLoader::reshuffle() {
  std::shuffle(order_.begin(), order_.end(), rng_);
  pos_ = 0;
}

const TrainSample& DataLoader::next_sample() {
  if (samples_.empty()) throw std::runtime_error("DataLoader: no samples");
  if (pos_ >= order_.size()) reshuffle();
  return samples_[order_[pos_++]];
}

}  // namespace oxidize
