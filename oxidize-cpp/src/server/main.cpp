// oxidize-cpp-server: a minimal OpenAI-compatible HTTP server wrapping the C++
// engine. Dependency-free (POSIX sockets + the local json parser).
//
// Endpoints:
//   GET  /health                 -> {"status":"ok"}
//   GET  /v1/models              -> {"object":"list","data":[{id,...}]}
//   POST /v1/chat/completions     -> OpenAI chat completion (stream or not)
//
// Inference is serialized behind a mutex (the model holds mutable KV state), so
// this is a correct reference server, not a high-concurrency one. Chat messages
// are rendered with a simple generic template; per-model Jinja chat templates
// are a follow-up.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"
#include "oxidize/model.hpp"
#include "oxidize/model_llama.hpp"
#include "oxidize/sampler.hpp"
#include "oxidize/tokenizer.hpp"

namespace {

using oxidize::LlamaModel;
using oxidize::Logits;
using oxidize::Model;
using oxidize::Session;
using oxidize::Token;
using oxidize::Tokenizer;
namespace json = oxidize::json;

struct ServerArgs {
  std::string model;
  std::string host = "127.0.0.1";
  int port = 8080;
  size_t default_max_tokens = 256;
};

// Render chat messages into a single prompt. Generic template (not model Jinja):
//   <role>: <content>\n ... \nassistant:
std::string render_messages(const json::Value& messages) {
  std::string prompt;
  if (messages.is_array()) {
    for (const auto& m : *messages.arr) {
      std::string role = m.get_str("role", "user");
      std::string content = m.get_str("content");
      prompt += role + ": " + content + "\n";
    }
  }
  prompt += "assistant:";
  return prompt;
}

// One global engine + lock (the model has mutable per-step state).
struct Engine {
  std::unique_ptr<Model> model;
  std::unique_ptr<Tokenizer> tok;
  std::mutex mu;
  std::string model_id;
  size_t vocab = 0, ctx = 0;
};

// Generate a completion string for `prompt`. Caller holds engine.mu.
std::string generate(Engine& eng, const std::string& prompt, size_t max_tokens,
                     float temperature, size_t top_k, float top_p, uint64_t seed,
                     size_t& prompt_toks, size_t& completion_toks) {
  std::vector<Token> ids = eng.tok->encode(prompt, /*add_bos=*/true);
  prompt_toks = ids.size();
  Session session;
  Logits logits = eng.model->forward(ids, session);

  oxidize::SamplerConfig scfg;
  scfg.temperature = temperature > 0.0f ? temperature : 1.0f;
  if (top_k > 0) scfg.top_k = top_k;
  if (top_p < 1.0f) scfg.top_p = top_p;
  const bool greedy = temperature <= 0.0f;
  oxidize::Rng rng(seed);

  std::vector<Token> out;
  size_t room = eng.ctx ? std::min(max_tokens, eng.ctx - ids.size()) : max_tokens;
  for (size_t step = 0; step < room; ++step) {
    Token next = greedy ? oxidize::greedy(logits)
                        : oxidize::sample(logits, scfg, rng.next_unit());
    if (eng.tok->is_eog(next)) break;
    out.push_back(next);
    if (step + 1 < room) logits = eng.model->forward({next}, session);
  }
  completion_toks = out.size();
  return eng.tok->decode(out);
}


void send_all(int fd, const std::string& data) {
  size_t sent = 0;
  while (sent < data.size()) {
    ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, 0);
    if (n <= 0) return;
    sent += static_cast<size_t>(n);
  }
}

void send_response(int fd, int code, const char* status, const std::string& body,
                   const char* ctype = "application/json") {
  std::string head = "HTTP/1.1 " + std::to_string(code) + " " + status + "\r\n";
  head += "Content-Type: " + std::string(ctype) + "\r\n";
  head += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  head += "Access-Control-Allow-Origin: *\r\n";
  head += "Connection: close\r\n\r\n";
  send_all(fd, head + body);
}

json::Value make_chat_response(const std::string& model_id,
                               const std::string& content, size_t ptoks,
                               size_t ctoks, long created) {
  json::Value msg = json::Value::object();
  msg.set("role", json::Value::string("assistant"));
  msg.set("content", json::Value::string(content));
  json::Value choice = json::Value::object();
  choice.set("index", json::Value::number(0));
  choice.set("message", msg);
  choice.set("finish_reason", json::Value::string("stop"));
  json::Value choices = json::Value::array();
  choices.arr->push_back(choice);
  json::Value usage = json::Value::object();
  usage.set("prompt_tokens", json::Value::number((double)ptoks));
  usage.set("completion_tokens", json::Value::number((double)ctoks));
  usage.set("total_tokens", json::Value::number((double)(ptoks + ctoks)));
  json::Value root = json::Value::object();
  root.set("id", json::Value::string("chatcmpl-oxcpp"));
  root.set("object", json::Value::string("chat.completion"));
  root.set("created", json::Value::number((double)created));
  root.set("model", json::Value::string(model_id));
  root.set("choices", choices);
  root.set("usage", usage);
  return root;
}

void handle_chat(int fd, Engine& eng, const std::string& body, long created) {
  json::Value req;
  try {
    req = json::parse(body);
  } catch (const std::exception& e) {
    send_response(fd, 400, "Bad Request",
                  "{\"error\":{\"message\":\"invalid JSON\"}}");
    return;
  }
  const json::Value* messages = req.find("messages");
  if (!messages || !messages->is_array()) {
    send_response(fd, 400, "Bad Request",
                  "{\"error\":{\"message\":\"missing messages\"}}");
    return;
  }
  std::string prompt = render_messages(*messages);
  size_t max_tokens = (size_t)req.get_num("max_tokens", 256);
  float temperature = (float)req.get_num("temperature", 0.0);
  float top_p = (float)req.get_num("top_p", 1.0);
  size_t top_k = (size_t)req.get_num("top_k", 0);
  uint64_t seed = (uint64_t)req.get_num("seed", 0);
  bool stream = req.get_bool("stream", false);

  std::lock_guard<std::mutex> lock(eng.mu);
  size_t ptoks = 0, ctoks = 0;

  if (!stream) {
    std::string text = generate(eng, prompt, max_tokens, temperature, top_k,
                                top_p, seed, ptoks, ctoks);
    send_response(fd, 200, "OK",
                  json::dump(make_chat_response(eng.model_id, text, ptoks,
                                                ctoks, created)));
    return;
  }

  // Streaming: emit the full completion as one SSE delta chunk + [DONE].
  // (Token-by-token SSE is a follow-up; this keeps the wire format valid.)
  std::string text = generate(eng, prompt, max_tokens, temperature, top_k, top_p,
                              seed, ptoks, ctoks);
  std::string head =
      "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
      "Cache-Control: no-cache\r\nConnection: close\r\n"
      "Access-Control-Allow-Origin: *\r\n\r\n";
  send_all(fd, head);
  json::Value delta = json::Value::object();
  delta.set("role", json::Value::string("assistant"));
  delta.set("content", json::Value::string(text));
  json::Value choice = json::Value::object();
  choice.set("index", json::Value::number(0));
  choice.set("delta", delta);
  json::Value chunk = json::Value::object();
  chunk.set("object", json::Value::string("chat.completion.chunk"));
  chunk.set("model", json::Value::string(eng.model_id));
  json::Value choices = json::Value::array();
  choices.arr->push_back(choice);
  chunk.set("choices", choices);
  send_all(fd, "data: " + json::dump(chunk) + "\n\n");
  send_all(fd, "data: [DONE]\n\n");
}

void handle_client(int fd, Engine& eng) {
  std::string req;
  char buf[8192];
  // Read headers.
  size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) { ::close(fd); return; }
    req.append(buf, (size_t)n);
    header_end = req.find("\r\n\r\n");
    if (req.size() > 16 * 1024 * 1024) { ::close(fd); return; }
  }
  // Parse request line + Content-Length.
  std::string line = req.substr(0, req.find("\r\n"));
  std::string method, path;
  {
    size_t s1 = line.find(' ');
    size_t s2 = line.find(' ', s1 + 1);
    if (s1 != std::string::npos && s2 != std::string::npos) {
      method = line.substr(0, s1);
      path = line.substr(s1 + 1, s2 - s1 - 1);
    }
  }
  size_t content_length = 0;
  {
    std::string lower = req;
    for (char& c : lower) c = (char)std::tolower((unsigned char)c);
    size_t cl = lower.find("content-length:");
    if (cl != std::string::npos)
      content_length = (size_t)std::strtoul(req.c_str() + cl + 15, nullptr, 10);
  }
  std::string body = req.substr(header_end + 4);
  while (body.size() < content_length) {
    ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) break;
    body.append(buf, (size_t)n);
  }

  long created = (long)::time(nullptr);
  if (method == "GET" && path == "/health") {
    send_response(fd, 200, "OK", "{\"status\":\"ok\"}");
  } else if (method == "GET" && path == "/v1/models") {
    json::Value m = json::Value::object();
    m.set("id", json::Value::string(eng.model_id));
    m.set("object", json::Value::string("model"));
    m.set("owned_by", json::Value::string("oxidize-cpp"));
    json::Value data = json::Value::array();
    data.arr->push_back(m);
    json::Value root = json::Value::object();
    root.set("object", json::Value::string("list"));
    root.set("data", data);
    send_response(fd, 200, "OK", json::dump(root));
  } else if (method == "POST" && path == "/v1/chat/completions") {
    handle_chat(fd, eng, body, created);
  } else if (method == "OPTIONS") {
    send_response(fd, 204, "No Content", "");
  } else {
    send_response(fd, 404, "Not Found",
                  "{\"error\":{\"message\":\"not found\"}}");
  }
  ::close(fd);
}

ServerArgs parse_args(int argc, char** argv) {
  ServerArgs a;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--model") a.model = next();
    else if (arg == "--host") a.host = next();
    else if (arg == "--port") a.port = std::atoi(next().c_str());
    else if (arg == "--max-tokens") a.default_max_tokens = (size_t)std::atoll(next().c_str());
    else if (arg == "-h" || arg == "--help") {
      std::printf("Usage: oxidize-cpp-server --model <gguf> [--host H] [--port P]\n");
      std::exit(0);
    }
  }
  if (a.model.empty()) {
    std::fprintf(stderr, "error: --model is required\n");
    std::exit(2);
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  ServerArgs args = parse_args(argc, argv);

  Engine eng;
  try {
    eng.model = oxidize::load_llama_gguf(args.model, /*want_cuda=*/false);
    if (auto* lm = dynamic_cast<LlamaModel*>(eng.model.get()))
      eng.tok = std::make_unique<Tokenizer>(Tokenizer::from_gguf(lm->gguf()));
    if (!eng.tok) throw std::runtime_error("no tokenizer in GGUF");
    eng.vocab = eng.model->vocab_size();
    eng.ctx = eng.model->context_size();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: failed to load model: %s\n", e.what());
    return 1;
  }
  eng.model_id = args.model.substr(args.model.find_last_of('/') + 1);

  int srv = ::socket(AF_INET, SOCK_STREAM, 0);
  if (srv < 0) { std::perror("socket"); return 1; }
  int opt = 1;
  ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons((uint16_t)args.port);
  if (::inet_pton(AF_INET, args.host.c_str(), &addr.sin_addr) <= 0) {
    std::fprintf(stderr, "error: bad host %s\n", args.host.c_str());
    return 1;
  }
  if (::bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0) {
    std::perror("bind");
    return 1;
  }
  if (::listen(srv, 16) < 0) { std::perror("listen"); return 1; }
  std::printf("oxidize-cpp-server: %s on http://%s:%d (model=%s, ctx=%zu)\n",
              "OpenAI-compatible", args.host.c_str(), args.port,
              eng.model_id.c_str(), eng.ctx);
  std::fflush(stdout);

  while (true) {
    int fd = ::accept(srv, nullptr, nullptr);
    if (fd < 0) continue;
    // Thread per connection; inference itself is serialized by eng.mu.
    std::thread(handle_client, fd, std::ref(eng)).detach();
  }
  return 0;
}
