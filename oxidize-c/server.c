/* Minimal OpenAI-compatible HTTP server for oxidize-c.
 * POST /v1/chat/completions {messages|prompt, max_tokens, temperature, stream}
 * GET  /health
 * One request at a time (single model instance, sequential decode); requests
 * queue on the listen backlog. ChatML prompt formatting (qwen convention). */
/* Must precede all includes: strcasestr is a GNU extension. Without this the
 * compiler implicitly declares it returning int, TRUNCATING the returned
 * pointer to 32 bits -> segfault on the first request that parses headers.
 * Belt-and-suspenders with the Makefile's -D_GNU_SOURCE (some toolchains here
 * don't apply it to this TU). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "oc.h"
#include "gen.h"
#include "batch.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static int g_spec_mode; /* drafting mode for all requests */
static volatile sig_atomic_t g_stop = 0;

typedef struct batch_runtime batch_runtime;

typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t ready;
  size_t active;
  size_t waiting;
  size_t max_seqs;
  size_t max_queue;
} admission_gate;

typedef struct {
  oc_model *model;
  oc_tokenizer *tokenizer;
  float temperature;
  size_t draft_k;
  const char *worker_id;
  batch_runtime *batch;
  admission_gate gate;
} serve_context;

typedef struct {
  int fd;
  serve_context *server;
} client_context;

typedef struct batch_request {
  uint64_t request_id;
  uint32_t *prompt_tokens;
  uint32_t *output_tokens;
  size_t prompt_count;
  size_t max_tokens;
  size_t output_count;
  size_t output_capacity;
  size_t generated;
  bool done;
  bool failed;
  pthread_mutex_t mutex;
  pthread_cond_t completed;
  struct batch_request *next;
} batch_request;

struct batch_runtime {
  pthread_mutex_t mutex;
  pthread_cond_t work;
  batch_request *head;
  batch_request *tail;
  size_t admitted;
  size_t capacity;
  size_t queue_capacity;
  uint64_t next_request_id;
  oc_batch_scheduler *scheduler;
  oc_model *model;
  oc_tokenizer *tokenizer;
  pthread_t worker;
};

static bool admission_acquire(admission_gate *gate) {
  bool admitted = true;
  pthread_mutex_lock(&gate->mutex);
  if (gate->active == gate->max_seqs) {
    if (gate->waiting == gate->max_queue) {
      admitted = false;
    } else {
      ++gate->waiting;
      while (gate->active == gate->max_seqs)
        pthread_cond_wait(&gate->ready, &gate->mutex);
      --gate->waiting;
      ++gate->active;
    }
  } else {
    ++gate->active;
  }
  pthread_mutex_unlock(&gate->mutex);
  return admitted;
}

static void admission_release(admission_gate *gate) {
  pthread_mutex_lock(&gate->mutex);
  --gate->active;
  pthread_cond_signal(&gate->ready);
  pthread_mutex_unlock(&gate->mutex);
}

#ifdef OC_CUDA
static void batch_request_free(batch_request *request) {
  if (!request) return;
  pthread_cond_destroy(&request->completed);
  pthread_mutex_destroy(&request->mutex);
  free(request->output_tokens);
  free(request->prompt_tokens);
  free(request);
}

static bool batch_request_append(batch_request *request, uint32_t token) {
  bool appended = false;
  pthread_mutex_lock(&request->mutex);
  if (!request->done) {
    if (request->output_count == request->output_capacity) {
      size_t capacity = request->output_capacity ? request->output_capacity * 2 : 64;
      uint32_t *tokens = realloc(request->output_tokens, capacity * sizeof(*tokens));
      if (tokens) {
        request->output_tokens = tokens;
        request->output_capacity = capacity;
      }
    }
    if (request->output_count < request->output_capacity) {
      request->output_tokens[request->output_count++] = token;
      appended = true;
    }
  }
  pthread_mutex_unlock(&request->mutex);
  return appended;
}

static void batch_request_finish(batch_request *request, bool failed) {
  pthread_mutex_lock(&request->mutex);
  request->failed = failed;
  request->done = true;
  pthread_cond_signal(&request->completed);
  pthread_mutex_unlock(&request->mutex);
}

static bool batch_request_wait(batch_request *request) {
  pthread_mutex_lock(&request->mutex);
  while (!request->done) pthread_cond_wait(&request->completed, &request->mutex);
  bool failed = request->failed;
  pthread_mutex_unlock(&request->mutex);
  return !failed;
}

static void batch_release_admission(batch_runtime *runtime) {
  pthread_mutex_lock(&runtime->mutex);
  --runtime->admitted;
  pthread_cond_signal(&runtime->work);
  pthread_mutex_unlock(&runtime->mutex);
}

static bool batch_enqueue(batch_runtime *runtime, batch_request *request) {
  bool accepted = false;
  pthread_mutex_lock(&runtime->mutex);
  if (runtime->admitted < runtime->capacity ||
      runtime->admitted - runtime->capacity < runtime->queue_capacity) {
    request->request_id = ++runtime->next_request_id;
    if (!request->request_id) request->request_id = ++runtime->next_request_id;
    if (runtime->tail) runtime->tail->next = request;
    else runtime->head = request;
    runtime->tail = request;
    ++runtime->admitted;
    pthread_cond_signal(&runtime->work);
    accepted = true;
  }
  pthread_mutex_unlock(&runtime->mutex);
  return accepted;
}

static batch_request *batch_take(batch_runtime *runtime) {
  pthread_mutex_lock(&runtime->mutex);
  batch_request *request = runtime->head;
  if (request) {
    runtime->head = request->next;
    if (!runtime->head) runtime->tail = NULL;
    request->next = NULL;
  }
  pthread_mutex_unlock(&runtime->mutex);
  return request;
}

static void batch_wait_for_work(batch_runtime *runtime) {
  pthread_mutex_lock(&runtime->mutex);
  while (!runtime->head && !g_stop) pthread_cond_wait(&runtime->work, &runtime->mutex);
  pthread_mutex_unlock(&runtime->mutex);
}
#endif

/* ---- tiny tolerant JSON helpers ---- */

/* Decode a JSON string starting at the opening quote; returns malloc'd UTF-8
 * and sets *end to just past the closing quote. NULL on malformed. */
static char *json_string(const char *p, const char **end) {
  if (*p != '"') return NULL;
  ++p;
  size_t cap = 256, n = 0;
  char *out = malloc(cap);
  while (*p && *p != '"') {
    char c = *p++;
    if (c == '\\' && *p) {
      char e = *p++;
      switch (e) {
        case 'n': c = '\n'; break;
        case 't': c = '\t'; break;
        case 'r': c = '\r'; break;
        case 'b': c = '\b'; break;
        case 'f': c = '\f'; break;
        case 'u': {
          unsigned cp = 0;
          if (sscanf(p, "%4x", &cp) == 1) p += 4;
          if (cp < 0x80) {
            c = (char)cp;
          } else {
            /* encode BMP codepoint as UTF-8 */
            if (n + 4 >= cap) out = realloc(out, cap *= 2);
            if (cp < 0x800) {
              out[n++] = (char)(0xC0 | (cp >> 6));
              c = (char)(0x80 | (cp & 0x3F));
            } else {
              out[n++] = (char)(0xE0 | (cp >> 12));
              out[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
              c = (char)(0x80 | (cp & 0x3F));
            }
          }
          break;
        }
        default: c = e; break;
      }
    }
    if (n + 2 >= cap) out = realloc(out, cap *= 2);
    out[n++] = c;
  }
  if (*p != '"') { free(out); return NULL; }
  out[n] = 0;
  if (end) *end = p + 1;
  return out;
}

/* find "key": and return pointer to the value start (or NULL) */
static const char *json_value(const char *body, const char *key) {
  char pat[64];
  snprintf(pat, sizeof pat, "\"%s\"", key);
  const char *p = body;
  while ((p = strstr(p, pat)) != NULL) {
    const char *q = p + strlen(pat);
    while (*q == ' ' || *q == '\t' || *q == '\n') ++q;
    if (*q == ':') {
      ++q;
      while (*q == ' ' || *q == '\t' || *q == '\n') ++q;
      return q;
    }
    p = q;
  }
  return NULL;
}

static double json_number(const char *body, const char *key, double dflt) {
  const char *v = json_value(body, key);
  return v ? strtod(v, NULL) : dflt;
}

static bool json_size_t_bounded(const char *body, const char *key, size_t dflt,
                                size_t min_v, size_t max_v, size_t *out) {
  const char *v = json_value(body, key);
  if (!v) {
    *out = dflt;
    return true;
  }
  char *end = NULL;
  double dv = strtod(v, &end);
  if (v == end || !isfinite(dv) || dv < (double)min_v || dv > (double)max_v) return false;
  *out = (size_t)dv;
  return true;
}

static bool json_bool(const char *body, const char *key, bool dflt) {
  const char *v = json_value(body, key);
  if (!v) return dflt;
  return strncmp(v, "true", 4) == 0;
}

/* append JSON-escaped text */
static void buf_append_json(char **buf, size_t *n, size_t *cap, const char *s,
                            size_t sn) {
  for (size_t i = 0; i < sn; ++i) {
    if (*n + 8 >= *cap) *buf = realloc(*buf, *cap *= 2);
    unsigned char c = (unsigned char)s[i];
    if (c == '"' || c == '\\') { (*buf)[(*n)++] = '\\'; (*buf)[(*n)++] = (char)c; }
    else if (c == '\n') { (*buf)[(*n)++] = '\\'; (*buf)[(*n)++] = 'n'; }
    else if (c == '\r') { (*buf)[(*n)++] = '\\'; (*buf)[(*n)++] = 'r'; }
    else if (c == '\t') { (*buf)[(*n)++] = '\\'; (*buf)[(*n)++] = 't'; }
    else if (c < 0x20) { *n += (size_t)snprintf(*buf + *n, 8, "\\u%04x", c); }
    else (*buf)[(*n)++] = (char)c;
  }
}

/* Build a ChatML prompt from a messages array (or bare "prompt"). malloc'd. */
static char *build_prompt(const char *body) {
  size_t cap = 4096, n = 0;
  char *out = malloc(cap);
  out[0] = 0;
  const char *msgs = json_value(body, "messages");
  if (msgs && *msgs == '[') {
    const char *p = msgs;
    while ((p = strstr(p, "\"role\"")) != NULL) {
      const char *rv = json_value(p, "role");
      if (!rv || *rv != '"') break;
      const char *e = p + 6;
      char *role = json_string(rv, &e);
      if (!role || !e) { free(role); break; }
      const char *cv = json_value(p, "content");
      char *content = cv && *cv == '"' ? json_string(cv, NULL) : NULL;
      if (role && content) {
        size_t need = strlen(role) + strlen(content) + 64;
        while (n + need >= cap) out = realloc(out, cap *= 2);
        n += (size_t)snprintf(out + n, cap - n,
                              "<|im_start|>%s\n%s<|im_end|>\n", role, content);
      }
      free(role);
      free(content);
      p = e;
    }
    while (n + 32 >= cap) out = realloc(out, cap *= 2);
    n += (size_t)snprintf(out + n, cap - n, "<|im_start|>assistant\n");
    return out;
  }
  const char *pv = json_value(body, "prompt");
  if (pv && *pv == '"') {
    char *prompt = json_string(pv, NULL);
    if (prompt) { free(out); return prompt; }
  }
  free(out);
  return NULL;
}

static void send_all(int fd, const char *data, size_t n) {
  while (n > 0) {
    ssize_t w = write(fd, data, n);
    if (w <= 0) return;
    data += w;
    n -= (size_t)w;
  }
}

static void send_simple(int fd, int code, const char *ctype, const char *body) {
  char hdr[256];
  const char *reason = code == 200 ? "OK" :
                       code == 400 ? "Bad Request" :
                       code == 404 ? "Not Found" :
                       code == 429 ? "Too Many Requests" : "Error";
  int hn = snprintf(hdr, sizeof hdr,
                    "HTTP/1.1 %d %s\r\nContent-Type: %s\r\n"
                    "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                    code, reason, ctype,
                    strlen(body));
  send_all(fd, hdr, (size_t)hn);
  send_all(fd, body, strlen(body));
}

static void send_completion(int fd, const char *body, const char *worker_id) {
  char hdr[512];
  int hn = snprintf(hdr, sizeof hdr,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    "X-Oxidize-Engine: oxidize-c\r\n"
                    "X-Oxidize-Benchmark-Worker: %s\r\n"
                    "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                    worker_id, strlen(body));
  send_all(fd, hdr, (size_t)hn);
  send_all(fd, body, strlen(body));
}

static void send_health(int fd, const char *worker_id) {
  const char *body = "{\"status\":\"ok\"}";
  char hdr[512];
  int hn = snprintf(hdr, sizeof hdr,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    "X-Oxidize-Engine: oxidize-c\r\n"
                    "X-Oxidize-Benchmark-Worker: %s\r\n"
                    "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                    worker_id, strlen(body));
  send_all(fd, hdr, (size_t)hn);
  send_all(fd, body, strlen(body));
}

typedef struct {
  int fd;
  oc_tokenizer *tok;
  bool started;
} stream_ctx;

static void stream_token(uint32_t id, void *ud) {
  stream_ctx *ctx = ud;
  char frag[512], chunk[2048];
  size_t fn = oc_detokenize(ctx->tok, id, frag, sizeof frag);
  size_t ecap = fn * 6 + 64;
  char *esc = malloc(ecap);
  char *ep = esc;
  size_t en = 0, tmpcap = ecap;
  buf_append_json(&ep, &en, &tmpcap, frag, fn);
  esc[en] = 0;
  int cn = snprintf(chunk, sizeof chunk,
                    "data: {\"choices\":[{\"delta\":{\"content\":\"%s\"},"
                    "\"index\":0}]}\n\n",
                    esc);
  send_all(ctx->fd, chunk, (size_t)cn);
  free(ep);
}

static void on_sigint(int sig) { (void)sig; g_stop = 1; }

/* ---- WebSocket (RFC 6455): ws://HOST:PORT/v1/realtime ---- */

/* compact SHA-1 (public-domain style implementation) */
static void sha1(const uint8_t *data, size_t len, uint8_t out[20]) {
  uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
  uint64_t bitlen = (uint64_t)len * 8;
  size_t total = ((len + 8) / 64 + 1) * 64;
  uint8_t *msg = calloc(total, 1);
  memcpy(msg, data, len);
  msg[len] = 0x80;
  for (int i = 0; i < 8; ++i) msg[total - 1 - i] = (uint8_t)(bitlen >> (8 * i));
  for (size_t off = 0; off < total; off += 64) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t)msg[off + 4 * i] << 24 | (uint32_t)msg[off + 4 * i + 1] << 16 |
             (uint32_t)msg[off + 4 * i + 2] << 8 | msg[off + 4 * i + 3];
    for (int i = 16; i < 80; ++i) {
      uint32_t v = w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16];
      w[i] = (v << 1) | (v >> 31);
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; ++i) {
      uint32_t f, kc;
      if (i < 20) { f = (b & c) | ((~b) & d); kc = 0x5A827999; }
      else if (i < 40) { f = b ^ c ^ d; kc = 0x6ED9EBA1; }
      else if (i < 60) { f = (b & c) | (b & d) | (c & d); kc = 0x8F1BBCDC; }
      else { f = b ^ c ^ d; kc = 0xCA62C1D6; }
      uint32_t tmp = ((a << 5) | (a >> 27)) + f + e + kc + w[i];
      e = d; d = c; c = (b << 30) | (b >> 2); b = a; a = tmp;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
  }
  free(msg);
  for (int i = 0; i < 5; ++i) {
    out[4 * i] = (uint8_t)(h[i] >> 24);
    out[4 * i + 1] = (uint8_t)(h[i] >> 16);
    out[4 * i + 2] = (uint8_t)(h[i] >> 8);
    out[4 * i + 3] = (uint8_t)h[i];
  }
}

static void b64(const uint8_t *in, size_t n, char *out) {
  static const char T[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  size_t o = 0;
  for (size_t i = 0; i < n; i += 3) {
    uint32_t v = (uint32_t)in[i] << 16;
    if (i + 1 < n) v |= (uint32_t)in[i + 1] << 8;
    if (i + 2 < n) v |= in[i + 2];
    out[o++] = T[(v >> 18) & 63];
    out[o++] = T[(v >> 12) & 63];
    out[o++] = i + 1 < n ? T[(v >> 6) & 63] : '=';
    out[o++] = i + 2 < n ? T[v & 63] : '=';
  }
  out[o] = 0;
}

static void ws_send(int fd, uint8_t opcode, const char *data, size_t n) {
  uint8_t hdr[10];
  size_t hn = 0;
  hdr[hn++] = (uint8_t)(0x80 | opcode);
  if (n < 126) {
    hdr[hn++] = (uint8_t)n;
  } else if (n < 65536) {
    hdr[hn++] = 126;
    hdr[hn++] = (uint8_t)(n >> 8);
    hdr[hn++] = (uint8_t)n;
  } else {
    hdr[hn++] = 127;
    for (int i = 7; i >= 0; --i) hdr[hn++] = (uint8_t)((uint64_t)n >> (8 * i));
  }
  send_all(fd, (const char *)hdr, hn);
  send_all(fd, data, n);
}

static void ws_send_text(int fd, const char *s) { ws_send(fd, 1, s, strlen(s)); }

/* read exactly n bytes; false on EOF/error */
static bool read_n(int fd, uint8_t *buf, size_t n) {
  while (n > 0) {
    ssize_t r = read(fd, buf, n);
    if (r <= 0) return false;
    buf += r;
    n -= (size_t)r;
  }
  return true;
}

/* receive one complete (possibly fragmented) message; returns malloc'd payload
 * or NULL on close/error. *opcode_out = first frame opcode. */
static char *ws_recv(int fd, uint8_t *opcode_out, size_t *len_out) {
  size_t cap = 65536, n = 0;
  char *msg = malloc(cap);
  uint8_t first_op = 0;
  for (;;) {
    uint8_t h[2];
    if (!read_n(fd, h, 2)) { free(msg); return NULL; }
    bool fin = (h[0] & 0x80) != 0;
    uint8_t op = h[0] & 0x0F;
    bool masked = (h[1] & 0x80) != 0;
    uint64_t plen = h[1] & 0x7F;
    if (plen == 126) {
      uint8_t e[2];
      if (!read_n(fd, e, 2)) { free(msg); return NULL; }
      plen = ((uint64_t)e[0] << 8) | e[1];
    } else if (plen == 127) {
      uint8_t e[8];
      if (!read_n(fd, e, 8)) { free(msg); return NULL; }
      plen = 0;
      for (int i = 0; i < 8; ++i) plen = (plen << 8) | e[i];
    }
    if (plen > (32u << 20)) { free(msg); return NULL; }
    uint8_t mask[4] = {0};
    if (masked && !read_n(fd, mask, 4)) { free(msg); return NULL; }
    while (n + plen + 1 > cap) msg = realloc(msg, cap *= 2);
    if (!read_n(fd, (uint8_t *)msg + n, (size_t)plen)) { free(msg); return NULL; }
    if (masked)
      for (uint64_t i = 0; i < plen; ++i) msg[n + i] ^= (char)mask[i & 3];
    if (op == 8) { free(msg); *opcode_out = 8; return NULL; }   /* close */
    if (op == 9) { ws_send(fd, 10, msg + n, (size_t)plen); continue; } /* ping */
    if (op == 10) continue;                                     /* pong */
    if (op != 0) first_op = op;
    n += (size_t)plen;
    if (fin) break;
  }
  msg[n] = 0;
  *opcode_out = first_op;
  *len_out = n;
  return msg;
}

typedef struct { int fd; oc_tokenizer *tok; } ws_ctx;

static void ws_token(uint32_t id, void *ud) {
  ws_ctx *ctx = ud;
  char frag[512], frame[2048];
  size_t fn = oc_detokenize(ctx->tok, id, frag, sizeof frag);
  size_t ecap = fn * 6 + 64;
  char *esc = malloc(ecap);
  char *ep = esc;
  size_t en = 0, tmpcap = ecap;
  buf_append_json(&ep, &en, &tmpcap, frag, fn);
  esc[en] = 0;
  snprintf(frame, sizeof frame, "{\"type\":\"token\",\"content\":\"%s\"}", esc);
  ws_send_text(ctx->fd, frame);
  free(ep);
}

/* One WebSocket session: each client text message is a request (same JSON as
 * /v1/chat/completions, or plain text = single user message). Tokens stream
 * back as {"type":"token"} frames, then {"type":"done"}. */
static void ws_session(int fd, const char *req_headers, oc_model *m,
                       oc_tokenizer *tok, float temperature, size_t draft_k) {
  const char *kh = strcasestr(req_headers, "Sec-WebSocket-Key:");
  if (!kh) { close(fd); return; }
  kh += 18;
  while (*kh == ' ') ++kh;
  char key[64];
  size_t kn = 0;
  while (*kh && *kh != '\r' && *kh != '\n' && kn + 1 < sizeof key) key[kn++] = *kh++;
  key[kn] = 0;

  char cat[128];
  snprintf(cat, sizeof cat, "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11", key);
  uint8_t digest[20];
  sha1((const uint8_t *)cat, strlen(cat), digest);
  char accept_b64[32];
  b64(digest, 20, accept_b64);
  char resp[256];
  int rn = snprintf(resp, sizeof resp,
                    "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                    "Connection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n",
                    accept_b64);
  send_all(fd, resp, (size_t)rn);
  fprintf(stderr, "ws: client connected\n");

  for (;;) {
    uint8_t op = 0;
    size_t mlen = 0;
    char *msg = ws_recv(fd, &op, &mlen);
    if (!msg) break;

    char *prompt = build_prompt(msg);
    if (!prompt) {
      /* plain text message -> single user turn */
      size_t need = mlen + 128;
      prompt = malloc(need);
      snprintf(prompt, need,
               "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", msg);
    }
    size_t max_new = 512;
    if (!json_size_t_bounded(msg, "max_tokens", 512, 1, m->kv_ctx, &max_new)) {
      ws_send_text(fd, "{\"type\":\"error\",\"message\":\"bad max_tokens\"}");
      free(msg);
      continue;
    }
    float temp = (float)json_number(msg, "temperature", (double)temperature);
    free(msg);
    if (max_new > m->kv_ctx - 64) max_new = m->kv_ctx - 64;

    size_t n_prompt;
    uint32_t *ids = oc_tokenize(tok, prompt, false, &n_prompt);
    free(prompt);
    if (n_prompt == 0 || n_prompt + 8 >= m->kv_ctx) {
      ws_send_text(fd, "{\"type\":\"error\",\"message\":\"prompt empty or too long\"}");
      free(ids);
      continue;
    }

    ws_ctx wctx = {fd, tok};
    oc_gen g = {0};
    g.m = m;
    g.tok = tok;
    g.temperature = temp;
    g.top_k = 40;
    g.top_p = 0.95f;
    g.draft_k = draft_k;
    g.spec_mode = g_spec_mode;
    g.on_token = ws_token;
    g.ud = &wctx;

    uint32_t *outids = malloc(max_new * sizeof(uint32_t));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    size_t n_out = oc_generate(&g, ids, n_prompt, max_new, outids);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) +
                  (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    char done[256];
    snprintf(done, sizeof done,
             "{\"type\":\"done\",\"prompt_tokens\":%zu,\"completion_tokens\":%zu,"
             "\"tok_per_s\":%.2f}",
             n_prompt, n_out, secs > 0 ? (double)n_out / secs : 0.0);
    ws_send_text(fd, done);
    fprintf(stderr, "ws req: %zu + %zu tokens, %.2f tok/s%s", n_prompt, n_out,
            secs > 0 ? (double)n_out / secs : 0.0, "");
    if (g.drafted)
      fprintf(stderr, "  [mtp accept %.0f%%]",
              100.0 * (double)g.accepted / (double)g.drafted);
    fprintf(stderr, "\n");
    free(outids);
    free(ids);
  }
  fprintf(stderr, "ws: client disconnected\n");
  close(fd);
}

#ifdef OC_CUDA
static batch_request *batch_request_for_id(batch_request *const *requests, size_t count,
                                           uint64_t request_id) {
  for (size_t i = 0; i < count; ++i)
    if (requests[i] && requests[i]->request_id == request_id) return requests[i];
  return NULL;
}

static void batch_fail_slot(batch_runtime *runtime, batch_request **requests,
                            size_t slot, size_t *active) {
  batch_request *request = requests[slot];
  if (!request) return;
  oc_batch_scheduler_cancel(runtime->scheduler, slot);
  oc_batch_scheduler_reclaim(runtime->scheduler, slot);
  requests[slot] = NULL;
  --*active;
  batch_request_finish(request, true);
  batch_release_admission(runtime);
}

static void *batch_worker(void *opaque) {
  batch_runtime *runtime = opaque;
  batch_request **requests = calloc(runtime->capacity, sizeof(*requests));
  oc_batch_lane lanes[64];
  oc_cuda_decode_lane cuda_lanes[64];
  uint32_t next_ids[64];
  size_t active = 0;
  if (!requests) return NULL;
  while (!g_stop) {
    while (active < runtime->capacity) {
      batch_request *request = batch_take(runtime);
      if (!request) break;
      oc_sequence_request sequence = {
          .request_id = request->request_id,
          .prompt_tokens = request->prompt_tokens,
          .prompt_count = request->prompt_count,
          .max_tokens = request->max_tokens,
          .rng_seed = request->request_id,
      };
      size_t slot = SIZE_MAX;
      if (!oc_batch_scheduler_admit_sequence(runtime->scheduler, &sequence, &slot) ||
          slot == SIZE_MAX || slot >= runtime->capacity) {
        batch_request_finish(request, true);
        batch_release_admission(runtime);
        continue;
      }
      requests[slot] = request;
      ++active;
    }
    if (!active) {
      batch_wait_for_work(runtime);
      continue;
    }
    size_t lane_count = 0;
    if (!oc_batch_scheduler_next_lanes(runtime->scheduler, lanes, runtime->capacity,
                                       &lane_count) || !lane_count) continue;
    for (size_t i = 0; i < lane_count; ++i) {
      cuda_lanes[i] = (oc_cuda_decode_lane){
          .token = lanes[i].token,
          .position = (uint32_t)lanes[i].position,
          .slot = (uint32_t)lanes[i].slot,
          .want_token = 1,
      };
    }
    if (oc_cuda_decode_batch(runtime->model, cuda_lanes, lane_count, next_ids) != 0) {
      for (size_t i = 0; i < lane_count; ++i)
        batch_fail_slot(runtime, requests, lanes[i].slot, &active);
      continue;
    }
    bool terminal[64] = {0};
    bool failed[64] = {0};
    for (size_t i = 0; i < lane_count; ++i) {
      batch_request *request = requests[lanes[i].slot];
      if (!request || next_ids[i] == UINT32_MAX) { failed[i] = true; continue; }
      if (!lanes[i].produces_output) {
        if (!oc_batch_scheduler_finish_lane(runtime->scheduler, &lanes[i])) failed[i] = true;
        continue;
      }
      bool eog = oc_is_eog(runtime->tokenizer, next_ids[i]);
      if (!oc_batch_scheduler_record_output(runtime->scheduler, &lanes[i], next_ids[i], eog)) {
        failed[i] = true;
        continue;
      }
      if (!eog) ++request->generated;
      terminal[i] = eog || request->generated == request->max_tokens;
    }
    uint64_t request_id = 0;
    uint32_t token = 0;
    while (oc_batch_scheduler_output_next(runtime->scheduler, &request_id, &token)) {
      batch_request *request = batch_request_for_id(requests, runtime->capacity, request_id);
      if (!request || !batch_request_append(request, token)) {
        for (size_t i = 0; i < lane_count; ++i)
          if (lanes[i].request_id == request_id) failed[i] = true;
      }
    }
    for (size_t i = 0; i < lane_count; ++i) {
      if (!terminal[i] && !failed[i]) continue;
      size_t slot = lanes[i].slot;
      batch_request *request = requests[slot];
      if (!request) continue;
      if (failed[i]) {
        batch_fail_slot(runtime, requests, slot, &active);
        continue;
      }
      if (!oc_batch_scheduler_reclaim(runtime->scheduler, slot)) {
        batch_fail_slot(runtime, requests, slot, &active);
        continue;
      }
      requests[slot] = NULL;
      --active;
      batch_request_finish(request, false);
      batch_release_admission(runtime);
    }
  }
  free(requests);
  return NULL;
}

static batch_runtime *batch_runtime_start(oc_model *model, oc_tokenizer *tokenizer,
                                          size_t capacity, size_t queue_capacity,
                                          int device_id) {
  if (!model || !model->gemma || model->cfg.kv_int8 || !capacity || capacity > 64 ||
      !model->kv_ctx) return NULL;
  const oc_batch_config config = {
      .capacity = capacity, .queue_capacity = capacity,
      .gpu_ids = &device_id, .gpu_count = 1,
  };
  batch_runtime *runtime = calloc(1, sizeof(*runtime));
  if (!runtime || pthread_mutex_init(&runtime->mutex, NULL) ||
      pthread_cond_init(&runtime->work, NULL)) {
    free(runtime);
    return NULL;
  }
  runtime->scheduler = oc_batch_scheduler_new(&config);
  runtime->model = model;
  runtime->tokenizer = tokenizer;
  runtime->capacity = capacity;
  runtime->queue_capacity = queue_capacity;
  if (!runtime->scheduler) goto fail;
  oc_cuda_release(model);
  if (oc_cuda_configure_batch(model, device_id, capacity, model->kv_ctx) ||
      oc_cuda_build(model) || pthread_create(&runtime->worker, NULL, batch_worker, runtime))
    goto fail;
  pthread_detach(runtime->worker);
  return runtime;

fail:
  oc_batch_scheduler_free(runtime->scheduler);
  pthread_cond_destroy(&runtime->work);
  pthread_mutex_destroy(&runtime->mutex);
  free(runtime);
  return NULL;
}
#endif

static void serve_batch_completion(int fd, serve_context *server, const char *body,
                                   bool completion_endpoint) {
#ifdef OC_CUDA
  char *prompt = build_prompt(body);
  if (!prompt) {
    send_simple(fd, 400, "application/json", "{\"error\":\"missing messages/prompt\"}");
    return;
  }
  size_t max_tokens = 0;
  bool stream = json_bool(body, "stream", false);
  double temperature = json_number(body, "temperature", 0.0);
  if (stream || temperature != 0.0 ||
      !json_size_t_bounded(body, "max_tokens", 512, 1, server->model->kv_ctx, &max_tokens)) {
    free(prompt);
    send_simple(fd, 400, "application/json", "{\"error\":\"batch requires stream=false, temperature=0, and valid max_tokens\"}");
    return;
  }
  size_t prompt_count = 0;
  uint32_t *prompt_tokens = oc_tokenize(server->tokenizer, prompt, false, &prompt_count);
  free(prompt);
  if (!prompt_tokens || !prompt_count || prompt_count + max_tokens > server->model->kv_ctx) {
    free(prompt_tokens);
    send_simple(fd, 400, "application/json", "{\"error\":\"prompt plus completion exceeds context\"}");
    return;
  }
  batch_request *request = calloc(1, sizeof(*request));
  if (!request || pthread_mutex_init(&request->mutex, NULL) ||
      pthread_cond_init(&request->completed, NULL)) {
    free(prompt_tokens);
    free(request);
    send_simple(fd, 500, "application/json", "{\"error\":\"allocation failed\"}");
    return;
  }
  request->prompt_tokens = prompt_tokens;
  request->prompt_count = prompt_count;
  request->max_tokens = max_tokens;
  if (!batch_enqueue(server->batch, request)) {
    batch_request_free(request);
    send_simple(fd, 429, "application/json", "{\"error\":\"server at capacity\"}");
    return;
  }
  if (!batch_request_wait(request)) {
    batch_request_free(request);
    send_simple(fd, 500, "application/json", "{\"error\":\"batched decode failed\"}");
    return;
  }
  size_t capacity = request->output_count * 6 + 512;
  char *text = calloc(capacity, 1);
  size_t text_size = 0;
  char fragment[512];
  for (size_t i = 0; text && i < request->output_count; ++i) {
    size_t size = oc_detokenize(server->tokenizer, request->output_tokens[i], fragment,
                                sizeof(fragment));
    buf_append_json(&text, &text_size, &capacity, fragment, size);
  }
  if (!text) {
    batch_request_free(request);
    send_simple(fd, 500, "application/json", "{\"error\":\"allocation failed\"}");
    return;
  }
  text[text_size] = 0;
  size_t response_capacity = text_size + 512;
  char *response = malloc(response_capacity);
  if (response) {
    if (completion_endpoint) {
      snprintf(response, response_capacity,
               "{\"id\":\"cmpl-oc\",\"object\":\"text_completion\","
               "\"choices\":[{\"index\":0,\"text\":\"%s\",\"finish_reason\":\"stop\"}],"
               "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%zu}}",
               text, request->prompt_count, request->output_count);
    } else {
      snprintf(response, response_capacity,
               "{\"id\":\"chatcmpl-oc\",\"object\":\"chat.completion\","
               "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
               "\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
               "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%zu}}",
               text, request->prompt_count, request->output_count);
    }
    send_completion(fd, response, server->worker_id);
  } else {
    send_simple(fd, 500, "application/json", "{\"error\":\"allocation failed\"}");
  }
  free(response);
  free(text);
  batch_request_free(request);
#else
  (void)server;
  (void)body;
  (void)completion_endpoint;
  send_simple(fd, 400, "application/json", "{\"error\":\"batch CUDA support unavailable\"}");
#endif
}

static void *serve_client(void *arg) {
  client_context *client = arg;
  int fd = client->fd;
  serve_context *server = client->server;
  oc_model *m = server->model;
  oc_tokenizer *tok = server->tokenizer;
  free(client);

  size_t req_cap = 8 << 20;
  char *req = malloc(req_cap);
  char *prompt = NULL;
  uint32_t *ids = NULL;
  uint32_t *outids = NULL;
  bool admitted = false;
  if (!req) goto done;

  size_t n = 0;
  char *body = NULL;
  size_t content_len = 0;
  while (n + 1 < req_cap) {
    ssize_t r = read(fd, req + n, req_cap - n - 1);
    if (r <= 0) break;
    n += (size_t)r;
    req[n] = 0;
    if (!body) {
      char *sep = strstr(req, "\r\n\r\n");
      if (sep) {
        body = sep + 4;
        char *cl = strcasestr(req, "Content-Length:");
        content_len = cl ? strtoul(cl + 15, NULL, 10) : 0;
      }
    }
    if (body && n - (size_t)(body - req) >= content_len) break;
  }
  if (!body) goto done;

  if (strncmp(req, "GET /v1/realtime", 16) == 0 &&
      strcasestr(req, "Upgrade: websocket")) {
    if (server->batch) {
      send_simple(fd, 400, "application/json", "{\"error\":\"websocket unavailable in batch mode\"}");
      goto done;
    }
    if (!admission_acquire(&server->gate)) {
      send_simple(fd, 429, "application/json",
                  "{\"error\":\"server at capacity\"}");
      goto done;
    }
    admitted = true;
    ws_session(fd, req, m, tok, server->temperature, server->draft_k);
    fd = -1;
    goto done;
  }
  if (strncmp(req, "GET /health", 11) == 0) {
    send_health(fd, server->worker_id);
    goto done;
  }
  if (strncmp(req, "POST /v1/chat/completions", 25) != 0 &&
      strncmp(req, "POST /v1/completions", 20) != 0) {
    send_simple(fd, 404, "application/json", "{\"error\":\"not found\"}");
    goto done;
  }

  if (server->batch) {
    serve_batch_completion(fd, server, body,
                           strncmp(req, "POST /v1/completions", 20) == 0);
    goto done;
  }

  prompt = build_prompt(body);
  if (!prompt) {
    send_simple(fd, 400, "application/json", "{\"error\":\"missing messages/prompt\"}");
    goto done;
  }
  size_t max_new = 512;
  if (!json_size_t_bounded(body, "max_tokens", 512, 1, m->kv_ctx, &max_new)) {
    send_simple(fd, 400, "application/json", "{\"error\":\"bad max_tokens\"}");
    goto done;
  }
  float temp = (float)json_number(body, "temperature", (double)server->temperature);
  bool stream = json_bool(body, "stream", false);
  if (max_new > m->kv_ctx - 64) max_new = m->kv_ctx - 64;

  size_t n_prompt;
  ids = oc_tokenize(tok, prompt, false, &n_prompt);
  free(prompt);
  prompt = NULL;
  if (n_prompt == 0 || n_prompt + 8 >= m->kv_ctx) {
    send_simple(fd, 400, "application/json", "{\"error\":\"prompt empty or too long\"}");
    goto done;
  }
  if (!admission_acquire(&server->gate)) {
    send_simple(fd, 429, "application/json", "{\"error\":\"server at capacity\"}");
    goto done;
  }
  admitted = true;

  oc_gen g = {0};
  g.m = m;
  g.tok = tok;
  g.temperature = temp;
  g.top_k = 40;
  g.top_p = 0.95f;
  g.draft_k = server->draft_k;
  g.spec_mode = g_spec_mode;
  stream_ctx sctx = {fd, tok, false};
  if (stream) {
    const char *hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                      "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
    send_all(fd, hdr, strlen(hdr));
    g.on_token = stream_token;
    g.ud = &sctx;
  }

  outids = malloc(max_new * sizeof(uint32_t));
  if (!outids) goto done;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  size_t n_out = oc_generate(&g, ids, n_prompt, max_new, outids);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
  fprintf(stderr, "req: %zu prompt + %zu gen tokens in %.2fs (%.2f tok/s)%s",
          n_prompt, n_out, secs, secs > 0 ? (double)n_out / secs : 0.0, "");
  if (g.drafted)
    fprintf(stderr, "  [mtp accept %.0f%%]", 100.0 * (double)g.accepted / (double)g.drafted);
  fprintf(stderr, "\n");

  if (stream) {
    send_all(fd, "data: [DONE]\n\n", 14);
  } else {
    size_t cap = 16384, bn = 0;
    char *text = malloc(cap);
    char frag[512];
    for (size_t i = 0; i < n_out; ++i) {
      size_t fn = oc_detokenize(tok, outids[i], frag, sizeof frag);
      buf_append_json(&text, &bn, &cap, frag, fn);
    }
    text[bn] = 0;
    size_t rcap = bn + 512;
    char *resp = malloc(rcap);
    snprintf(resp, rcap,
        "{\"id\":\"chatcmpl-oc\",\"object\":\"chat.completion\","
        "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
        "\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%zu}}",
        text, n_prompt, n_out);
    send_completion(fd, resp, server->worker_id);
    free(resp);
    free(text);
  }

done:
  if (admitted) admission_release(&server->gate);
  free(outids);
  free(ids);
  free(prompt);
  free(req);
  if (fd >= 0) close(fd);
  return NULL;
}

int oc_serve(oc_model *m, oc_tokenizer *tok, const char *host, int port,
             float temperature, size_t draft_k, int spec_mode,
             size_t max_seqs, size_t max_queue, const char *worker_id, int device_id) {
  g_spec_mode = spec_mode;
  signal(SIGPIPE, SIG_IGN);
  signal(SIGINT, on_sigint);
  serve_context server = {
    .model = m,
    .tokenizer = tok,
    .temperature = temperature,
    .draft_k = draft_k,
    .worker_id = worker_id,
    .gate = {
      .mutex = PTHREAD_MUTEX_INITIALIZER,
      .ready = PTHREAD_COND_INITIALIZER,
      .max_seqs = max_seqs,
      .max_queue = max_queue,
    },
  };
  if (max_seqs > 1) {
#ifdef OC_CUDA
    server.batch = batch_runtime_start(m, tok, max_seqs, max_queue, device_id);
    if (!server.batch)
      oc_die("serve: batched mode requires a CUDA Gemma model with available slot KV memory");
#else
    (void)device_id;
    oc_die("serve: --max-seqs > 1 requires an OC_CUDA build");
#endif
  }
  int lfd = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in addr = {0};
  addr.sin_family = AF_INET;
  if (inet_pton(AF_INET, host, &addr.sin_addr) != 1)
    oc_die("serve: bad --host %s", host);
  addr.sin_port = htons((uint16_t)port);
  if (bind(lfd, (struct sockaddr *)&addr, sizeof addr) != 0)
    oc_die("serve: bind %s:%d failed (%s)", host, port, strerror(errno));
  listen(lfd, 16);
  fprintf(stderr,
          "oxidize-c: serving on http://%s:%d\n"
          "  POST /v1/chat/completions  (OpenAI-compatible, stream=SSE)\n"
          "  ws://%s:%d/v1/realtime  (WebSocket: send text or JSON, "
          "token frames back)\n"
          "  %s admission: max-seqs=%zu max-queue=%zu worker=%s\n",
          host, port, host, port, server.batch ? "batched CUDA" : "scalar",
          max_seqs, max_queue, worker_id);

  while (!g_stop) {
    int fd = accept(lfd, NULL, NULL);
    if (fd < 0) continue;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    client_context *client = malloc(sizeof(*client));
    if (!client) {
      close(fd);
      continue;
    }
    *client = (client_context){.fd = fd, .server = &server};
    pthread_t thread;
    if (pthread_create(&thread, NULL, serve_client, client) != 0) {
      close(fd);
      free(client);
      continue;
    }
    pthread_detach(thread);
  }
  close(lfd);
  return 0;
}
