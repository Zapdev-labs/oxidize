/* Pipeline parallelism over TCP. See distributed.h for the protocol and the
 * honest scope (capacity not speed; no discovery/crypto/tensor-parallel).
 *
 * Bottom-up: little-endian header pack -> full-buffer socket I/O -> framing ->
 * the family-dispatched slice runner -> head/worker step loops -> TCP bringup. */
#include "distributed.h"

#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "model_gemma4.h"
#include "model_llama.h"

/* ---- little-endian header codec (fixed 20-byte frame header) --------------- */
static void put_u32le(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void put_u64le(uint8_t* p, uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static uint32_t get_u32le(const uint8_t* p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}
static uint64_t get_u64le(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i);
  return v;
}

/* ---- full-buffer socket I/O ------------------------------------------------
 * TCP is a byte stream: one write may be split across many, and one read may
 * return a partial frame. These loop until the whole buffer moves, retrying on
 * EINTR. MSG_NOSIGNAL turns a write to a closed peer into EPIPE instead of a
 * process-killing SIGPIPE, so no global signal handler is needed. */
static int write_full(int fd, const void* buf, size_t n, char* err, size_t errlen) {
  const uint8_t* p = (const uint8_t*)buf;
  size_t off = 0;
  while (off < n) {
    ssize_t w = send(fd, p + off, n - off, MSG_NOSIGNAL);
    if (w < 0) {
      if (errno == EINTR) continue;
      snprintf(err, errlen, "socket write: %s", strerror(errno));
      return -1;
    }
    off += (size_t)w;
  }
  return 0;
}
static int read_full(int fd, void* buf, size_t n, char* err, size_t errlen) {
  uint8_t* p = (uint8_t*)buf;
  size_t off = 0;
  while (off < n) {
    ssize_t r = recv(fd, p + off, n - off, 0);
    if (r < 0) {
      if (errno == EINTR) continue;
      snprintf(err, errlen, "socket read: %s", strerror(errno));
      return -1;
    }
    if (r == 0) {
      snprintf(err, errlen, "socket read: peer closed mid-frame");
      return -1;
    }
    off += (size_t)r;
  }
  return 0;
}

/* ---- framing --------------------------------------------------------------- */
int oc_dist_send(int fd, uint32_t kind, uint64_t pos, const float* v, uint32_t n,
                 char* err, size_t errlen) {
  uint8_t hdr[20];
  put_u32le(hdr, OC_DIST_MAGIC);
  put_u32le(hdr + 4, kind);
  put_u64le(hdr + 8, pos);
  put_u32le(hdr + 16, n);
  if (write_full(fd, hdr, sizeof hdr, err, errlen) != 0) return -1;
  if (n && write_full(fd, v, (size_t)n * sizeof(float), err, errlen) != 0) return -1;
  return 0;
}
int oc_dist_recv(int fd, uint32_t* kind, uint64_t* pos, float* v, uint32_t cap,
                 uint32_t* n, char* err, size_t errlen) {
  uint8_t hdr[20];
  if (read_full(fd, hdr, sizeof hdr, err, errlen) != 0) return -1;
  if (get_u32le(hdr) != OC_DIST_MAGIC) {
    snprintf(err, errlen, "bad frame magic (stream desync)");
    return -1;
  }
  *kind = get_u32le(hdr + 4);
  *pos = get_u64le(hdr + 8);
  uint32_t nn = get_u32le(hdr + 16);
  if (nn > cap) {
    snprintf(err, errlen, "frame floats %u exceed capacity %u", nn, cap);
    return -1;
  }
  *n = nn;
  if (nn && read_full(fd, v, (size_t)nn * sizeof(float), err, errlen) != 0) return -1;
  return 0;
}

/* ---- family dispatch: only llama and gemma4 expose a layer-range hook ------
 * The model structs are public, so the residual buffer x and the layer count
 * n_layers are addressable here. Narrowing n_layers bounds the *_forward_from
 * tail to [l0, l1); it is restored immediately after the call. */
static size_t* n_layers_ptr(Model* m) {
  if (m->family == MODEL_LLAMA) return &((LlamaModel*)m->handle)->n_layers;
  if (m->family == MODEL_GEMMA4) return &((Gemma4Model*)m->handle)->n_layers;
  return NULL; /* qwen36 / deepseek: no forward_from hook */
}
static float* x_ptr(Model* m) {
  if (m->family == MODEL_LLAMA) return ((LlamaModel*)m->handle)->x;
  if (m->family == MODEL_GEMMA4) return ((Gemma4Model*)m->handle)->x;
  return NULL;
}
static size_t model_hidden(Model* m) {
  if (m->family == MODEL_LLAMA) return ((LlamaModel*)m->handle)->hidden;
  if (m->family == MODEL_GEMMA4) return ((Gemma4Model*)m->handle)->hidden;
  return 0;
}
static float* forward_from(Model* m, size_t pos, size_t l0, bool need_logits) {
  if (m->family == MODEL_LLAMA)
    return llama_forward_from((LlamaModel*)m->handle, pos, l0, need_logits);
  if (m->family == MODEL_GEMMA4)
    return gemma4_forward_from((Gemma4Model*)m->handle, pos, l0, need_logits);
  return NULL;
}

/* Run this rank's layer slice for one token at `pos`.
 *   head (rank 0): embed `token`, then layers [0, l1); x_in ignored.
 *   others: load x_in (hidden floats) into the residual stream, layers [l0, l1).
 * The last rank (l1 == n_layers) also runs the final norm + output head.
 * Returns the residual stream (hidden floats) on a non-last rank, or the logits
 * (vocab floats) on the last rank — both point into the model's own buffers.
 * NULL + err on an unsupported arch or a forward failure (e.g. pos >= ctx). */
static float* run_slice(OcPipe* p, int32_t token, const float* x_in, size_t pos,
                        char* err, size_t errlen) {
  Model* m = p->model;
  size_t* nlp = n_layers_ptr(m);
  float* x = x_ptr(m);
  if (!nlp || !x) {
    snprintf(err, errlen, "arch cannot pipeline-split (no *_forward_from hook)");
    return NULL;
  }
  bool is_head = p->rank == 0;
  bool is_last = p->rank == p->nranks - 1;
  bool need_logits = is_last;

  size_t saved = *nlp;
  *nlp = p->l1; /* bound the tail hook to end at l1 (l1 == saved on the last rank) */
  float* out;
  if (is_head) {
    out = m->forward(m->handle, token, pos, need_logits); /* embed + [0, l1) */
  } else {
    if (!x_in) {
      snprintf(err, errlen, "worker slice needs a hidden-state buffer");
      *nlp = saved;
      return NULL;
    }
    memcpy(x, x_in, p->hidden * sizeof(float));
    out = forward_from(m, pos, p->l0, need_logits); /* [l0, l1) */
  }
  *nlp = saved;

  if (need_logits) {
    if (!out) {
      snprintf(err, errlen, "forward failed at pos %zu (context full?)", pos);
      return NULL;
    }
    return out; /* logits */
  }
  return x; /* residual stream to hand to the next rank */
}

/* ---- head / worker step loops ---------------------------------------------- */
int oc_pipe_head_step(OcPipe* p, int32_t token, size_t pos, float* logits_out,
                      char* err, size_t errlen) {
  float* x = run_slice(p, token, NULL, pos, err, errlen);
  if (!x) return -1;
  if (oc_dist_send(p->next_fd, OC_DIST_HIDDEN, pos, x, (uint32_t)p->hidden, err,
                   errlen) != 0)
    return -1;
  uint32_t kind, n;
  uint64_t rpos;
  if (oc_dist_recv(p->next_fd, &kind, &rpos, logits_out, (uint32_t)p->vocab, &n,
                   err, errlen) != 0)
    return -1;
  if (kind != OC_DIST_LOGITS || n != p->vocab) {
    snprintf(err, errlen, "head: expected a logits frame (got kind %u, n %u)", kind, n);
    return -1;
  }
  return 0;
}

int oc_pipe_head_shutdown(OcPipe* p, char* err, size_t errlen) {
  return oc_dist_send(p->next_fd, OC_DIST_SHUTDOWN, 0, NULL, 0, err, errlen);
}

int oc_pipe_serve(OcPipe* p, char* err, size_t errlen) {
  bool is_last = p->rank == p->nranks - 1;
  uint32_t cap = (uint32_t)(p->vocab > p->hidden ? p->vocab : p->hidden);
  float* buf = malloc((size_t)cap * sizeof(float));
  if (!buf) {
    snprintf(err, errlen, "out of memory (%u floats)", cap);
    return -1;
  }
  int rc = 0;
  for (;;) {
    uint32_t kind, n;
    uint64_t pos;
    if (oc_dist_recv(p->prev_fd, &kind, &pos, buf, cap, &n, err, errlen) != 0) {
      rc = -1;
      break;
    }
    if (kind == OC_DIST_SHUTDOWN) {
      /* Relay the shutdown downstream so the whole tail drains, then return. */
      if (!is_last)
        rc = oc_dist_send(p->next_fd, OC_DIST_SHUTDOWN, 0, NULL, 0, err, errlen);
      break;
    }
    if (kind != OC_DIST_HIDDEN || n != p->hidden) {
      snprintf(err, errlen, "worker: expected a hidden frame (got kind %u, n %u)", kind, n);
      rc = -1;
      break;
    }
    float* out = run_slice(p, 0, buf, pos, err, errlen);
    if (!out) {
      rc = -1;
      break;
    }
    if (is_last) {
      if (oc_dist_send(p->prev_fd, OC_DIST_LOGITS, pos, out, (uint32_t)p->vocab,
                       err, errlen) != 0) {
        rc = -1;
        break;
      }
    } else {
      /* Push the hidden state up, then relay the returning logits back down.
       * `out` is the model's own x buffer; recv lands in `buf`, a different
       * buffer, so the send above is fully consumed before it is overwritten. */
      if (oc_dist_send(p->next_fd, OC_DIST_HIDDEN, pos, out, (uint32_t)p->hidden,
                       err, errlen) != 0) {
        rc = -1;
        break;
      }
      uint32_t k2, n2;
      uint64_t p2;
      if (oc_dist_recv(p->next_fd, &k2, &p2, buf, cap, &n2, err, errlen) != 0) {
        rc = -1;
        break;
      }
      if (k2 != OC_DIST_LOGITS) {
        snprintf(err, errlen, "worker: expected logits from next rank (got kind %u)", k2);
        rc = -1;
        break;
      }
      if (oc_dist_send(p->prev_fd, OC_DIST_LOGITS, p2, buf, n2, err, errlen) != 0) {
        rc = -1;
        break;
      }
    }
  }
  free(buf);
  return rc;
}

/* ---- TCP bringup ----------------------------------------------------------- */
/* Split "host:port" at the last ':'. host may be empty (used only for connect,
 * not for the passive bind, which listens on all interfaces). */
static int split_hostport(const char* s, char* host, size_t hostn, char* port,
                          size_t portn, char* err, size_t errlen) {
  const char* colon = strrchr(s, ':');
  if (!colon || colon[1] == '\0') {
    snprintf(err, errlen, "peer \"%s\" is not host:port", s);
    return -1;
  }
  size_t hl = (size_t)(colon - s);
  if (hl >= hostn || strlen(colon + 1) >= portn) {
    snprintf(err, errlen, "peer \"%s\" too long", s);
    return -1;
  }
  memcpy(host, s, hl);
  host[hl] = '\0';
  strcpy(port, colon + 1);
  return 0;
}

static void set_sockopts(int fd) {
  int one = 1;
  /* Frames are tiny and latency-sensitive: don't let Nagle coalesce them. */
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
}

/* Bind + listen on `port` (all interfaces). Returns the listening fd or -1. */
static int listen_on(const char* port, char* err, size_t errlen) {
  struct addrinfo hints = {0}, *res = NULL;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  int gai = getaddrinfo(NULL, port, &hints, &res);
  if (gai != 0) {
    snprintf(err, errlen, "listen port %s: %s", port, gai_strerror(gai));
    return -1;
  }
  int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (fd < 0) {
    snprintf(err, errlen, "socket: %s", strerror(errno));
    freeaddrinfo(res);
    return -1;
  }
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  if (bind(fd, res->ai_addr, res->ai_addrlen) != 0 || listen(fd, 1) != 0) {
    snprintf(err, errlen, "bind/listen port %s: %s", port, strerror(errno));
    close(fd);
    freeaddrinfo(res);
    return -1;
  }
  freeaddrinfo(res);
  return fd;
}

/* Connect to host:port, retrying for ~30s so the next rank can start after us. */
static int connect_to(const char* hostport, char* err, size_t errlen) {
  char host[256], port[32];
  if (split_hostport(hostport, host, sizeof host, port, sizeof port, err, errlen) != 0)
    return -1;
  struct addrinfo hints = {0}, *res = NULL;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  int gai = getaddrinfo(host[0] ? host : "127.0.0.1", port, &hints, &res);
  if (gai != 0) {
    snprintf(err, errlen, "resolve %s: %s", hostport, gai_strerror(gai));
    return -1;
  }
  int fd = -1;
  for (int attempt = 0; attempt < 300; ++attempt) {
    fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) {
      snprintf(err, errlen, "socket: %s", strerror(errno));
      break;
    }
    if (connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
      set_sockopts(fd);
      freeaddrinfo(res);
      return fd;
    }
    close(fd);
    fd = -1;
    nanosleep(&(struct timespec){.tv_sec = 0, .tv_nsec = 100000000L}, NULL); /* 100ms */
  }
  if (fd < 0 && err[0] == '\0')
    snprintf(err, errlen, "connect %s: %s", hostport, strerror(errno));
  freeaddrinfo(res);
  return -1;
}

int oc_pipe_init(OcPipe* p, int rank, int nranks, const char* const* peers,
                 size_t n_peers, Model* model, char* err, size_t errlen) {
  memset(p, 0, sizeof *p);
  p->rank = rank;
  p->nranks = nranks;
  p->prev_fd = p->next_fd = -1;
  p->model = model;
  p->hidden = model_hidden(model);
  p->vocab = model->vocab;

  size_t* nlp = n_layers_ptr(model);
  if (!nlp || p->hidden == 0) {
    snprintf(err, errlen,
             "this model's arch cannot pipeline-split (needs a *_forward_from "
             "hook: llama- or gemma4-family only)");
    return -1;
  }
  p->n_layers = *nlp;
  if (nranks < 1 || rank < 0 || rank >= nranks) {
    snprintf(err, errlen, "bad --shard %d/%d", rank, nranks);
    return -1;
  }
  if ((size_t)nranks > p->n_layers) {
    snprintf(err, errlen, "cannot split %zu layers across %d ranks", p->n_layers, nranks);
    return -1;
  }
  if (n_peers < (size_t)(nranks - 1)) {
    snprintf(err, errlen, "need %d --peer host:port entries (rank 1..%d), got %zu",
             nranks - 1, nranks - 1, n_peers);
    return -1;
  }
  /* Even layer split: rank r runs [r*N/nranks, (r+1)*N/nranks). */
  p->l0 = (size_t)rank * p->n_layers / (size_t)nranks;
  p->l1 = (size_t)(rank + 1) * p->n_layers / (size_t)nranks;

  /* Order matters for deadlock freedom: listen first, then connect to the next
   * rank (whose listener the kernel already backlogs), then accept from prev. */
  int listen_fd = -1;
  if (rank > 0) {
    char host[256], port[32];
    if (split_hostport(peers[rank - 1], host, sizeof host, port, sizeof port, err,
                       errlen) != 0)
      return -1;
    listen_fd = listen_on(port, err, errlen);
    if (listen_fd < 0) return -1;
  }
  if (rank < nranks - 1) {
    p->next_fd = connect_to(peers[rank], err, errlen);
    if (p->next_fd < 0) {
      if (listen_fd >= 0) close(listen_fd);
      return -1;
    }
  }
  if (rank > 0) {
    p->prev_fd = accept(listen_fd, NULL, NULL);
    close(listen_fd);
    if (p->prev_fd < 0) {
      snprintf(err, errlen, "accept: %s", strerror(errno));
      oc_pipe_close(p);
      return -1;
    }
    set_sockopts(p->prev_fd);
  }
  return 0;
}

void oc_pipe_close(OcPipe* p) {
  if (p->prev_fd >= 0) close(p->prev_fd);
  if (p->next_fd >= 0) close(p->next_fd);
  p->prev_fd = p->next_fd = -1;
}
