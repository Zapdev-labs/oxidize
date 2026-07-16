/* Pipeline parallelism over plain TCP — run a model too big for one machine
 * across N processes. Rank r holds transformer layers [l0, l1); it runs its
 * slice on CPU and passes the residual-stream hidden vector to rank r+1 over a
 * socket. The last rank computes logits and sends them back down the chain to
 * the head (rank 0), which owns the prompt, sampling and streaming.
 *
 * This is NOT the Rust libp2p mesh (oxidize-core/src/mesh): there is no peer
 * discovery, no gossip and no crypto — peers are named explicitly with
 * --peer host:port. It is also not tensor parallelism: the ranks serialize per
 * token (only one rank computes at a time), so this buys CAPACITY, not speed.
 *
 * Splitting reuses each family's *_forward_from hook (llama_forward_from /
 * gemma4_forward_from), which runs layers [l0, n_layers) with the residual
 * stream already in the model's x buffer. Narrowing the model's own n_layers to
 * l1 for the duration of a call turns that tail hook into an exact [l0, l1)
 * slice. Only families that expose such a hook can split — llama and gemma4;
 * qwen36 (recurrent DeltaNet state) and deepseek do not, and init fails loudly.
 *
 * With the weights mmap'd (see gguf.c), a rank only faults in the pages of the
 * layers it actually runs, so its resident set is bounded by its slice even
 * though it maps the whole file — that is what lets the pipeline hold a model
 * whose per-rank working set fits where the whole model would not. Every rank
 * still needs the file on disk. */
#ifndef OC_DISTRIBUTED_H
#define OC_DISTRIBUTED_H

#include <stddef.h>
#include <stdint.h>

#include "model.h"

/* Wire frame: a 20-byte little-endian header (magic, kind, pos, n_floats)
 * followed by n_floats IEEE-754 f32 values, little-endian. x86 and arm are both
 * little-endian, so the payload is copied raw; a mixed-endian cluster would have
 * to byte-swap it (out of scope, and asserted nowhere — do not run one). */
#define OC_DIST_MAGIC 0x4F584450u /* 'OXDP' */
enum { OC_DIST_HIDDEN = 1, OC_DIST_LOGITS = 2, OC_DIST_SHUTDOWN = 3 };

/* One rank's endpoint. Fields are public so the acceptance test can build a
 * pipeline over an in-process socketpair without a real TCP bringup. */
typedef struct {
  int rank, nranks;
  int prev_fd, next_fd;    /* socket to rank-1 / rank+1; -1 at the ends */
  size_t l0, l1, n_layers; /* this rank runs layers [l0, l1); n_layers = total */
  size_t hidden, vocab;    /* hidden-state / logit vector lengths */
  Model* model;            /* the loaded model (all layers; only [l0,l1) run) */
} OcPipe;

/* Bring up rank `rank` of `nranks`. peers[k] for k in [0, nranks-1) is the
 * host:port that rank k+1 listens on — the SAME list is passed to every rank,
 * so only --shard differs between the processes of one pipeline. Binds this
 * rank's listen port (rank>0) and connects to the next rank (rank<nranks-1,
 * retried until it is up), then splits [0, n_layers) evenly across the ranks.
 * Returns 0, or -1 with err set (unsupported arch, bad shard, socket failure). */
int oc_pipe_init(OcPipe* p, int rank, int nranks, const char* const* peers,
                 size_t n_peers, Model* model, char* err, size_t errlen);
void oc_pipe_close(OcPipe* p);

/* Head (rank 0): run the local slice for `token` at position `pos`, push the
 * hidden state up the pipeline, block until the logits return, and write them
 * into logits_out (caller-owned, at least `vocab` floats). 0 / -1 + err. */
int oc_pipe_head_step(OcPipe* p, int32_t token, size_t pos, float* logits_out,
                      char* err, size_t errlen);
/* Head: send a shutdown frame so every worker returns from oc_pipe_serve. */
int oc_pipe_head_shutdown(OcPipe* p, char* err, size_t errlen);

/* Worker (rank>0): loop — recv the hidden state from the previous rank, run this
 * slice, then either (last rank) send the logits back, or forward the hidden
 * state to the next rank and relay the logits it returns back down. Returns 0 on
 * a clean shutdown frame, -1 + err on a socket/protocol/forward failure. */
int oc_pipe_serve(OcPipe* p, char* err, size_t errlen);

/* Length-prefixed frame I/O (short reads/writes handled; EPIPE not a signal).
 * recv refuses a frame whose n exceeds cap. Exposed for the framing round-trip
 * test. 0 / -1 + err. */
int oc_dist_send(int fd, uint32_t kind, uint64_t pos, const float* v, uint32_t n,
                 char* err, size_t errlen);
int oc_dist_recv(int fd, uint32_t* kind, uint64_t* pos, float* v, uint32_t cap,
                 uint32_t* n, char* err, size_t errlen);

#endif
