/* CLIP / SigLIP-style vision encoder (the "mmproj" tower of a LLaVA-family
 * multimodal model). Pure C11, CPU. Mirrors the tower in
 * oxidize-core/src/vision/ but implements the full transformer the Rust
 * skeleton stops short of, matching the on-disk format llama.cpp writes.
 *
 * FORMAT. A vision GGUF (usually a separate mmproj-*.gguf) carries clip.*
 * metadata and v.* / mm.* tensors:
 *   geometry   clip.vision.image_size / patch_size / embedding_length /
 *              feed_forward_length / projection_dim / block_count /
 *              attention.head_count / attention.layer_norm_epsilon,
 *              clip.projector_type, clip.use_gelu
 *   tensors    v.patch_embd.weight(+bias), v.class_embd (CLIP only),
 *              v.position_embd.weight, v.pre_ln.*(opt), v.post_ln.*(opt),
 *              v.blk.{i}.{ln1,attn_q,attn_k,attn_v,attn_out,ln2,ffn_up,
 *              ffn_down}.{weight,bias}, mm.0.*(+ mm.2.* for the 2-layer MLP).
 *
 * FORWARD. image (f32 CHW, already resized+normalized to image_size) ->
 * non-overlapping patches -> patch/conv embed + optional class token +
 * learned position embed -> optional pre-LayerNorm -> N encoder blocks
 * (LayerNorm; BIDIRECTIONAL multi-head attention, NO causal mask; MLP with
 * quick-GELU or GELU) -> optional post-LayerNorm -> LLaVA MLP projector
 * (mm.0 -> GELU -> mm.2) -> [n_image_tokens x projection_dim]. Vision towers
 * use LayerNorm (mean+var+bias), not RMSNorm, so this file carries its own
 * layer_norm helper; the projection_dim is the LM hidden size so the rows drop
 * straight into the language model's input-embedding stream.
 *
 * SPLICE (LLaVA). The returned [n_image_tokens x lm_hidden] rows replace the
 * image placeholder token's slot in the LM's *input embedding* sequence: build
 * the text embeddings, then at the <image> position substitute these rows
 * before the first decoder layer. This file produces the rows and reports the
 * shape; wiring them into a decoder needs an embedding-input forward entry the
 * text models here do not expose, so the splice is documented, not executed.
 *
 * SCOPE. Patch/conv embed order matches ggml's [in_ch, kh, kw] weight layout;
 * activation is quick-GELU by default (CLIP) or GELU when clip.use_gelu; only
 * the "mlp" projector is handled (ldp/resampler are not). Image DECODING
 * (JPEG/PNG) and resizing are out of scope: callers pass raw f32 CHW pixels. */
#ifndef OC_VISION_H
#define OC_VISION_H

#include <stddef.h>
#include <stdint.h>

#include "gguf.h"

typedef struct {
  size_t image_size;  /* square input edge in pixels */
  size_t patch_size;  /* square patch edge; image_size % patch_size == 0 */
  size_t hidden;      /* encoder embedding width (embedding_length) */
  size_t n_head;      /* attention heads; head_dim = hidden / n_head */
  size_t n_layer;     /* encoder blocks */
  size_t inter;       /* MLP feed-forward width */
  size_t proj_dim;    /* projector output width == target LM hidden size */
  size_t n_patches;   /* (image_size / patch_size)^2 */
  size_t n_positions; /* n_patches (+1 when a class token is present) */
  float eps;          /* LayerNorm epsilon */
  int has_class_token;/* 1 iff v.class_embd is present (CLIP; SigLIP has none) */
  int use_gelu;       /* MLP activation: 1 = GELU(tanh), 0 = quick-GELU */
} VisionConfig;

typedef struct VisionEncoder VisionEncoder;

/* Load the tower from an already-open vision GGUF. `g` must outlive the encoder
 * (weight matrices are read straight from its mmap). Returns 0 and sets *out on
 * success; -1 with a message in err on a missing/unsupported tensor or geometry.
 * Does NOT take ownership of *g. */
int vision_load(VisionEncoder** out, const GgufFile* g, char* err, size_t errlen);
void vision_free(VisionEncoder* v);

const VisionConfig* vision_config(const VisionEncoder* v);

/* Encode one image. `image_chw` is 3*image_size*image_size f32 pixels in CHW
 * order (channel-major, row-major within a channel), already resized and
 * normalized. Returns a freshly malloc'd [n_image_tokens * proj_dim] row-major
 * buffer (caller frees) and writes the shape into *n_tokens / *dim; NULL with a
 * message in err on failure. n_image_tokens == config.n_patches (the class
 * token, if any, is consumed by attention but dropped from the projector out,
 * as LLaVA does). */
float* vision_encode(VisionEncoder* v, const float* image_chw, size_t* n_tokens,
                     size_t* dim, char* err, size_t errlen);

#endif
