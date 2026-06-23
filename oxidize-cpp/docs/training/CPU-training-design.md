# oxidize-cpp CPU training/finetuning design (LoRA + full-FT, small models)

Target box: dual-socket Xeon Silver 4110, 16c/32t, 123 GB RAM, no GPU. Data: ~/rl-finetune/rl-all-sft.jsonl (chat: {"id","messages":[{role,content}]}). Small model: Qwen2.5-0.5B. **GLM-5.2 training is INFEASIBLE on this box (base model ~213GB > 123GB RAM) — out of scope.**

## Approach: SEPARATE fp32 training forward + tape. Do NOT attach a tape to inference forward_single/run_layers (those use quantized weights + shared workspace + CUDA dispatch). Inference path must stay byte-identical. Reuse the pure fp32 scalar ops in tensor.hpp (rms_norm, matvec, swiglu_inplace, geglu_inplace, softmax_inplace).

## New files (ADDITIVE only)
- include/oxidize/{autograd.hpp, train_types.hpp, lora.hpp, train_forward.hpp, train_loss.hpp, train_optim.hpp, train_data.hpp}
- src/train/{autograd.cpp, train_forward.cpp, train_loss.cpp, train_optim.cpp, lora.cpp, train_data.cpp, grad_check.cpp}
- src/cli/train_main.cpp  → CLI target `oxidize-cpp-train`
- CMakeLists.txt: add `oxidize_train` lib (glob src/train/*.cpp, link oxidize_core + OpenMP) + `oxidize-cpp-train` exe. No OXIDIZE_CUDA dep.

## Autograd: minimal reverse-mode tape over fp32 buffers. Backward ops needed + formulas:
- MatMul y=Wx: dx=W^T dy; dW=dy⊗x.
- RmsNorm out=x*r*w (r=1/sqrt(mean(x^2)+eps)): dx=r*(dy*w - x*r^2*(1/n)*Σ(dy*w*x)); dw=dy*x*r. (Qwen plus_one: w→w+1, dw same.)
- RoPE (no params): inverse rotation dx0=dy0*cos+dy1*sin; dx1=-dy0*sin+dy1*cos.
- SwiGLU out=silu(g)*u, s=sigmoid(g): dg=dout*u*s*(1+g*(1-s)); du=dout*g*s.
- GeGLU: analogous with gelu_tanh derivative.
- Attention (materialized, NOT flash): scores=qk*scale; attn=softmax(causal); out=Σ attn*v. Backward: dv+=attn*dout; dattn=dout·v; dscores=attn*(dattn-Σ attn*dattn); dq+=scale*Σ dscores*k; dk+=scale*dscores*q. Save q,k,v,attn per layer (Qwen0.5B seq512: ~48MB total).
- Embedding: scatter-add dx into row token_id (frozen under LoRA).
- CrossEntropy(+softmax): p=softmax(z); L=-log p[y]; dz=p-onehot(y). Apply loss mask (assistant tokens only), mean over active count.

## Optimizer AdamW (per trainable tensor): m,v fp32 same shape.
g'=g+wd*p; m=b1*m+(1-b1)g'; v=b2*v+(1-b2)g'^2; p-=lr*(m/(1-b1^t))/(sqrt(v/(1-b2^t))+1e-8). b1=.9 b2=.999 wd=.01 (not on norms/biases/LoRA-B). Grad accumulation (divide loss by accum). LR: linear warmup + cosine decay.

## LoRA: adapters on attn_q/k/v/o + ffn_gate/up/down. rank=16 alpha=32 (scaling=alpha/rank). A:[r,cols]~N(0,1/sqrt(r)); B:[rows,r]=0 (delta_W=0 at init → inference-identical start). Forward: y = W_frozen*x + scaling*(B*(A*x)). Backward: dAx=A*x; dB+=scaling*dy⊗dAx; dA+=scaling*(B^T dy)⊗x; dx += W^T dy + scaling*A^T(B^T dy). Base W stays quantized/frozen (read via LlamaWeight); only A,B + Adam in fp32. Merge: W += scaling*B*A → optional requantize via quantize_row_q8_0.

## Full-FT (small only): fp32 master weights copied/dequantized from GGUF. Qwen2.5-0.5B (~500M params): weights+grad+m+v = 4*2GB = 7.8GB + activations ~0.8GB ≈ 8.6GB peak — FEASIBLE in 123GB. (h=896, heads=14, kv_heads=2, head_dim=64, inter=4864, layers=24, vocab=151936; embeddings 136M dominate, tied output.)

## Data pipeline: parse jsonl → ChatTemplate (Qwen2.5 im_start/im_end) wrapping existing Tokenizer → token ids + loss mask (1 only on assistant content, not role tags). Batch=1 + grad-accum first (no padding). seq cap default 2048 (≤ ctx). DataLoader pre-tokenizes all samples at construction, shuffles (seeded mt19937_64).

## Verification (REQUIRED, gate completion):
1. grad_check.cpp finite-difference (delta 1e-4, rtol 1%) on: matmul, rmsnorm, swiglu, attention, cross_entropy, lora. As ctest tests.
2. `--overfit-one-batch`: train 200 steps on one batch; loss must fall from ~11.9 (=-log(1/V)) toward <0.5 (LoRA) / <0.1 (full-FT). 
3. Inference unchanged: logits byte-identical before/after adding training code (TrainModel holds const model*, never mutates LlamaLayer/LlamaWeight; separate fp32/adapter buffers).

Verify on box: build oxidize-cpp-train, run grad-checks (ctest), run overfit on qwen0.5b (LoRA then full-FT), confirm qwen inference still token-exact + existing ctest green. Report loss curves + memory (RSS).
