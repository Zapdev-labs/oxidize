"""K2-Horizon-MoVA-36B-A4B on vLLM-TPU: correctness gate + speed benchmark.

Runs inside the vllm-tpu venv on a Kaggle TPU v5e-8 (TP=8). Prints one
`RESULT {json}` line per measurement so the runner can forward them.

The correctness gates (finite perplexity under K2_MAX_PPL, expected keywords in the
greedy chat answers) exit non-zero before the speed benchmark when they fail.

Env overrides: K2_MODEL (path), K2_MAX_LEN, K2_MAX_SEQS, K2_BATCHES ("1,8,32,64"),
K2_OUT_TOKENS, K2_IN_TOKENS, K2_EXTRA_LLM_KWARGS (json), K2_MAX_PPL (default 50),
K2_SKIP_GATES=1 (report gate failures but keep going).
"""
import json
import math
import os
import sys
import time

from vllm import LLM, SamplingParams

MODEL = os.environ["K2_MODEL"]
MAX_LEN = int(os.environ.get("K2_MAX_LEN", "4096"))
MAX_SEQS = int(os.environ.get("K2_MAX_SEQS", "64"))
BATCHES = [int(b) for b in os.environ.get("K2_BATCHES", "1,8,32,64").split(",")]
OUT_TOK = int(os.environ.get("K2_OUT_TOKENS", "256"))
IN_TOK = int(os.environ.get("K2_IN_TOKENS", "128"))
EXTRA = json.loads(os.environ.get("K2_EXTRA_LLM_KWARGS", "{}"))
MAX_PPL = float(os.environ.get("K2_MAX_PPL", "50"))
SKIP_GATES = os.environ.get("K2_SKIP_GATES") == "1"


def result(kind, **kw):
    print("RESULT " + json.dumps({"kind": kind, **kw}), flush=True)


def gate(name, ok, **kw):
    result("gate", gate=name, ok=bool(ok), **kw)
    if not ok and not SKIP_GATES:
        result("done", failed_gate=name)
        sys.exit(2)


PPL_TEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
    "It is named after the engineer Gustave Eiffel, whose company designed and built the tower "
    "from 1887 to 1889. Locally nicknamed \"La dame de fer\" (French for \"Iron Lady\"), it was "
    "constructed as the centrepiece of the 1889 World's Fair, and to crown the centennial "
    "anniversary of the French Revolution. Although initially criticised by some of France's "
    "leading artists and intellectuals for its design, it has since become a global cultural "
    "icon of France and one of the most recognisable structures in the world. The tower received "
    "5,889,000 visitors in 2022. The Eiffel Tower is the most visited monument with an entrance "
    "fee in the world. The tower is 330 metres tall, about the same height as an 81-storey "
    "building, and the tallest structure in Paris. Its base is square, measuring 125 metres on "
    "each side. During its construction, the Eiffel Tower surpassed the Washington Monument to "
    "become the tallest human-made structure in the world, a title it held for 41 years until the "
    "Chrysler Building in New York City was finished in 1930.")

t0 = time.time()
llm = LLM(model=MODEL, tensor_parallel_size=8, max_model_len=MAX_LEN,
          max_num_seqs=MAX_SEQS, trust_remote_code=True, dtype="bfloat16", **EXTRA)
result("load", secs=round(time.time() - t0, 1))
tok = llm.get_tokenizer()

# ---- correctness gate 1: perplexity on neutral factual text (README: wikitext ppl ~15)
out = llm.generate([PPL_TEXT], SamplingParams(max_tokens=1, prompt_logprobs=0))[0]
lps = [next(iter(d.values())).logprob for d in out.prompt_logprobs[1:] if d]
ppl = math.exp(min(-sum(lps) / len(lps), 700.0)) if lps else float("inf")
result("ppl", ppl=round(ppl, 3) if math.isfinite(ppl) else str(ppl), tokens=len(lps))
gate("ppl", lps and math.isfinite(ppl) and ppl <= MAX_PPL,
     ppl=ppl if math.isfinite(ppl) else str(ppl), max_ppl=MAX_PPL)

# ---- correctness gate 2: greedy chat answers
# (question, keywords: at least one must appear, case-insensitive, in the greedy answer)
CHAT_CHECKS = [
    ("What is the capital of Australia? Answer in one sentence.", ["canberra"]),
    ("Write a Python function that returns the n-th Fibonacci number.", ["def "]),
    ("Explain in two sentences why the sky is blue.", ["scatter", "rayleigh"]),
]
qs = [q for q, _ in CHAT_CHECKS]
chats = [[{"role": "user", "content": q}] for q in qs]
try:
    outs = llm.chat(chats, SamplingParams(temperature=0, max_tokens=384),
                    chat_template_kwargs={"reasoning_effort": "low"})
except TypeError:
    outs = llm.chat(chats, SamplingParams(temperature=0, max_tokens=384))
chat_fail = []
for (q, keys), o in zip(CHAT_CHECKS, outs):
    a = o.outputs[0].text
    ok = any(k in a.lower() for k in keys)
    result("chat", q=q, ok=ok, a=a[-1200:])
    if not ok:
        chat_fail.append(q)
gate("chat", len(outs) == len(qs) and not chat_fail, failed=chat_fail)

# ---- speed: fixed-length synthetic prompts, ignore_eos so every request decodes OUT_TOK
base = tok.encode(PPL_TEXT * 8, add_special_tokens=False)


def prompts(n):
    return [{"prompt_token_ids": base[i % 50: i % 50 + IN_TOK]} for i in range(n)]


sp_one = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
sp_gen = SamplingParams(temperature=0, max_tokens=OUT_TOK, ignore_eos=True)
for b in BATCHES:
    if b > MAX_SEQS:
        continue
    ps = prompts(b)
    llm.generate(ps, sp_gen, use_tqdm=False)                 # warm (compile) this shape
    t = time.time(); llm.generate(ps, sp_one, use_tqdm=False); ttft = time.time() - t
    t = time.time(); llm.generate(ps, sp_gen, use_tqdm=False); dt = time.time() - t
    decode = max(dt - ttft, 1e-6)
    result("speed", batch=b, in_tokens=IN_TOK, out_tokens=OUT_TOK,
           wall_s=round(dt, 3), prefill_s=round(ttft, 3),
           out_tok_per_s=round(b * OUT_TOK / dt, 1),
           decode_tok_per_s_per_seq=round((OUT_TOK - 1) / decode, 2),
           decode_tok_per_s_total=round(b * (OUT_TOK - 1) / decode, 1))
result("done", total_secs=round(time.time() - t0, 1))
