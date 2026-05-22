#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"

#include
#include
#include
#include
#include
#include
#include

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template
struct ring\_buffer {
 ring\_buffer(size\_t cap) : capacity(cap), data(cap) {}

 T & front() {
 if (sz == 0) {
 throw std::runtime\_error("ring buffer is empty");
 }
 return data\[first\];
 }

 const T & front() const {
 if (sz == 0) {
 throw std::runtime\_error("ring buffer is empty");
 }
 return data\[first\];
 }

 T & back() {
 if (sz == 0) {
 throw std::runtime\_error("ring buffer is empty");
 }
 return data\[pos\];
 }

 const T & back() const {
 if (sz == 0) {
 throw std::runtime\_error("ring buffer is empty");
 }
 return data\[pos\];
 }

 void push\_back(const T & value) {
 if (sz == capacity) {
 // advance the start when buffer is full
 first = (first + 1) % capacity;
 } else {
 sz++;
 }
 data\[pos\] = value;
 pos = (pos + 1) % capacity;
 }

 T pop\_front() {
 if (sz == 0) {
 throw std::runtime\_error("ring buffer is empty");
 }
 T value = data\[first\];
 first = (first + 1) % capacity;
 sz--;
 return value;
 }

 const T & rat(size\_t i) const {
 if (i >= sz) {
 throw std::runtime\_error("ring buffer: index out of bounds");
 }
 return data\[(first + sz - i - 1) % capacity\];
 }

 std::vector to\_vector() const {
 std::vector result;
 result.reserve(sz);
 for (size\_t i = 0; i < sz; i++) {
 result.push\_back(data\[(first + i) % capacity\]);
 }
 return result;
 }

 void clear() {
 // here only reset the status of the buffer
 sz = 0;
 first = 0;
 pos = 0;
 }

 bool empty() const {
 return sz == 0;
 }

 size\_t size() const {
 return sz;
 }

 size\_t capacity = 0;
 size\_t sz = 0;
 size\_t first = 0;
 size\_t pos = 0;
 std::vector data;
};

struct common\_sampler {
 common\_params\_sampling params;

 struct llama\_sampler \* grmr;
 struct llama\_sampler \* rbudget;
 struct llama\_sampler \* chain;

 ring\_buffer prev;

 std::vector cur;

 llama\_token\_data\_array cur\_p;

 void reset() {
 prev.clear();

 llama\_sampler\_reset(chain);
 }

 void set\_logits(struct llama\_context \* ctx, int idx) {
 const float \* sampled\_probs = llama\_get\_sampled\_probs\_ith (ctx, idx);
 const float \* sampled\_logits = llama\_get\_sampled\_logits\_ith (ctx, idx);
 const llama\_token \* sampled\_ids = llama\_get\_sampled\_candidates\_ith(ctx, idx);

 const llama\_model \* model = llama\_get\_model(ctx);
 const llama\_vocab \* vocab = llama\_model\_get\_vocab(model);

 const int n\_vocab = llama\_vocab\_n\_tokens(vocab);

 if (sampled\_probs) {
 const uint32\_t sampled\_probs\_count = llama\_get\_sampled\_probs\_count\_ith(ctx, idx);
 cur.resize(sampled\_probs\_count);
 for (uint32\_t i = 0; i < sampled\_probs\_count; ++i) {
 cur\[i\] = llama\_token\_data{sampled\_ids\[i\], sampled\_logits\[i\], sampled\_probs\[i\]};
 }
 } else if (sampled\_logits) {
 const uint32\_t sampled\_logits\_count = llama\_get\_sampled\_logits\_count\_ith(ctx, idx);
 cur.resize(sampled\_logits\_count);
 for (uint32\_t i = 0; i < sampled\_logits\_count; i++) {
 cur\[i\] = llama\_token\_data{sampled\_ids\[i\], sampled\_logits\[i\], 0.0f};
 }
 } else {
 const auto \* logits = llama\_get\_logits\_ith(ctx, idx);
 GGML\_ASSERT(logits != nullptr);
 cur.resize(n\_vocab);
 for (llama\_token token\_id = 0; token\_id < n\_vocab; token\_id++) {
 cur\[token\_id\] = llama\_token\_data{token\_id, logits\[token\_id\], 0.0f};
 }
 }

 cur\_p = { cur.data(), cur.size(), -1, false };
 }

 common\_time\_meas tm() {
 return common\_time\_meas(t\_total\_us, params.no\_perf);
 }

 mutable int64\_t t\_total\_us = 0;
};

std::string common\_params\_sampling::print() const {
 char result\[1024\];

 snprintf(result, sizeof(result),
 "\\trepeat\_last\_n = %d, repeat\_penalty = %.3f, frequency\_penalty = %.3f, presence\_penalty = %.3f\\n"
 "\\tdry\_multiplier = %.3f, dry\_base = %.3f, dry\_allowed\_length = %d, dry\_penalty\_last\_n = %d\\n"
 "\\ttop\_k = %d, top\_p = %.3f, min\_p = %.3f, xtc\_probability = %.3f, xtc\_threshold = %.3f, typical\_p = %.3f, top\_n\_sigma = %.3f, temp = %.3f\\n"
 "\\tmirostat = %d, mirostat\_lr = %.3f, mirostat\_ent = %.3f, adaptive\_target = %.3f, adaptive\_decay = %.3f",
 penalty\_last\_n, penalty\_repeat, penalty\_freq, penalty\_present,
 dry\_multiplier, dry\_base, dry\_allowed\_length, dry\_penalty\_last\_n,
 top\_k, top\_p, min\_p, xtc\_probability, xtc\_threshold, typ\_p, top\_n\_sigma, temp,
 mirostat, mirostat\_eta, mirostat\_tau, adaptive\_target, adaptive\_decay);

 return std::string(result);
}

struct common\_sampler \* common\_sampler\_init(const struct llama\_model \* model, struct common\_params\_sampling & params) {
 const llama\_vocab \* vocab = llama\_model\_get\_vocab(model);

 llama\_sampler\_chain\_params lparams = llama\_sampler\_chain\_default\_params();

 lparams.no\_perf = params.no\_perf;

 llama\_sampler \* grmr = nullptr;
 llama\_sampler \* rbudget = nullptr;
 llama\_sampler \* chain = llama\_sampler\_chain\_init(lparams);

 std::vector samplers;

 const std::string & grammar\_str = common\_grammar\_value(params.grammar);
 if (grammar\_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA\_USE\_LLGUIDANCE
 grmr = llama\_sampler\_init\_llg(vocab, "lark", grammar\_str.c\_str());
#else
 GGML\_ABORT("llguidance (cmake -DLLAMA\_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA\_USE\_LLGUIDANCE
 } else {
 std::vector trigger\_patterns;
 std::vector trigger\_tokens;
 for (const auto & trigger : params.grammar\_triggers) {
 switch (trigger.type) {
 case COMMON\_GRAMMAR\_TRIGGER\_TYPE\_WORD:
 {
 const auto & word = trigger.value;
 trigger\_patterns.push\_back(regex\_escape(word));
 break;
 }
 case COMMON\_GRAMMAR\_TRIGGER\_TYPE\_PATTERN:
 {
 trigger\_patterns.push\_back(trigger.value);
 break;
 }
 case COMMON\_GRAMMAR\_TRIGGER\_TYPE\_PATTERN\_FULL:
 {
 const auto & pattern = trigger.value;
 std::string anchored = "^$";
 if (!pattern.empty()) {
 anchored = (pattern.front() != '^' ? "^" : "")
 \+ pattern
 \+ (pattern.back() != '$' ? "$" : "");
 }
 trigger\_patterns.push\_back(anchored);
 break;
 }
 case COMMON\_GRAMMAR\_TRIGGER\_TYPE\_TOKEN:
 {
 const auto token = trigger.token;
 trigger\_tokens.push\_back(token);
 break;
 }
 default:
 GGML\_ASSERT(false && "unknown trigger type");
 }
 }

 std::vector trigger\_patterns\_c;
 trigger\_patterns\_c.reserve(trigger\_patterns.size());
 for (const auto & regex : trigger\_patterns) {
 trigger\_patterns\_c.push\_back(regex.c\_str());
 }

 if (!grammar\_str.empty()) {
 if (params.grammar\_lazy) {
 grmr = llama\_sampler\_init\_grammar\_lazy\_patterns(vocab, grammar\_str.c\_str(), "root",
 trigger\_patterns\_c.data(), trigger\_patterns\_c.size(),
 trigger\_tokens.data(), trigger\_tokens.size());
 } else {
 grmr = llama\_sampler\_init\_grammar(vocab, grammar\_str.c\_str(), "root");
 }
 }
 }

 // Compute prefill tokens from the generation prompt
 std::vector prefill\_tokens;
 if (!params.generation\_prompt.empty()) {
 GGML\_ASSERT(vocab != nullptr);
 auto tokens = common\_tokenize(vocab, params.generation\_prompt, false, true);
 for (size\_t i = 0; i < tokens.size(); i++) {
 std::string piece = common\_token\_to\_piece(vocab, tokens\[i\], true);
 if (i == 0 && std::isspace(piece\[0\]) && !std::isspace(params.generation\_prompt\[0\])) {
 // Some tokenizers will add a space before the first special token, need to exclude
 continue;
 }
 LOG\_DBG("%s: prefill token: %d = %s\\n", \_\_func\_\_, tokens\[i\], piece.c\_str());
 prefill\_tokens.push\_back(tokens\[i\]);
 }
 }

 // Feed generation prompt tokens to the grammar sampler so it advances past
 // tokens the template already placed in the prompt.
 // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
 if (grmr && !params.grammar\_lazy && common\_grammar\_needs\_prefill(params.grammar)) {
 try {
 for (const auto & token : prefill\_tokens) {
 llama\_sampler\_accept(grmr, token);
 LOG\_DBG("%s: grammar accepted prefill token (%d)\\n", \_\_func\_\_, token);
 }
 } catch (std::exception &e) {
 LOG\_ERR("%s: error initializing grammar sampler for grammar:\\n%s\\n\\nGeneration prompt:\\n'%s'\\n", \_\_func\_\_,
 common\_grammar\_value(params.grammar).c\_str(), params.generation\_prompt.c\_str());
 throw e;
 }
 }

 // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
 if (!params.reasoning\_budget\_start.empty() && !params.reasoning\_budget\_end.empty() && (params.grammar\_lazy \|\| params.reasoning\_budget\_tokens >= 0)) {
 rbudget = common\_reasoning\_budget\_init(
 vocab,
 params.reasoning\_budget\_start,
 params.reasoning\_budget\_end,
 params.reasoning\_budget\_forced,
 params.reasoning\_budget\_tokens < 0 ? INT\_MAX : params.reasoning\_budget\_tokens);

 for (const auto & token : prefill\_tokens) {
 llama\_sampler\_accept(rbudget, token);
 LOG\_DBG("%s: reasoning-budget accepted prefill token (%d)\\n", \_\_func\_\_, token);
 }
 }

 if (params.has\_logit\_bias()) {
 samplers.push\_back(llama\_sampler\_init\_logit\_bias(llama\_vocab\_n\_tokens(vocab), params.logit\_bias.size(), params.logit\_bias.data()));
 }

 if (params.mirostat == 0) {

 bool use\_adaptive\_p = false; // see below

 for (const auto & cnstr : params.samplers) {
 switch (cnstr) {
 case COMMON\_SAMPLER\_TYPE\_DRY:
 {
 std::vector c\_breakers;
 c\_breakers.reserve(params.dry\_sequence\_breakers.size());
 for (const auto & str : params.dry\_sequence\_breakers) {
 c\_breakers.push\_back(str.c\_str());
 }
 samplers.push\_back(llama\_sampler\_init\_dry(vocab, llama\_model\_n\_ctx\_train(model), params.dry\_multiplier, params.dry\_base, params.dry\_allowed\_length, params.dry\_penalty\_last\_n, c\_breakers.data(), c\_breakers.size()));
 }
 break;
 case COMMON\_SAMPLER\_TYPE\_TOP\_K:
 samplers.push\_back(llama\_sampler\_init\_top\_k(params.top\_k));
 break;
 case COMMON\_SAMPLER\_TYPE\_TOP\_P:
 samplers.push\_back(llama\_sampler\_init\_top\_p(params.top\_p, params.min\_keep));
 break;
 case COMMON\_SAMPLER\_TYPE\_TOP\_N\_SIGMA:
 samplers.push\_back(llama\_sampler\_init\_top\_n\_sigma(params.top\_n\_sigma));
 break;
 case COMMON\_SAMPLER\_TYPE\_MIN\_P:
 samplers.push\_back(llama\_sampler\_init\_min\_p(params.min\_p, params.min\_keep));
 break;
 case COMMON\_SAMPLER\_TYPE\_XTC:
 samplers.push\_back(llama\_sampler\_init\_xtc(params.xtc\_probability, params.xtc\_threshold, params.min\_keep, params.seed));
 break;
 case COMMON\_SAMPLER\_TYPE\_TYPICAL\_P:
 samplers.push\_back(llama\_sampler\_init\_typical(params.typ\_p, params.min\_keep));
 break;
 case COMMON\_SAMPLER\_TYPE\_TEMPERATURE:
 samplers.push\_back(llama\_sampler\_init\_temp\_ext(params.temp, params.dynatemp\_range, params.dynatemp\_exponent));
 break;
 case COMMON\_SAMPLER\_TYPE\_INFILL:
 samplers.push\_back(llama\_sampler\_init\_infill(vocab));
 break;
 case COMMON\_SAMPLER\_TYPE\_PENALTIES:
 samplers.push\_back(llama\_sampler\_init\_penalties(params.penalty\_last\_n, params.penalty\_repeat, params.penalty\_freq, params.penalty\_present));
 break;
 case COMMON\_SAMPLER\_TYPE\_ADAPTIVE\_P:
 // the \`adaptive-p\` sampler is like \`dist\` and \`mirostat\` in that it selects
 // a single token, so we will add \`dist\` at the end of the chain by default,
 // unless the user specifically included \`adaptive-p\`. we set this flag here
 // so we know to add the sampler at the very end.
 use\_adaptive\_p = true;
 break;
 default:
 GGML\_ASSERT(false && "unknown sampler type");
 }
 }
 if (use\_adaptive\_p) {
 // only if user explicitly included adaptive-p sampler
 samplers.push\_back(llama\_sampler\_init\_adaptive\_p(params.adaptive\_target, params.adaptive\_decay, params.seed));
 } else {
 // default: sample from distribution
 samplers.push\_back(llama\_sampler\_init\_dist(params.seed));
 }
 } else if (params.mirostat == 1) {
 samplers.push\_back(llama\_sampler\_init\_temp(params.temp));
 samplers.push\_back(llama\_sampler\_init\_mirostat(llama\_vocab\_n\_tokens(vocab), params.seed, params.mirostat\_tau, params.mirostat\_eta, 100));
 } else if (params.mirostat == 2) {
 samplers.push\_back(llama\_sampler\_init\_temp(params.temp));
 samplers.push\_back(llama\_sampler\_init\_mirostat\_v2(params.seed, params.mirostat\_tau, params.mirostat\_eta));
 } else {
 GGML\_ASSERT(false && "unknown mirostat version");
 }

 for (auto \* smpl : samplers) {
 llama\_sampler\_chain\_add(chain, smpl);
 }

 if (grmr && params.backend\_sampling) {
 LOG\_WRN("%s: backend sampling is not compatible with grammar, disabling\\n", \_\_func\_\_);

 params.backend\_sampling = false;
 }

 if (rbudget && params.backend\_sampling) {
 LOG\_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\\n", \_\_func\_\_);

 params.backend\_sampling = false;
 }

 auto \* result = new common\_sampler {
 /\\* .params = \*/ params,
 /\\* .grmr = \*/ grmr,
 /\\* .rbudget = \*/ rbudget,
 /\\* .chain = \*/ chain,
 /\\* .prev = \*/ ring\_buffer(std::max(32, params.n\_prev)),
 /\\* .cur = \*/ {},
 /\\* .cur\_p = \*/ {},
 };

 return result;
}

void common\_sampler\_free(struct common\_sampler \* gsmpl) {
 if (!gsmpl) {
 return;
 }

 llama\_sampler\_free(gsmpl->grmr);
 llama\_sampler\_free(gsmpl->rbudget);
 llama\_sampler\_free(gsmpl->chain);

 delete gsmpl;
}

static bool grammar\_should\_apply(struct common\_sampler \* gsmpl) {
 if (!gsmpl->grmr) {
 return false;
 }
 if (!gsmpl->rbudget) {
 return true;
 }
 if (gsmpl->params.grammar\_lazy) {
 // if grammar is lazy, only apply when reasoning budget is not active
 const auto state = common\_reasoning\_budget\_get\_state(gsmpl->rbudget);
 return state == REASONING\_BUDGET\_IDLE \|\| state == REASONING\_BUDGET\_DONE;
 }
 return true;
}

void common\_sampler\_accept(struct common\_sampler \* gsmpl, llama\_token token, bool is\_generated) {
 if (!gsmpl) {
 return;
 }

 const auto tm = gsmpl->tm();

 // grammar\_should\_apply() checks the reasoning budget state, so calculate this before we accept
 const auto accept\_grammar = is\_generated && grammar\_should\_apply(gsmpl);

 if (gsmpl->rbudget && is\_generated) {
 llama\_sampler\_accept(gsmpl->rbudget, token);
 }

 if (gsmpl->grmr && accept\_grammar) {
 llama\_sampler\_accept(gsmpl->grmr, token);
 }

 llama\_sampler\_accept(gsmpl->chain, token);

 gsmpl->prev.push\_back(token);
}

void common\_sampler\_reset(struct common\_sampler \* gsmpl) {
 if (!gsmpl) {
 return;
 }

 gsmpl->reset();
}

struct common\_sampler \* common\_sampler\_clone(common\_sampler \* gsmpl) {
 return new common\_sampler {
 /\\* .params = \*/ gsmpl->params,
 /\\* .grmr = \*/ llama\_sampler\_clone(gsmpl->grmr),
 /\\* .rbudget = \*/ llama\_sampler\_clone(gsmpl->rbudget),
 /\\* .chain = \*/ llama\_sampler\_clone(gsmpl->chain),
 /\\* .prev = \*/ gsmpl->prev,
 /\\* .cur = \*/ gsmpl->cur,
 /\\* .cur\_p = \*/ gsmpl->cur\_p,
 };
}

void common\_perf\_print(const struct llama\_context \* ctx, const struct common\_sampler \* gsmpl) {
 // TODO: measure grammar performance

 const double t\_sampling\_ms = gsmpl ? 1e-3\*gsmpl->t\_total\_us : 0;

 llama\_perf\_sampler\_data data\_smpl;
 llama\_perf\_context\_data data\_ctx;

 memset(&data\_smpl, 0, sizeof(data\_smpl));
 memset(&data\_ctx, 0, sizeof(data\_ctx));

 if (gsmpl) {
 auto & data = data\_smpl;

 data = llama\_perf\_sampler(gsmpl->chain);

 // note: the sampling time includes the samplers time + extra time spent in common/sampling
 LOG\_INF("%s: sampling time = %10.2f ms\\n", \_\_func\_\_, t\_sampling\_ms);
 LOG\_INF("%s: samplers time = %10.2f ms / %5d tokens\\n", \_\_func\_\_, data.t\_sample\_ms, data.n\_sample);
 }

 if (ctx) {
 auto & data = data\_ctx;

 data = llama\_perf\_context(ctx);

 const double t\_end\_ms = 1e-3 \* ggml\_time\_us();

 const double t\_total\_ms = t\_end\_ms - data.t\_start\_ms;
 const double t\_unacc\_ms = t\_total\_ms - (t\_sampling\_ms + data.t\_p\_eval\_ms + data.t\_eval\_ms);
 const double t\_unacc\_pc = 100.0 \* t\_unacc\_ms / t\_total\_ms;

 LOG\_INF("%s: load time = %10.2f ms\\n", \_\_func\_\_, data.t\_load\_ms);
 LOG\_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\\n",
 \_\_func\_\_, data.t\_p\_eval\_ms, data.n\_p\_eval, data.t\_p\_eval\_ms / data.n\_p\_eval, 1e3 / data.t\_p\_eval\_ms \* data.n\_p\_eval);
 LOG\_INF("%s: eval time = %10.2f ms / %5d runs (%8.2f ms per token, %8.2f tokens per second)\\n",
 \_\_func\_\_, data.t\_eval\_ms, data.n\_eval, data.t\_eval\_ms / data.n\_eval, 1e3 / data.t\_eval\_ms \* data.n\_eval);
 LOG\_INF("%s: total time = %10.2f ms / %5d tokens\\n", \_\_func\_\_, (t\_end\_ms - data.t\_start\_ms), (data.n\_p\_eval + data.n\_eval));
 LOG\_INF("%s: unaccounted time = %10.2f ms / %5.1f %% (total - sampling - prompt eval - eval) / (total)\\n", \_\_func\_\_, t\_unacc\_ms, t\_unacc\_pc);
 LOG\_INF("%s: graphs reused = %10d\\n", \_\_func\_\_, data.n\_reused);

 common\_memory\_breakdown\_print(ctx);
 }
}

struct llama\_sampler \* common\_sampler\_get(const struct common\_sampler \* gsmpl) {
 if (!gsmpl) {
 return nullptr;
 }

 return gsmpl->chain;
}

llama\_token common\_sampler\_sample(struct common\_sampler \* gsmpl, struct llama\_context \* ctx, int idx, bool grammar\_first) {
 llama\_synchronize(ctx);

 // start measuring sampling time after the llama\_context synchronization in order to not measure any ongoing async operations
 const auto tm = gsmpl->tm();

 llama\_token id = LLAMA\_TOKEN\_NULL;

 auto & grmr = gsmpl->grmr;
 auto & rbudget = gsmpl->rbudget;
 auto & chain = gsmpl->chain;
 auto & cur\_p = gsmpl->cur\_p; // initialized by set\_logits

 gsmpl->set\_logits(ctx, idx);

 // Check if a backend sampler has already sampled a token in which case we
 // return that token id directly.
 {
 id = llama\_get\_sampled\_token\_ith(ctx, idx);

 if (id != LLAMA\_TOKEN\_NULL) {
 LOG\_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\\n", \_\_func\_\_, id);

 GGML\_ASSERT(!gsmpl->grmr && "using grammar in combination with backend sampling is not supported");
 GGML\_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

 for (size\_t i = 0; i < cur\_p.size; ++i) {
 if (cur\_p.data\[i\].id == id) {
 cur\_p.selected = i;
 break;
 }
 }

 return id;
 }
 }

 // apply reasoning budget first
 llama\_sampler\_apply(rbudget, &cur\_p);

 if (grammar\_first && grammar\_should\_apply(gsmpl)) {
 llama\_sampler\_apply(grmr, &cur\_p);
 }

 llama\_sampler\_apply(chain, &cur\_p);

 id = cur\_p.data\[cur\_p.selected\].id;

 if (grammar\_first \|\| !grammar\_should\_apply(gsmpl)) {
 return id;
 }

 // check if it the sampled token fits the grammar (grammar-based rejection sampling)
 {
 llama\_token\_data single\_token\_data = { id, 1.0f, 0.0f };
 llama\_token\_data\_array single\_token\_data\_array = { &single\_token\_data, 1, -1, false };

 llama\_sampler\_apply(grmr, &single\_token\_data\_array);

 const bool is\_valid = single\_token\_data\_array.data\[0\].logit != -INFINITY;
 if (is\_valid) {
 return id;
 }
 }

 // resampling:
 // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
 gsmpl->set\_logits(ctx, idx);

 llama\_sampler\_apply(rbudget, &cur\_p);

 if (grammar\_should\_apply(gsmpl)) {
 llama\_sampler\_apply(grmr, &cur\_p);
 }

 llama\_sampler\_apply(chain, &cur\_p);

 GGML\_ASSERT(cur\_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

 id = cur\_p.data\[cur\_p.selected\].id;

 return id;
}

std::vector common\_sampler\_sample\_and\_accept\_n(struct common\_sampler \* gsmpl, struct llama\_context \* ctx, const std::vector & idxs, const llama\_tokens & draft, bool grammar\_first) {
 GGML\_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

 std::vector result;
 result.reserve(idxs.size());

 size\_t i = 0;
 for (; i < draft.size(); i++) {
 const llama\_token id = common\_sampler\_sample(gsmpl, ctx, idxs\[i\], grammar\_first);

 common\_sampler\_accept(gsmpl, id, true);

 result.push\_back(id);

 if (draft\[i\] != id) {
 break;
 }
 }

 if (i == draft.size()) {
 const llama\_token id = common\_sampler\_sample(gsmpl, ctx, idxs\[i\], grammar\_first);

 common\_sampler\_accept(gsmpl, id, true);

 result.push\_back(id);
 }

 return result;
}

std::vector common\_sampler\_sample\_and\_accept\_n(struct common\_sampler \* gsmpl, struct llama\_context \* ctx, const llama\_tokens & draft, bool grammar\_first) {
 std::vector idxs(draft.size() + 1);
 for (size\_t i = 0; i < idxs.size(); ++i) {
 idxs\[i\] = i;
 }

 return common\_sampler\_sample\_and\_accept\_n(gsmpl, ctx, idxs, draft, grammar\_first);
}

uint32\_t common\_sampler\_get\_seed(const struct common\_sampler \* gsmpl) {
 return llama\_sampler\_get\_seed(gsmpl->chain);
}

// helpers

llama\_token\_data\_array \* common\_sampler\_get\_candidates(struct common\_sampler \* gsmpl, bool do\_sort) {
 const auto tm = gsmpl->tm();

 auto \* res = &gsmpl->cur\_p;

 if (do\_sort && !res->sorted) {
 // remember the selected token before sorting
 const llama\_token id = res->data\[res->selected\].id;

 std::sort(res->data, res->data + res->size, \[\](const llama\_token\_data & a, const llama\_token\_data & b) {
 return a.p > b.p;
 });

 // restore the selected token after sorting
 for (size\_t i = 0; i < res->size; ++i) {
 if (res->data\[i\].id == id) {
 res->selected = i;
 break;
 }
 }

 res->sorted = true;
 }

 return res;
}

llama\_token common\_sampler\_last(const struct common\_sampler \* gsmpl) {
 return gsmpl->prev.rat(0);
}

std::string common\_sampler\_print(const struct common\_sampler \* gsmpl) {
 std::string result = "logits ";

 for (int i = 0; i < llama\_sampler\_chain\_n(gsmpl->chain); i++) {
 const auto \* smpl = llama\_sampler\_chain\_get(gsmpl->chain, i);
 result += std::string("-> ");
 result += std::string(llama\_sampler\_name(smpl)) + " ";
 }

 return result;
}

std::string common\_sampler\_prev\_str(common\_sampler \* gsmpl, llama\_context \* ctx\_main, int n) {
 n = std::min(n, (int) gsmpl->prev.size());

 if (n <= 0) {
 return "";
 }

 std::string result;
 result.reserve(8\*n); // 8 is the average length of a token \[citation needed\], TODO: compute this from the vocab

 for (int i = n - 1; i >= 0; i--) {
 const llama\_token id = gsmpl->prev.rat(i);

 GGML\_ASSERT(id != LLAMA\_TOKEN\_NULL && "null token in the sampling history - should not happen");

 result += common\_token\_to\_piece(ctx\_main, id);
 }

 return result;
}

char common\_sampler\_type\_to\_chr(enum common\_sampler\_type cnstr) {
 switch (cnstr) {
 case COMMON\_SAMPLER\_TYPE\_DRY: return 'd';
 case COMMON\_SAMPLER\_TYPE\_TOP\_K: return 'k';
 case COMMON\_SAMPLER\_TYPE\_TYPICAL\_P: return 'y';
 case COMMON\_SAMPLER\_TYPE\_TOP\_P: return 'p';
 case COMMON\_SAMPLER\_TYPE\_TOP\_N\_SIGMA: return 's';
 case COMMON\_SAMPLER\_TYPE\_MIN\_P: return 'm';
 case COMMON\_SAMPLER\_TYPE\_TEMPERATURE: return 't';
 case COMMON\_SAMPLER\_TYPE\_XTC: return 'x';
 case COMMON\_SAMPLER\_TYPE\_INFILL: return 'i';
 case COMMON\_SAMPLER\_TYPE\_PENALTIES: return 'e';
 case COMMON\_SAMPLER\_TYPE\_ADAPTIVE\_P: return 'a';
 default : return '?';
 }
}

std::string common\_sampler\_type\_to\_str(enum common\_sampler\_type cnstr) {
 switch (cnstr) {
 case COMMON\_SAMPLER\_TYPE\_DRY: return "dry";
 case COMMON\_SAMPLER\_TYPE\_TOP\_K: return "top\_k";
 case COMMON\_SAMPLER\_TYPE\_TYPICAL\_P: return "typ\_p";
 case COMMON\_SAMPLER\_TYPE\_TOP\_P: return "top\_p";
 case COMMON\_SAMPLER\_TYPE\_TOP\_N\_SIGMA: return "top\_n\_sigma";
 case COMMON\_SAMPLER\_TYPE\_MIN\_P: return "min\_p";
 case COMMON\_SAMPLER\_TYPE\_TEMPERATURE: return "temperature";
 case COMMON\_SAMPLER\_TYPE\_XTC: return "xtc";
 case COMMON\_SAMPLER\_TYPE\_INFILL: return "infill";
 case COMMON\_SAMPLER\_TYPE\_PENALTIES: return "penalties";
 case COMMON\_SAMPLER\_TYPE\_ADAPTIVE\_P: return "adaptive\_p";
 default : return "";
 }
}

std::vector common\_sampler\_types\_from\_names(const std::vector & names, bool allow\_alt\_names) {
 std::unordered\_map sampler\_canonical\_name\_map {
 { "dry", COMMON\_SAMPLER\_TYPE\_DRY },
 { "top\_k", COMMON\_SAMPLER\_TYPE\_TOP\_K },
 { "top\_p", COMMON\_SAMPLER\_TYPE\_TOP\_P },
 { "top\_n\_sigma", COMMON\_SAMPLER\_TYPE\_TOP\_N\_SIGMA },
 { "typ\_p", COMMON\_SAMPLER\_TYPE\_TYPICAL\_P },
 { "min\_p", COMMON\_SAMPLER\_TYPE\_MIN\_P },
 { "temperature", COMMON\_SAMPLER\_TYPE\_TEMPERATURE },
 { "xtc", COMMON\_SAMPLER\_TYPE\_XTC },
 { "infill", COMMON\_SAMPLER\_TYPE\_INFILL },
 { "penalties", COMMON\_SAMPLER\_TYPE\_PENALTIES },
 { "adaptive\_p", COMMON\_SAMPLER\_TYPE\_ADAPTIVE\_P },
 };

 // since samplers names are written multiple ways
 // make it ready for both system names and input names
 std::unordered\_map sampler\_alt\_name\_map {
 { "top-k", COMMON\_SAMPLER\_TYPE\_TOP\_K },
 { "top-p", COMMON\_SAMPLER\_TYPE\_TOP\_P },
 { "top-n-sigma", COMMON\_SAMPLER\_TYPE\_TOP\_N\_SIGMA },
 { "nucleus", COMMON\_SAMPLER\_TYPE\_TOP\_P },
 { "typical-p", COMMON\_SAMPLER\_TYPE\_TYPICAL\_P },
 { "typical", COMMON\_SAMPLER\_TYPE\_TYPICAL\_P },
 { "typ-p", COMMON\_SAMPLER\_TYPE\_TYPICAL\_P },
 { "typ", COMMON\_SAMPLER\_TYPE\_TYPICAL\_P },
 { "min-p", COMMON\_SAMPLER\_TYPE\_MIN\_P },
 { "temp", COMMON\_SAMPLER\_TYPE\_TEMPERATURE },
 { "adaptive-p", COMMON\_SAMPLER\_TYPE\_ADAPTIVE\_P },
 };

 std::vector samplers;
 samplers.reserve(names.size());

 for (const auto & name : names) {
 auto sampler = sampler\_canonical\_name\_map.find(name);
 if (sampler != sampler\_canonical\_name\_map.end()) {
 samplers.push\_back(sampler->second);
 continue;
 }
 if (allow\_alt\_names) {
 sampler = sampler\_alt\_name\_map.find(name);
 if (sampler != sampler\_alt\_name\_map.end()) {
 samplers.push\_back(sampler->second);
 continue;
 }
 }
 LOG\_WRN("%s: unable to match sampler by name '%s'\\n", \_\_func\_\_, name.c\_str());
 }

 return samplers;
}

std::vector common\_sampler\_types\_from\_chars(const std::string & chars) {
 std::unordered\_map sampler\_name\_map = {
 { common\_sampler\_type\_to\_chr(COMMON\_SAMPLER\_TYPE\_DRY), COMMON\_SAMPLER\_TYPE\_DRY },
 { common\_sampler\_type\_to\_chr(COMMON\_SAMPLER\_TYPE\_TOP\_K), COMMON\_SAMPLER\_TYPE\_TOP\_K },
 { common\_sampler\_type\_to\_chr(COMMON\_SAMPLER\_TYPE\_TYPICAL\_P), COMMON\_SAMPLER\_TYPE\_TYPICAL\_P },
 { common\_sampler\_type\_to\_chr(COMMON\_SAMPLER\_TYPE\_TOP\_P), COMMON\_SAMPLER\_TYPE\_TOP\_P },
 { common\_sampler\_type\_to\_chr(COMMON\_SAMPLER\_TYPE\_TOP\_N\_SIGMA), COMMON\_SAMPLER\_TYPE\_TOP\_N\_SIGMA },
 { common\_sampler\_type\_to\_chr(COMMON\_SAMPLER\_TYPE\_MIN\_P), COMMON\_SAMPLER\_TYPE\_MIN\_P },
 { common\_sampler\_type\_to\_chr(COMMON\_SAMPLER\_TYPE\_TEMPERATURE), COMMON\_SAMPLER\_TYPE\_TEMPERATURE },
 { common\_sampler\_type\_to\_chr(COMMON\_SAMPLER\_TYPE\_XTC), COMMON\_SAMPLER\_TYPE\_XTC },
 { common\_sampler\_type\_to\_chr(COMMON\_SAMPLER\_TYPE\_INFILL), COMMON\_SAMPLER\_TYPE\_INFILL },
 { common\_sampler\_type\_to\_chr(COMMON\_SAMPLER\_TYPE\_PENALTIES), COMMON\_SAMPLER\_TYPE\_PENALTIES },
 { common\_sampler\_type\_to\_chr(COMMON\_SAMPLER\_TYPE\_ADAPTIVE\_P), COMMON\_SAMPLER\_TYPE\_ADAPTIVE\_P },
 };

 std::vector samplers;
 samplers.reserve(chars.size());

 for (const auto & c : chars) {
 const auto sampler = sampler\_name\_map.find(c);
 if (sampler != sampler\_name\_map.end()) {
 samplers.push\_back(sampler->second);
 } else {
 LOG\_WRN("%s: unable to match sampler by char '%c'\\n", \_\_func\_\_, c);
 }
 }

 return samplers;
}