#pragma once

#include "llama.h"

#include "common.h"

#include
#include

// common\_sampler extends llama\_sampler with additional functionality:
//
// \- grammar support
// \- custom sampler logic based on the parameters
// \- history of the last accepted tokens
// \- performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common\_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common\_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common\_sampler;

// llama\_sampler API overloads

// note: can mutate params in some cases
struct common\_sampler \* common\_sampler\_init(const struct llama\_model \* model, struct common\_params\_sampling & params);

void common\_sampler\_free(struct common\_sampler \* gsmpl);

// if is\_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar sampler
void common\_sampler\_accept(struct common\_sampler \* gsmpl, llama\_token token, bool is\_generated);
void common\_sampler\_reset (struct common\_sampler \* gsmpl);
struct common\_sampler \* common\_sampler\_clone (struct common\_sampler \* gsmpl);

// arguments can be nullptr to skip printing
void common\_perf\_print(const struct llama\_context \* ctx, const struct common\_sampler \* gsmpl);

// get the underlying llama\_sampler\_chain
struct llama\_sampler \* common\_sampler\_get(const struct common\_sampler \* gsmpl);

// extended sampling implementation:
//
// \- set logits
// \- apply the configured sampler chain
// \- check if the token fits the grammar (if any)
// \- if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar\_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama\_token common\_sampler\_sample(struct common\_sampler \* gsmpl, struct llama\_context \* ctx, int idx, bool grammar\_first = false);

// generalized version of common\_sampler\_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
// common\_sampler\_sample\_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
// common\_sampler\_sample(gsmpl, ctx, idx);
// common\_sampler\_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
std::vector common\_sampler\_sample\_and\_accept\_n(struct common\_sampler \* gsmpl, struct llama\_context \* ctx, const std::vector & idxs, const llama\_tokens & draft, bool grammar\_first = false);

// assume idxs == \[ 0, 1, 2, ..., draft.size() \]
std::vector common\_sampler\_sample\_and\_accept\_n(struct common\_sampler \* gsmpl, struct llama\_context \* ctx, const llama\_tokens & draft, bool grammar\_first = false);

uint32\_t common\_sampler\_get\_seed(const struct common\_sampler \* gsmpl);

// helpers

// access the internal list of current candidate tokens
// if do\_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama\_token\_data\_array \* common\_sampler\_get\_candidates(struct common\_sampler \* gsmpl, bool do\_sort);

// get the last accepted token
llama\_token common\_sampler\_last(const struct common\_sampler \* gsmpl);

// print the sampler chain into a string
std::string common\_sampler\_print(const struct common\_sampler \* gsmpl);

// get a string representation of the last accepted tokens
std::string common\_sampler\_prev\_str(common\_sampler \* gsmpl, llama\_context \* ctx, int n);

char common\_sampler\_type\_to\_chr(enum common\_sampler\_type cnstr);
std::string common\_sampler\_type\_to\_str(enum common\_sampler\_type cnstr);

std::vector common\_sampler\_types\_from\_names(const std::vector & names, bool allow\_alt\_names);
std::vector common\_sampler\_types\_from\_chars(const std::string & chars);

llama\_sampler \* llama\_sampler\_init\_llg(const llama\_vocab \* vocab,
 const char \* grammar\_kind, const char \* grammar\_data);

struct common\_sampler\_deleter {
 void operator()(common\_sampler \* s) { common\_sampler\_free(s); }
};

typedef std::unique\_ptr common\_sampler\_ptr;