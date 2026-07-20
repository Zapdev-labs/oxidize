/*
 * oc.h — umbrella public header for the oxidize-c C11 port.
 *
 * Includes all sub-headers so callers `#include <oxidize/oc.h>` once. Mirrors
 * the flat module system of oxidize-core's `lib.rs` (`#[path=...]` flattening).
 *
 * Convention: public types use `Oc` PascalCase, functions `oc_snake_case`,
 * macros/constants `OC_SCREAMING_CASE`. Every subsystem follows the
 * Config + Error + Trait trinity.
 */
#ifndef OXIDIZE_OC_H
#define OXIDIZE_OC_H

#include "error.h"
#include "dtype.h"
#include "arena.h"
#include "hashtable.h"
#include "vector.h"
#include "log.h"
#include "model.h"
#include "gguf.h"
#include "tokenizer.h"
#include "quant.h"

#endif /* OXIDIZE_OC_H */
