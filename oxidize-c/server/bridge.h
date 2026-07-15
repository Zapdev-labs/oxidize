/* C-side glue for the cgo bindings. The Go file that uses //export may not put
 * any C *definitions* in its preamble, so the token-callback trampoline and the
 * generate wrapper live in bridge.c and are only declared here. */
#ifndef OX_BRIDGE_H
#define OX_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize.h"

/* Drive ox_generate with the C trampoline installed. `user` carries a
 * runtime/cgo.Handle (an integer), passed straight through to the Go sink. */
int oxGenerate(OxSession* s, const char* prompt, int max_tokens, uintptr_t user,
               char* err, size_t errlen);

#endif /* OX_BRIDGE_H */
