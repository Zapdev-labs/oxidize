#include "bridge.h"

/* Defined in oxidize.go via //export oxGoSink. Receives each decoded UTF-8
 * fragment plus the cgo.Handle we threaded through ox_generate's `user`. */
extern int oxGoSink(char* piece, size_t len, uintptr_t user);

/* Matches OxTokenCb; forwards the piece to Go. Nonzero return stops generation. */
static int oxTrampoline(const char* piece, size_t len, void* user) {
  return oxGoSink((char*)piece, len, (uintptr_t)user);
}

int oxGenerate(OxSession* s, const char* prompt, int max_tokens, uintptr_t user,
               char* err, size_t errlen) {
  return ox_generate(s, prompt, max_tokens, oxTrampoline, (void*)user, err, errlen);
}
