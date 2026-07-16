/* HuggingFace SafeTensors -> GGUF converter core (tools/convert.c).
 * The CLI main() is a thin wrapper; the tests drive tool_convert() and the
 * JSON parser directly. See convert.c for the honest scope + permute notes. */
#ifndef OC_TOOLS_CONVERT_H
#define OC_TOOLS_CONVERT_H

#include <stddef.h>

typedef struct {
  const char* arch_override; /* NULL => read config.json model_type */
  const char* outtype;       /* "F32"/"F16"/"Q8_0"/"Q4_0"/"Q4_K"/"Q5_K"/"Q6_K"/"AL5_XS"; NULL => F16 */
} ConvertOpts;

/* 0 on success, nonzero on failure (message on stderr). */
int tool_convert(const char* input, const char* output, const ConvertOpts* opts,
                 int verbose);

/* Strict JSON parser exposed so the tests can prove it rejects malformed input
 * without crashing (ASAN). Returns NULL on any parse error; free with jfree. */
typedef struct JNode JNode;
JNode* json_parse(const char* buf, size_t len);
void jfree(JNode* n);

#endif
