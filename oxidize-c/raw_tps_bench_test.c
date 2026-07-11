#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

static char *read_all(const char *path) {
  FILE *file = fopen(path, "rb");
  if (!file) return NULL;
  if (fseek(file, 0, SEEK_END) != 0) return NULL;
  long length = ftell(file);
  if (length < 0 || fseek(file, 0, SEEK_SET) != 0) return NULL;
  char *text = calloc((size_t)length + 1, 1);
  if (!text || fread(text, 1, (size_t)length, file) != (size_t)length) {
    free(text);
    text = NULL;
  }
  fclose(file);
  return text;
}

static void require_field(const char *json, const char *field) {
  if (!strstr(json, field)) {
    fprintf(stderr, "FAIL raw TPS JSON contract missing %s\n", field);
    exit(1);
  }
}

int main(void) {
  const char *output = "raw-tps-bench-contract.json";
  int status = system("./raw-tps-bench --dry-run > raw-tps-bench-contract.json");
  if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "FAIL raw TPS dry run\n");
    return 1;
  }
  char *json = read_all(output);
  remove(output);
  if (!json) {
    fprintf(stderr, "FAIL raw TPS JSON was not written\n");
    return 1;
  }
  require_field(json, "\"sequences\":1");
  require_field(json, "\"counted_tokens\":\"target_verified_committed\"");
  require_field(json, "\"prefill_seconds\":");
  require_field(json, "\"decode_seconds\":");
  require_field(json, "\"committed_target_tokens\":");
  require_field(json, "\"raw_committed_tps\":");
  require_field(json, "\"committed_token_hash\":");
  require_field(json, "\"draft_proposed_tokens\":0");
  require_field(json, "\"draft_rejected_tokens\":0");
  free(json);

  status = system("./raw-tps-bench --dry-run --concurrency 2 >/dev/null 2>&1");
  if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) == 0) {
    fprintf(stderr, "FAIL raw TPS benchmark accepted aggregate concurrency\n");
    return 1;
  }
  puts("ok raw single-sequence TPS output contract");
  return 0;
}
