#ifndef OXIDIZE_TUI_H
#define OXIDIZE_TUI_H

#include <stdbool.h>
#include <stddef.h>

#include "oxidize/error.h"

struct OcCliContext;

OcError oc_cli_run_tui(struct OcCliContext *ctx);

bool oc_tui_fuzzy(const char *query, const char *hay);
int oc_tui_sse_delta(const char *line, char *out, size_t out_cap);
double oc_tui_metric_value(const char *text, const char *name);

#endif
