/* main.c — minimal entry point stub.
 *
 * The full CLI (flag parsing, prompt/chat/serve modes) is implemented by the
 * `cli-flags-modes` feature. This stub exists so `make` produces a runnable
 * `oxidize-c` binary from the foundation library, allowing the milestone gate
 * (`make build` succeeds) to pass during the foundation milestone.
 */
#include "oxidize/oc.h"

#include <stdio.h>

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    oc_log_init_from_env();
    oc_log_info("oxidize-c v0.1.0 (foundation) — CLI not yet implemented");
    printf("oxidize-c v0.1.0 (foundation)\n");
    return 0;
}
