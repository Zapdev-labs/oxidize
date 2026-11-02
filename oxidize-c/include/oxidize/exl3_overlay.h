#pragma once

#include "oxidize/llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Owned mappings and unpacked scales; weight views borrow their matrices. */
void oc_exl3_overlay_free(OcExl3Overlay *overlay);

#ifdef __cplusplus
}
#endif
