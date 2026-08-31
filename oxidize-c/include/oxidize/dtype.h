/* dtype.h — OcDType elementary data type enum. */
#ifndef OXIDIZE_DTYPE_H
#define OXIDIZE_DTYPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_DTYPE_F32 = 0,
    OC_DTYPE_F16,
    OC_DTYPE_BF16,
    OC_DTYPE_I8,
    OC_DTYPE_I16,
    OC_DTYPE_I32,
    OC_DTYPE_I64,
    OC_DTYPE_F64,
    OC_DTYPE__COUNT,
    OC_DTYPE_UNKNOWN = 0xffffffffu,
} OcDType;

/* Size in bytes of one element of the given dtype. Returns 0 for unknown. */
size_t oc_dtype_size(OcDType d);

/* Short string name ("F32", "F16", "BF16", "I8", ...). Returns "?" for unknown. */
const char *oc_dtype_name(OcDType d);

/* Parse a short name into OcDType. Returns OC_DTYPE_UNKNOWN if unrecognized. */
OcDType oc_dtype_from_str(const char *s);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_DTYPE_H */
