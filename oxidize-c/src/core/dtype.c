/* dtype.c — OcDType elementary type enum. */
#include "oxidize/dtype.h"

#include <string.h>

static const struct {
    OcDType      d;
    size_t       size;
    const char  *name;
} kDtypeTable[OC_DTYPE__COUNT] = {
    [OC_DTYPE_F32] = { OC_DTYPE_F32, 4,  "F32"  },
    [OC_DTYPE_F16] = { OC_DTYPE_F16, 2,  "F16"  },
    [OC_DTYPE_BF16] = { OC_DTYPE_BF16, 2, "BF16" },
    [OC_DTYPE_I8]  = { OC_DTYPE_I8,  1,  "I8"   },
    [OC_DTYPE_I16] = { OC_DTYPE_I16, 2,  "I16"  },
    [OC_DTYPE_I32] = { OC_DTYPE_I32, 4,  "I32"  },
    [OC_DTYPE_I64] = { OC_DTYPE_I64, 8,  "I64"  },
    [OC_DTYPE_F64] = { OC_DTYPE_F64, 8,  "F64"  },
};

size_t oc_dtype_size(OcDType d)
{
    if ((unsigned)d >= (unsigned)OC_DTYPE__COUNT) return 0;
    return kDtypeTable[(unsigned)d].size;
}

const char *oc_dtype_name(OcDType d)
{
    if ((unsigned)d >= (unsigned)OC_DTYPE__COUNT) return "?";
    return kDtypeTable[(unsigned)d].name;
}

OcDType oc_dtype_from_str(const char *s)
{
    if (!s) return OC_DTYPE_UNKNOWN;
    for (unsigned i = 0; i < (unsigned)OC_DTYPE__COUNT; i++) {
        if (kDtypeTable[i].name && strcmp(s, kDtypeTable[i].name) == 0) {
            return kDtypeTable[i].d;
        }
    }
    return OC_DTYPE_UNKNOWN;
}
