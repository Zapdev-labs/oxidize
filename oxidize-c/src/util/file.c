/* file.c — file utility helpers. */
#define _POSIX_C_SOURCE 200809L  /* for stat, S_ISREG */

#include "oxidize/util/file.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

OcError oc_file_size(const char *path, uint64_t *out_size)
{
    if (!path || !out_size) return OC_ERR_INVALID_ARG;
    struct stat st;
    if (stat(path, &st) != 0) return OC_ERR_IO;
    *out_size = (uint64_t)st.st_size;
    return OC_OK;
}

OcError oc_file_read_all(const char *path, uint8_t **out_data, size_t *out_size)
{
    if (!path || !out_data || !out_size) return OC_ERR_INVALID_ARG;
    *out_data = NULL;
    *out_size = 0;

    FILE *f = fopen(path, "rb");
    if (!f) return OC_ERR_IO;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return OC_ERR_IO; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return OC_ERR_IO; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return OC_ERR_IO; }

    /* Allocate at least 1 byte to distinguish empty-file from OOM. */
    size_t alloc = (size_t)sz + 1;
    uint8_t *buf = (uint8_t *)malloc(alloc);
    if (!buf) { fclose(f); return OC_ERR_OOM; }

    size_t read = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (read != (size_t)sz) { free(buf); return OC_ERR_IO; }

    *out_data = buf;
    *out_size = (size_t)sz;
    return OC_OK;
}

OcError oc_file_write_all(const char *path, const uint8_t *data, size_t n)
{
    if (!path) return OC_ERR_INVALID_ARG;
    if (n > 0 && !data) return OC_ERR_INVALID_ARG;
    FILE *f = fopen(path, "wb");
    if (!f) return OC_ERR_IO;
    if (n > 0) {
        size_t w = fwrite(data, 1, n, f);
        if (w != n) { fclose(f); return OC_ERR_IO; }
    }
    if (fclose(f) != 0) return OC_ERR_IO;
    return OC_OK;
}

bool oc_file_exists(const char *path)
{
    if (!path) return false;
    struct stat st;
    if (stat(path, &st) != 0) return false;
    return S_ISREG(st.st_mode);
}
