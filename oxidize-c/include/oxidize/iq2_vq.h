#pragma once

#include <stdint.h>

#define OC_IQ2_VQ_BLOCK_ELEMENTS 128u
#define OC_IQ2_VQ_SUBBLOCK_ELEMENTS 32u
#define OC_IQ2_VQ_BLOCK_BYTES 38u
#define OC_IQ2_VQ_CODEBOOK_COUNT 4u
#define OC_IQ2_VQ_CODES_PER_BOOK 256u
#define OC_IQ2_VQ_VALUES_PER_CODE 4u
#define OC_IQ2_VQ_CODEBOOK_VALUES 4096u

extern const int8_t oc_iq2_vq_codebook[OC_IQ2_VQ_CODEBOOK_VALUES];
