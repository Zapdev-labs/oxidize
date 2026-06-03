"""Kernel constants mirroring oxidize_core::compute::tensor."""

FLASH_ATTENTION_BLOCK_TOKENS = 64
PARALLEL_GEMV_MIN_OPS = 1 << 20
TRANSPOSED_GEMV_COL_CHUNK = 256

QK8_0 = 32
QK4_0 = 32
QK4_1 = 32
QK5_0 = 32
QK5_1 = 32
QK_K = 256
QK_NVFP4 = 64
QK_NVFP4_SUB = 16

BLOCK_Q8_0_SIZE = 2 + QK8_0
BLOCK_Q4_0_SIZE = 18
BLOCK_Q4_1_SIZE = 20
BLOCK_Q5_0_SIZE = 22
BLOCK_Q5_1_SIZE = 24
BLOCK_Q2_K_SIZE = 84
BLOCK_Q3_K_SIZE = 110
BLOCK_Q4_K_SIZE = 144
BLOCK_Q5_K_SIZE = 176
BLOCK_Q6_K_SIZE = 210
BLOCK_Q8_K_SIZE = 292
BLOCK_NVFP4_SIZE = 34

E2M1_DOUBLED_VALUES: tuple[float, ...] = (
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    12.0,
    0.0,
    -1.0,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
    -8.0,
    -12.0,
)
