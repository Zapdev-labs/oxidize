#![allow(clippy::manual_checked_ops, clippy::needless_range_loop)]

use crate::gguf::GgufQuantizationType;
use rayon::prelude::*;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK_K: usize = 256;
pub const QK_NVFP4: usize = 64;
pub const QK_NVFP4_SUB: usize = 16;

pub const BLOCK_Q4_0_SIZE: usize = 2 + 16;
pub const BLOCK_Q4_1_SIZE: usize = 2 + 2 + 16;
pub const BLOCK_Q5_0_SIZE: usize = 2 + 4 + 16;
pub const BLOCK_Q5_1_SIZE: usize = 2 + 2 + 4 + 16;
pub const BLOCK_Q8_0_SIZE: usize = 2 + 32;

const fn sizeof_of_f16() -> usize {
    2
}
const fn sizeof_of_f32() -> usize {
    4
}
const fn sizeof_of_i16() -> usize {
    2
}

pub const BLOCK_Q2_K_SIZE: usize = 2 * sizeof_of_f16() + QK_K / 16 + QK_K / 4;
pub const BLOCK_Q3_K_SIZE: usize = sizeof_of_f16() + QK_K / 4 + QK_K / 8 + 12;
pub const BLOCK_Q4_K_SIZE: usize = 2 * sizeof_of_f16() + 12 + QK_K / 2;
pub const BLOCK_Q5_K_SIZE: usize = 2 * sizeof_of_f16() + 12 + QK_K / 2 + QK_K / 8;
pub const BLOCK_Q6_K_SIZE: usize = sizeof_of_f16() + QK_K / 16 + 3 * QK_K / 4;
pub const BLOCK_Q8_K_SIZE: usize = sizeof_of_f32() + QK_K + QK_K / 16 * sizeof_of_i16();

// IQ (importance matrix) quantization block sizes
// block_iq1_s: ggml_half d + uint8_t qs[QK_K/8] + uint16_t qh[QK_K/32]
const BLOCK_IQ1_S_SIZE: usize = sizeof_of_f16() + QK_K / 8 + QK_K / 16;
// block_iq1_m: uint8_t qs[QK_K/8] + uint8_t qh[QK_K/16] + uint8_t scales[QK_K/32]
const BLOCK_IQ1_M_SIZE: usize = QK_K / 8 + QK_K / 16 + QK_K / 32;
// block_nvfp4: uint8_t d[4] (UE4M3 scales) + uint8_t qs[32] (packed E2M1)
pub const BLOCK_NVFP4_SIZE: usize = QK_NVFP4 / QK_NVFP4_SUB + QK_NVFP4 / 2;
// block_iq4_xs: ggml_half d + uint16_t scales_h + uint8_t scales_l[QK_K/64] + uint8_t qs[QK_K/2]
pub const BLOCK_IQ4_XS_SIZE: usize = sizeof_of_f16() + 2 + QK_K / 64 + QK_K / 2;
// block_iq3_s: ggml_half d + uint8_t qs[QK_K/4] + uint8_t qh[QK_K/32] + uint8_t signs[QK_K/8] + uint8_t scales[QK_K/64]
const BLOCK_IQ3_S_SIZE: usize = sizeof_of_f16() + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64;
// IQ4_NL nonlinear codebook (shared by IQ4_NL and IQ4_XS)
pub(crate) const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
// sign mask used by IQ2/IQ3 dequant (kmask_iq2xs)
const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
// iq3s_grid: 512 packed u32 entries (4 nonlinear int8 grid values each, little-endian).
// Generated verbatim from ggml-common.h (ggml-org/llama.cpp) — do not hand-edit.
pub(crate) static IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];
const E2M1_DOUBLED_VALUES: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizationError {
    InvalidInputLength {
        quantization: GgufQuantizationType,
        expected_multiple: usize,
        actual: usize,
    },
    InvalidOutputLength {
        quantization: GgufQuantizationType,
        expected: usize,
        actual: usize,
    },
    InvalidImportanceMatrix {
        reason: &'static str,
    },
    InvalidMixedQuantizationPlan {
        reason: &'static str,
    },
    InvalidMixedInputLength {
        expected: usize,
        actual: usize,
    },
    UnsupportedQuantizationType(GgufQuantizationType),
}

impl std::fmt::Display for QuantizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInputLength {
                quantization,
                expected_multiple,
                actual,
            } => write!(
                f,
                "invalid input length for {quantization:?}: expected multiple of {expected_multiple}, got {actual}"
            ),
            Self::InvalidOutputLength {
                quantization,
                expected,
                actual,
            } => write!(
                f,
                "invalid output length for {quantization:?}: expected {expected}, got {actual}"
            ),
            Self::InvalidImportanceMatrix { reason } => {
                write!(f, "invalid importance matrix: {reason}")
            }
            Self::InvalidMixedQuantizationPlan { reason } => {
                write!(f, "invalid mixed quantization plan: {reason}")
            }
            Self::InvalidMixedInputLength { expected, actual } => {
                write!(
                    f,
                    "invalid mixed quantization input length: expected {expected} bytes, got {actual}"
                )
            }
            Self::UnsupportedQuantizationType(quantization) => {
                write!(f, "unsupported quantization type: {quantization:?}")
            }
        }
    }
}

impl std::error::Error for QuantizationError {}

#[derive(Debug, Clone, PartialEq)]
pub struct IMatrix {
    values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedLayerPlan {
    pub name: String,
    pub value_count: usize,
    pub target: GgufQuantizationType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedLayer {
    pub name: String,
    pub target: GgufQuantizationType,
    pub bytes: Vec<u8>,
}

impl IMatrix {
    pub fn from_values(values: Vec<f32>) -> Result<Self, QuantizationError> {
        if values.is_empty() {
            return Err(QuantizationError::InvalidImportanceMatrix {
                reason: "matrix must not be empty",
            });
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(QuantizationError::InvalidImportanceMatrix {
                reason: "matrix values must be finite",
            });
        }
        if values.iter().any(|v| *v < 0.0) {
            return Err(QuantizationError::InvalidImportanceMatrix {
                reason: "matrix values must be non-negative",
            });
        }
        Ok(Self { values })
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

pub fn quantized_size(
    quantization: GgufQuantizationType,
    value_count: usize,
) -> Result<usize, QuantizationError> {
    let (values_per_block, bytes_per_block) = match quantization {
        GgufQuantizationType::F32 => (1, 4),
        GgufQuantizationType::F16 => (1, 2),
        GgufQuantizationType::I8 => (1, 1),
        GgufQuantizationType::I16 => (1, 2),
        GgufQuantizationType::I32 => (1, 4),
        GgufQuantizationType::I64 => (1, 8),
        GgufQuantizationType::F64 => (1, 8),
        GgufQuantizationType::BF16 => (1, 2),
        GgufQuantizationType::Q4_0 => (QK4_0, BLOCK_Q4_0_SIZE),
        GgufQuantizationType::Q4_1 => (QK4_1, BLOCK_Q4_1_SIZE),
        GgufQuantizationType::Q5_0 => (QK5_0, BLOCK_Q5_0_SIZE),
        GgufQuantizationType::Q5_1 => (QK5_1, BLOCK_Q5_1_SIZE),
        GgufQuantizationType::Q8_0 => (QK8_0, BLOCK_Q8_0_SIZE),
        GgufQuantizationType::Q2_K => (QK_K, BLOCK_Q2_K_SIZE),
        GgufQuantizationType::Q3_K_S
        | GgufQuantizationType::Q3_K_M
        | GgufQuantizationType::Q3_K_L => (QK_K, BLOCK_Q3_K_SIZE),
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => (QK_K, BLOCK_Q4_K_SIZE),
        GgufQuantizationType::Q5_K_S | GgufQuantizationType::Q5_K_M => (QK_K, BLOCK_Q5_K_SIZE),
        GgufQuantizationType::Q6_K => (QK_K, BLOCK_Q6_K_SIZE),
        GgufQuantizationType::IQ1_S => (QK_K, BLOCK_IQ1_S_SIZE),
        GgufQuantizationType::IQ1_M => (QK_K, BLOCK_IQ1_M_SIZE),
        GgufQuantizationType::NVFP4 => (QK_NVFP4, BLOCK_NVFP4_SIZE),
        GgufQuantizationType::IQ2_XXS
        | GgufQuantizationType::IQ2_XS
        | GgufQuantizationType::IQ2_S => (QK_K, BLOCK_Q2_K_SIZE), // approximate
        GgufQuantizationType::IQ3_S => (QK_K, BLOCK_IQ3_S_SIZE),
        GgufQuantizationType::IQ4_XS => (QK_K, BLOCK_IQ4_XS_SIZE),
        GgufQuantizationType::IQ3_XXS => (QK_K, BLOCK_Q3_K_SIZE), // approximate (unsupported dequant)
        GgufQuantizationType::IQ4_NL => (QK_K, BLOCK_Q4_K_SIZE), // approximate (unsupported dequant)
        other => return Err(QuantizationError::UnsupportedQuantizationType(other)),
    };

    if !value_count.is_multiple_of(values_per_block) {
        return Err(QuantizationError::InvalidInputLength {
            quantization,
            expected_multiple: values_per_block,
            actual: value_count,
        });
    }

    Ok((value_count / values_per_block) * bytes_per_block)
}

pub fn quantize_scalar(
    source: GgufQuantizationType,
    target: GgufQuantizationType,
    input: &[u8],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    let value_count = match source {
        GgufQuantizationType::F32 => {
            if !input.len().is_multiple_of(4) {
                return Err(QuantizationError::InvalidInputLength {
                    quantization: source,
                    expected_multiple: 4,
                    actual: input.len(),
                });
            }
            input.len() / 4
        }
        GgufQuantizationType::F16 | GgufQuantizationType::BF16 => {
            if !input.len().is_multiple_of(2) {
                return Err(QuantizationError::InvalidInputLength {
                    quantization: source,
                    expected_multiple: 2,
                    actual: input.len(),
                });
            }
            input.len() / 2
        }
        other => return Err(QuantizationError::UnsupportedQuantizationType(other)),
    };

    let expected_output = quantized_size(target, value_count)?;
    if output.len() != expected_output {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: target,
            expected: expected_output,
            actual: output.len(),
        });
    }

    let mut values = vec![0.0_f32; value_count];
    dequantize_scalar(source, input, &mut values)?;
    quantize_from_f32_scalar(target, &values, output)
}

pub fn quantize_scalar_with_imatrix(
    source: GgufQuantizationType,
    target: GgufQuantizationType,
    input: &[u8],
    output: &mut [u8],
    imatrix: &IMatrix,
) -> Result<(), QuantizationError> {
    let value_count = match source {
        GgufQuantizationType::F32 => {
            if !input.len().is_multiple_of(4) {
                return Err(QuantizationError::InvalidInputLength {
                    quantization: source,
                    expected_multiple: 4,
                    actual: input.len(),
                });
            }
            input.len() / 4
        }
        GgufQuantizationType::F16 | GgufQuantizationType::BF16 => {
            if !input.len().is_multiple_of(2) {
                return Err(QuantizationError::InvalidInputLength {
                    quantization: source,
                    expected_multiple: 2,
                    actual: input.len(),
                });
            }
            input.len() / 2
        }
        other => return Err(QuantizationError::UnsupportedQuantizationType(other)),
    };
    if imatrix.values().len() != value_count {
        return Err(QuantizationError::InvalidImportanceMatrix {
            reason: "matrix length must match input value count",
        });
    }

    let expected_output = quantized_size(target, value_count)?;
    if output.len() != expected_output {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: target,
            expected: expected_output,
            actual: output.len(),
        });
    }

    let mut values = vec![0.0_f32; value_count];
    dequantize_scalar(source, input, &mut values)?;

    // IQ4_XS minimizes a properly importance-weighted error in the encoder, so
    // the raw values and the importance weights are passed through untouched.
    // Other targets keep the legacy behaviour of pre-scaling values by
    // importance before a plain encode.
    if target == GgufQuantizationType::IQ4_XS {
        return quantize_iq4_xs(&values, Some(imatrix.values()), output);
    }

    let weighted_values = values
        .iter()
        .zip(imatrix.values())
        .map(|(value, importance)| value * importance)
        .collect::<Vec<_>>();
    quantize_from_f32_scalar(target, &weighted_values, output)
}

/// Quantize one F16/F32 byte chunk with per-value importance weights.
///
/// `weights` must hold one non-negative value per source element. Only targets
/// with an importance-aware encoder (currently IQ4_XS) consume the weights;
/// other targets fall back to the unweighted encode so callers can use a single
/// streaming path regardless of target.
pub fn quantize_scalar_weighted(
    source: GgufQuantizationType,
    target: GgufQuantizationType,
    input: &[u8],
    output: &mut [u8],
    weights: &[f32],
) -> Result<(), QuantizationError> {
    let value_count = match source {
        GgufQuantizationType::F32 => {
            if !input.len().is_multiple_of(4) {
                return Err(QuantizationError::InvalidInputLength {
                    quantization: source,
                    expected_multiple: 4,
                    actual: input.len(),
                });
            }
            input.len() / 4
        }
        GgufQuantizationType::F16 | GgufQuantizationType::BF16 => {
            if !input.len().is_multiple_of(2) {
                return Err(QuantizationError::InvalidInputLength {
                    quantization: source,
                    expected_multiple: 2,
                    actual: input.len(),
                });
            }
            input.len() / 2
        }
        other => return Err(QuantizationError::UnsupportedQuantizationType(other)),
    };
    if weights.len() != value_count {
        return Err(QuantizationError::InvalidImportanceMatrix {
            reason: "matrix length must match input value count",
        });
    }
    let expected_output = quantized_size(target, value_count)?;
    if output.len() != expected_output {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: target,
            expected: expected_output,
            actual: output.len(),
        });
    }

    let mut values = vec![0.0_f32; value_count];
    dequantize_scalar(source, input, &mut values)?;
    match target {
        GgufQuantizationType::IQ4_XS => quantize_iq4_xs(&values, Some(weights), output),
        other => quantize_from_f32_scalar(other, &values, output),
    }
}

pub fn quantize_mixed_scalar(
    source: GgufQuantizationType,
    input: &[u8],
    plans: &[MixedLayerPlan],
) -> Result<Vec<QuantizedLayer>, QuantizationError> {
    if plans.is_empty() {
        return Err(QuantizationError::InvalidMixedQuantizationPlan {
            reason: "at least one layer plan is required",
        });
    }

    let source_bytes_per_value = match source {
        GgufQuantizationType::F32 => 4,
        GgufQuantizationType::F16 | GgufQuantizationType::BF16 => 2,
        other => return Err(QuantizationError::UnsupportedQuantizationType(other)),
    };
    let mut expected_total_bytes = 0_usize;
    for plan in plans {
        if plan.name.is_empty() {
            return Err(QuantizationError::InvalidMixedQuantizationPlan {
                reason: "layer name must not be empty",
            });
        }
        if plan.value_count == 0 {
            return Err(QuantizationError::InvalidMixedQuantizationPlan {
                reason: "layer value_count must be greater than zero",
            });
        }
        let layer_bytes = plan.value_count.checked_mul(source_bytes_per_value).ok_or(
            QuantizationError::InvalidMixedQuantizationPlan {
                reason: "layer value_count overflows byte calculation",
            },
        )?;
        expected_total_bytes = expected_total_bytes.checked_add(layer_bytes).ok_or(
            QuantizationError::InvalidMixedQuantizationPlan {
                reason: "total planned input size overflows byte calculation",
            },
        )?;
    }

    if input.len() != expected_total_bytes {
        return Err(QuantizationError::InvalidMixedInputLength {
            expected: expected_total_bytes,
            actual: input.len(),
        });
    }

    if plans.len() <= 1 {
        return quantize_mixed_scalar_sequential(source, source_bytes_per_value, input, plans);
    }

    let mut offset = 0_usize;
    let mut jobs = Vec::with_capacity(plans.len());
    for (index, plan) in plans.iter().enumerate() {
        let layer_input_len = plan.value_count.checked_mul(source_bytes_per_value).ok_or(
            QuantizationError::InvalidMixedQuantizationPlan {
                reason: "layer value_count overflows byte calculation",
            },
        )?;
        jobs.push((index, plan, offset, layer_input_len));
        offset += layer_input_len;
    }

    let thread_count = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(plans.len());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count.max(1))
        .build()
        .map_err(|_| QuantizationError::InvalidMixedQuantizationPlan {
            reason: "failed to initialize layer thread pool",
        })?;

    let mut indexed_layers = pool.install(|| {
        jobs.par_iter()
            .map(|(index, plan, offset, layer_input_len)| {
                let layer_input = &input[*offset..*offset + *layer_input_len];
                let layer_output_len = quantized_size(plan.target, plan.value_count)?;
                let mut layer_output = vec![0_u8; layer_output_len];
                quantize_scalar(source, plan.target, layer_input, &mut layer_output)?;
                Ok::<_, QuantizationError>((
                    *index,
                    QuantizedLayer {
                        name: plan.name.clone(),
                        target: plan.target,
                        bytes: layer_output,
                    },
                ))
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    indexed_layers.sort_unstable_by_key(|(index, _)| *index);
    Ok(indexed_layers.into_iter().map(|(_, layer)| layer).collect())
}

fn quantize_mixed_scalar_sequential(
    source: GgufQuantizationType,
    source_bytes_per_value: usize,
    input: &[u8],
    plans: &[MixedLayerPlan],
) -> Result<Vec<QuantizedLayer>, QuantizationError> {
    let mut offset = 0_usize;
    let mut output_layers = Vec::with_capacity(plans.len());
    for plan in plans {
        let layer_input_len = plan.value_count.checked_mul(source_bytes_per_value).ok_or(
            QuantizationError::InvalidMixedQuantizationPlan {
                reason: "layer value_count overflows byte calculation",
            },
        )?;
        let layer_input = &input[offset..offset + layer_input_len];
        let layer_output_len = quantized_size(plan.target, plan.value_count)?;
        let mut layer_output = vec![0_u8; layer_output_len];
        quantize_scalar(source, plan.target, layer_input, &mut layer_output)?;
        output_layers.push(QuantizedLayer {
            name: plan.name.clone(),
            target: plan.target,
            bytes: layer_output,
        });
        offset += layer_input_len;
    }
    Ok(output_layers)
}

pub fn dequantize_scalar(
    quantization: GgufQuantizationType,
    input: &[u8],
    output: &mut [f32],
) -> Result<(), QuantizationError> {
    match quantization {
        GgufQuantizationType::F32 => {
            dequantize_f32_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::F16 => {
            dequantize_f16_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::BF16 => {
            dequantize_bf16_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q4_0 => {
            dequantize_q4_0_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q4_1 => {
            dequantize_q4_1_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q5_0 => {
            dequantize_q5_0_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q5_1 => {
            dequantize_q5_1_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q8_0 => {
            dequantize_q8_0_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q2_K => {
            dequantize_q2_k_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q3_K_S
        | GgufQuantizationType::Q3_K_M
        | GgufQuantizationType::Q3_K_L => {
            dequantize_q3_k_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => {
            dequantize_q4_k_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q5_K_S | GgufQuantizationType::Q5_K_M => {
            dequantize_q5_k_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::Q6_K => {
            dequantize_q6_k_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::IQ1_S => {
            dequantize_iq1_s_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::IQ1_M => {
            dequantize_iq1_m_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::NVFP4 => {
            dequantize_nvfp4_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::IQ4_XS => {
            dequantize_iq4_xs_scalar(input, output)?;
            Ok(())
        }
        GgufQuantizationType::IQ3_S => {
            dequantize_iq3_s_scalar(input, output)?;
            Ok(())
        }
        other => Err(QuantizationError::UnsupportedQuantizationType(other)),
    }
}

fn quantize_from_f32_scalar(
    target: GgufQuantizationType,
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    match target {
        GgufQuantizationType::F32 => quantize_f32_scalar(input, output),
        GgufQuantizationType::F16 => quantize_f16_scalar(input, output),
        GgufQuantizationType::Q4_0 => quantize_q4_0_scalar(input, output),
        GgufQuantizationType::Q4_1 => quantize_q4_1_scalar(input, output),
        GgufQuantizationType::Q5_0 => quantize_q5_0_scalar(input, output),
        GgufQuantizationType::Q5_1 => quantize_q5_1_scalar(input, output),
        GgufQuantizationType::Q8_0 => quantize_q8_0_scalar(input, output),
        GgufQuantizationType::Q2_K => quantize_k_packed_scalar(
            GgufQuantizationType::Q2_K,
            input,
            output,
            BLOCK_Q2_K_SIZE,
            2,
            1.5,
        ),
        GgufQuantizationType::Q3_K_S
        | GgufQuantizationType::Q3_K_M
        | GgufQuantizationType::Q3_K_L => {
            quantize_k_packed_scalar(target, input, output, BLOCK_Q3_K_SIZE, 3, 3.5)
        }
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => {
            quantize_q4_k_scalar(target, input, output)
        }
        GgufQuantizationType::Q5_K_S | GgufQuantizationType::Q5_K_M => {
            quantize_k_packed_scalar(target, input, output, BLOCK_Q5_K_SIZE, 5, 16.0)
        }
        GgufQuantizationType::Q6_K => quantize_k_packed_scalar(
            GgufQuantizationType::Q6_K,
            input,
            output,
            BLOCK_Q6_K_SIZE,
            6,
            32.0,
        ),
        GgufQuantizationType::IQ4_XS => quantize_iq4_xs(input, None, output),
        other => Err(QuantizationError::UnsupportedQuantizationType(other)),
    }
}

fn quantize_f32_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    if output.len() != input.len() * 4 {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::F32,
            expected: input.len() * 4,
            actual: output.len(),
        });
    }
    for (value, bytes) in input.iter().zip(output.chunks_exact_mut(4)) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn quantize_f16_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    if output.len() != input.len() * 2 {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::F16,
            expected: input.len() * 2,
            actual: output.len(),
        });
    }
    for (value, bytes) in input.iter().zip(output.chunks_exact_mut(2)) {
        bytes.copy_from_slice(&f32_to_f16_bits(*value).to_le_bytes());
    }
    Ok(())
}

pub(crate) fn quantize_q8_0_scalar(
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK8_0) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::Q8_0,
            expected_multiple: QK8_0,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK8_0) * BLOCK_Q8_0_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::Q8_0,
            expected: (input.len() / QK8_0) * BLOCK_Q8_0_SIZE,
            actual: output.len(),
        });
    }

    for (in_block, out_block) in input
        .chunks_exact(QK8_0)
        .zip(output.chunks_exact_mut(BLOCK_Q8_0_SIZE))
    {
        let max_abs = in_block.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        let d = if max_abs == 0.0 { 0.0 } else { max_abs / 127.0 };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for (value, dst) in in_block.iter().zip(out_block[2..].iter_mut()) {
            let q = if d == 0.0 {
                0
            } else {
                (value / d).round().clamp(-128.0, 127.0) as i32
            };
            *dst = (q as i8) as u8;
        }
    }

    Ok(())
}

fn quantize_q4_0_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK4_0) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::Q4_0,
            expected_multiple: QK4_0,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK4_0) * BLOCK_Q4_0_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::Q4_0,
            expected: (input.len() / QK4_0) * BLOCK_Q4_0_SIZE,
            actual: output.len(),
        });
    }

    for (in_block, out_block) in input
        .chunks_exact(QK4_0)
        .zip(output.chunks_exact_mut(BLOCK_Q4_0_SIZE))
    {
        let max_abs = in_block.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        let d = if max_abs == 0.0 { 0.0 } else { max_abs / 8.0 };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for i in 0..16 {
            let q_low = if d == 0.0 {
                8_u8
            } else {
                ((in_block[2 * i] / d).round() as i32 + 8).clamp(0, 15) as u8
            };
            let q_high = if d == 0.0 {
                8_u8
            } else {
                ((in_block[2 * i + 1] / d).round() as i32 + 8).clamp(0, 15) as u8
            };
            out_block[2 + i] = q_low | (q_high << 4);
        }
    }

    Ok(())
}

fn quantize_q4_1_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    quantize_linear_4bit(
        input,
        output,
        GgufQuantizationType::Q4_1,
        BLOCK_Q4_1_SIZE,
        QK4_1,
        4,
        15.0,
    )
}

fn quantize_q5_0_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK5_0) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::Q5_0,
            expected_multiple: QK5_0,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK5_0) * BLOCK_Q5_0_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::Q5_0,
            expected: (input.len() / QK5_0) * BLOCK_Q5_0_SIZE,
            actual: output.len(),
        });
    }

    for (in_block, out_block) in input
        .chunks_exact(QK5_0)
        .zip(output.chunks_exact_mut(BLOCK_Q5_0_SIZE))
    {
        let max_abs = in_block.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        let d = if max_abs == 0.0 { 0.0 } else { max_abs / 16.0 };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out_block[2..6].fill(0);

        for (i, value) in in_block.iter().enumerate() {
            let q = if d == 0.0 {
                16_u8
            } else {
                ((value / d).round() as i32 + 16).clamp(0, 31) as u8
            };
            if q & 0x10 != 0 {
                out_block[2 + i / 8] |= 1 << (i % 8);
            }
            let low = q & 0x0F;
            let qs_index = 6 + i / 2;
            if i % 2 == 0 {
                out_block[qs_index] = low;
            } else {
                out_block[qs_index] |= low << 4;
            }
        }
    }

    Ok(())
}

fn quantize_q5_1_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK5_1) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::Q5_1,
            expected_multiple: QK5_1,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK5_1) * BLOCK_Q5_1_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::Q5_1,
            expected: (input.len() / QK5_1) * BLOCK_Q5_1_SIZE,
            actual: output.len(),
        });
    }

    for (in_block, out_block) in input
        .chunks_exact(QK5_1)
        .zip(output.chunks_exact_mut(BLOCK_Q5_1_SIZE))
    {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for value in in_block {
            min = min.min(*value);
            max = max.max(*value);
        }
        let d = if max <= min { 0.0 } else { (max - min) / 31.0 };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out_block[2..4].copy_from_slice(&f32_to_f16_bits(min).to_le_bytes());
        out_block[4..8].fill(0);

        for (i, value) in in_block.iter().enumerate() {
            let q = if d == 0.0 {
                0_u8
            } else {
                ((value - min) / d).round().clamp(0.0, 31.0) as u8
            };
            if q & 0x10 != 0 {
                out_block[4 + i / 8] |= 1 << (i % 8);
            }
            let low = q & 0x0F;
            let qs_index = 8 + i / 2;
            if i % 2 == 0 {
                out_block[qs_index] = low;
            } else {
                out_block[qs_index] |= low << 4;
            }
        }
    }

    Ok(())
}

fn quantize_linear_4bit(
    input: &[f32],
    output: &mut [u8],
    quantization: GgufQuantizationType,
    block_size: usize,
    values_per_block: usize,
    payload_offset: usize,
    levels: f32,
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(values_per_block) {
        return Err(QuantizationError::InvalidInputLength {
            quantization,
            expected_multiple: values_per_block,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / values_per_block) * block_size {
        return Err(QuantizationError::InvalidOutputLength {
            quantization,
            expected: (input.len() / values_per_block) * block_size,
            actual: output.len(),
        });
    }

    for (in_block, out_block) in input
        .chunks_exact(values_per_block)
        .zip(output.chunks_exact_mut(block_size))
    {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for value in in_block {
            min = min.min(*value);
            max = max.max(*value);
        }
        let d = if max <= min {
            0.0
        } else {
            (max - min) / levels
        };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out_block[2..4].copy_from_slice(&f32_to_f16_bits(min).to_le_bytes());
        for i in 0..(values_per_block / 2) {
            let q_low = if d == 0.0 {
                0_u8
            } else {
                ((in_block[2 * i] - min) / d).round().clamp(0.0, 15.0) as u8
            };
            let q_high = if d == 0.0 {
                0_u8
            } else {
                ((in_block[2 * i + 1] - min) / d).round().clamp(0.0, 15.0) as u8
            };
            out_block[payload_offset + i] = q_low | (q_high << 4);
        }
    }
    Ok(())
}

/// llama.cpp `nearest_int` — fast round-to-nearest for quant heuristics.
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0;
    (val.to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
}

/// Port of llama.cpp `make_qkx1_quants` (ggml-quants.c).
fn make_qkx1_quants(x: &[f32], l: &mut [u8], the_min: &mut f32, ntry: i32, alpha: f32) -> f32 {
    debug_assert_eq!(x.len(), l.len());
    let n = x.len();
    let nmax = 15;

    let mut min = x[0];
    let mut max = x[0];
    for &v in &x[1..] {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if max == min {
        l.fill(0);
        *the_min = 0.0;
        return 0.0;
    }
    if min > 0.0 {
        min = 0.0;
    }

    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;

    for _ in 0..ntry {
        let mut sumlx = 0.0_f32;
        let mut suml2 = 0_i32;
        let mut did_change = false;
        for (i, &xv) in x.iter().enumerate() {
            let mut ql = nearest_int(iscale * (xv - min));
            ql = ql.clamp(0, nmax);
            if l[i] != ql as u8 {
                l[i] = ql as u8;
                did_change = true;
            }
            sumlx += (xv - min) * ql as f32;
            suml2 += ql * ql;
        }
        if suml2 > 0 {
            scale = sumlx / suml2 as f32;
        }
        let mut sum = 0.0_f32;
        for (i, &xv) in x.iter().enumerate() {
            sum += xv - scale * l[i] as f32;
        }
        min = alpha * min + (1.0 - alpha) * sum / n as f32;
        if min > 0.0 {
            min = 0.0;
        }
        iscale = 1.0 / scale;
        if !did_change {
            break;
        }
    }

    *the_min = -min;
    scale
}

/// llama.cpp-compatible Q4_K block quantizer (`quantize_row_q4_K_ref` with make_qkx1).
pub fn quantize_q4_k_scalar(
    target: GgufQuantizationType,
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK_K) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: target,
            expected_multiple: QK_K,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK_K) * BLOCK_Q4_K_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: target,
            expected: (input.len() / QK_K) * BLOCK_Q4_K_SIZE,
            actual: output.len(),
        });
    }

    let mut l = [0_u8; QK_K];
    let mut mins = [0.0_f32; QK_K / 32];
    let mut scales = [0.0_f32; QK_K / 32];

    for (in_block, out_block) in input
        .chunks_exact(QK_K)
        .zip(output.chunks_exact_mut(BLOCK_Q4_K_SIZE))
    {
        let mut max_scale = 0.0_f32;
        let mut max_min = 0.0_f32;
        for j in 0..QK_K / 32 {
            let chunk = &in_block[32 * j..32 * j + 32];
            let l_chunk = &mut l[32 * j..32 * j + 32];
            scales[j] = make_qkx1_quants(chunk, l_chunk, &mut mins[j], 5, 0.5);
            if scales[j] > max_scale {
                max_scale = scales[j];
            }
            if mins[j] > max_min {
                max_min = mins[j];
            }
        }

        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

        out_block[4..16].fill(0);
        for j in 0..QK_K / 32 {
            let ls = nearest_int(inv_scale * scales[j]).clamp(0, 63) as u8;
            let lm = nearest_int(inv_min * mins[j]).clamp(0, 63) as u8;
            if j < 4 {
                out_block[4 + j] = ls;
                out_block[4 + j + 4] = lm;
            } else {
                out_block[4 + j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                out_block[4 + j - 4] |= (ls >> 4) << 6;
                out_block[4 + j] |= (lm >> 4) << 6;
            }
        }

        out_block[0..2].copy_from_slice(&f32_to_f16_bits(max_scale / 63.0).to_le_bytes());
        out_block[2..4].copy_from_slice(&f32_to_f16_bits(max_min / 63.0).to_le_bytes());

        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &out_block[4..16]);
            let d = f16_le_to_f32(&out_block[0..2]) * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = f16_le_to_f32(&out_block[2..4]) * m as f32;
            for ii in 0..32 {
                let ql = nearest_int((in_block[32 * j + ii] + dm) / d).clamp(0, 15) as u8;
                l[32 * j + ii] = ql;
            }
        }

        out_block[16..144].fill(0);
        for j in (0..QK_K).step_by(64) {
            for l_idx in 0..32 {
                out_block[16 + (j / 64) * 32 + l_idx] = l[j + l_idx] | (l[j + l_idx + 32] << 4);
            }
        }
    }

    Ok(())
}

/// Nearest index into the (sorted, asymmetric) `KVALUES_IQ4NL` codebook.
///
/// The codebook is deliberately asymmetric: scaling a finite Gaussian sample so
/// its max-magnitude entry maps to a fixed value yields an asymmetric
/// distribution, so a symmetric table reconstructs poorly (see ikawrakow's
/// IQ4_NL design notes). Linear scan over 16 entries is cheap and branchless.
#[inline]
fn best_index_iq4nl(value: f32) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for (idx, &v) in KVALUES_IQ4NL.iter().enumerate() {
        let dist = (value - v as f32).abs();
        if dist < best_dist {
            best_dist = dist;
            best = idx;
        }
    }
    best
}

/// IQ4_XS encoder (inverse of [`dequantize_iq4_xs_scalar`]).
///
/// Non-linear 4-bit at ~4.25 bpw: super-block of 256 = 8 sub-blocks of 32, each
/// reconstructed as `d * ls * KVALUES_IQ4NL[q]` with a 6-bit signed sub-block
/// scale `ls` and a single f16 super-scale `d`. `weights` (when provided) is the
/// per-value importance used to steer error away from activation-heavy columns;
/// `None` falls back to the `x²` heuristic. The per-sub-block scale search is the
/// `ntry` heuristic from ggml — deliberately *not* an exact RMSE solve, which is
/// known to hurt observed quality.
pub fn quantize_iq4_xs(
    input: &[f32],
    weights: Option<&[f32]>,
    output: &mut [u8],
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK_K) {
        return Err(QuantizationError::InvalidInputLength {
            quantization: GgufQuantizationType::IQ4_XS,
            expected_multiple: QK_K,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK_K) * BLOCK_IQ4_XS_SIZE {
        return Err(QuantizationError::InvalidOutputLength {
            quantization: GgufQuantizationType::IQ4_XS,
            expected: (input.len() / QK_K) * BLOCK_IQ4_XS_SIZE,
            actual: output.len(),
        });
    }
    if let Some(w) = weights
        && w.len() != input.len()
    {
        return Err(QuantizationError::InvalidImportanceMatrix {
            reason: "matrix length must match input value count",
        });
    }

    const SUB_BLOCKS: usize = QK_K / 32;
    const NTRY: i32 = 7;
    let val0 = KVALUES_IQ4NL[0] as f32;

    let mut scales = [0.0_f32; SUB_BLOCKS];
    let mut l_idx = [0_u8; QK_K];
    let mut weight = [0.0_f32; 32];

    for (block_idx, (in_block, out_block)) in input
        .chunks_exact(QK_K)
        .zip(output.chunks_exact_mut(BLOCK_IQ4_XS_SIZE))
        .enumerate()
    {
        let block_weights = weights.map(|w| &w[block_idx * QK_K..block_idx * QK_K + QK_K]);

        let mut sigma2 = 0.0_f32;
        for &v in in_block {
            sigma2 += v * v;
        }
        sigma2 *= 2.0 / QK_K as f32;

        out_block.fill(0);

        for sb in 0..SUB_BLOCKS {
            let xb = &in_block[sb * 32..sb * 32 + 32];
            match block_weights {
                Some(qw) => {
                    let qw = &qw[sb * 32..sb * 32 + 32];
                    for j in 0..32 {
                        weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
                    }
                }
                None => {
                    for j in 0..32 {
                        weight[j] = xb[j] * xb[j];
                    }
                }
            }

            let mut amax = 0.0_f32;
            let mut max = 0.0_f32;
            for &v in xb {
                let ax = v.abs();
                if ax > amax {
                    amax = ax;
                    max = v;
                }
            }
            if amax < 1.0e-8 {
                scales[sb] = 0.0;
                for j in 0..32 {
                    l_idx[sb * 32 + j] = 0;
                }
                continue;
            }

            // Evaluate a candidate inverse-scale: returns (Σ w·q·x, Σ w·q²).
            let eval = |inv_scale: f32| -> (f32, f32) {
                let mut sumqx = 0.0_f32;
                let mut sumq2 = 0.0_f32;
                for j in 0..32 {
                    let q = KVALUES_IQ4NL[best_index_iq4nl(inv_scale * xb[j])] as f32;
                    let w = weight[j];
                    sumqx += w * q * xb[j];
                    sumq2 += w * q * q;
                }
                (sumqx, sumq2)
            };

            let inv0 = -val0 / max;
            let (sumqx, sumq2) = eval(inv0);
            let mut best = if sumq2 > 0.0 { sumqx * sumqx / sumq2 } else { 0.0 };
            let mut best_scale = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
            for itry in -NTRY..=NTRY {
                let inv_scale = (itry as f32 + val0) / max;
                let (sx, s2) = eval(inv_scale);
                if s2 > 0.0 && sx * sx > best * s2 {
                    best = sx * sx / s2;
                    best_scale = sx / s2;
                }
            }

            scales[sb] = best_scale;
            let inv_final = if best_scale != 0.0 { 1.0 / best_scale } else { 0.0 };
            for j in 0..32 {
                l_idx[sb * 32 + j] = best_index_iq4nl(inv_final * xb[j]) as u8;
            }
        }

        // Quantize the eight sub-block scales against one f16 super-scale into
        // signed 6-bit values, matching the decoder's `dl = d * (ls - 32)`.
        let mut amax_scale = 0.0_f32;
        let mut max_scale = 0.0_f32;
        for &s in &scales {
            let a = s.abs();
            if a > amax_scale {
                amax_scale = a;
                max_scale = s;
            }
        }
        if amax_scale == 0.0 {
            continue;
        }

        let d_super = -max_scale / 32.0;
        let inv_super = 1.0 / d_super;
        let mut scales_h: u16 = 0;
        for sb in 0..SUB_BLOCKS {
            let ls = nearest_int(inv_super * scales[sb]).clamp(-32, 31);
            let si = (ls + 32) as u16;
            out_block[4 + sb / 2] |= ((si & 0xf) as u8) << (4 * (sb % 2));
            scales_h |= (si >> 4) << (2 * sb);
        }
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d_super).to_le_bytes());
        out_block[2..4].copy_from_slice(&scales_h.to_le_bytes());

        for sb in 0..SUB_BLOCKS {
            let qoff = 8 + sb * 16;
            for k in 0..16 {
                let lo = l_idx[sb * 32 + k] & 0xf;
                let hi = l_idx[sb * 32 + k + 16] & 0xf;
                out_block[qoff + k] = lo | (hi << 4);
            }
        }
    }

    Ok(())
}

/// IQ4_XS encoder without an importance matrix (uses the `x²` weight heuristic).
pub fn quantize_iq4_xs_scalar(input: &[f32], output: &mut [u8]) -> Result<(), QuantizationError> {
    quantize_iq4_xs(input, None, output)
}

fn quantize_k_packed_scalar(
    quantization: GgufQuantizationType,
    input: &[f32],
    output: &mut [u8],
    block_size: usize,
    bits: usize,
    zero_point: f32,
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(QK_K) {
        return Err(QuantizationError::InvalidInputLength {
            quantization,
            expected_multiple: QK_K,
            actual: input.len(),
        });
    }
    if output.len() != (input.len() / QK_K) * block_size {
        return Err(QuantizationError::InvalidOutputLength {
            quantization,
            expected: (input.len() / QK_K) * block_size,
            actual: output.len(),
        });
    }

    let max_q = ((1_u32 << bits) - 1) as f32;
    let positive_span = zero_point.max(max_q - zero_point);
    for (in_block, out_block) in input
        .chunks_exact(QK_K)
        .zip(output.chunks_exact_mut(block_size))
    {
        let max_abs = in_block.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        let d = if max_abs == 0.0 {
            0.0
        } else {
            max_abs / positive_span
        };
        out_block[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out_block[2..].fill(0);

        for (i, value) in in_block.iter().enumerate() {
            let q = if d == 0.0 {
                zero_point.round() as u32
            } else {
                ((value / d) + zero_point).round().clamp(0.0, max_q) as u32
            };
            write_bits(&mut out_block[2..], i, bits, q);
        }
    }

    Ok(())
}

pub fn dequantize_f32_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(GgufQuantizationType::F32, input, output, 4, 1)?;

    for (src, dst) in input.chunks_exact(4).zip(output.iter_mut()) {
        *dst = f32::from_le_bytes([src[0], src[1], src[2], src[3]]);
    }

    Ok(())
}

pub fn dequantize_f16_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(GgufQuantizationType::F16, input, output, 2, 1)?;

    for (src, dst) in input.chunks_exact(2).zip(output.iter_mut()) {
        *dst = f16_le_to_f32(src);
    }

    Ok(())
}

pub fn dequantize_bf16_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(GgufQuantizationType::BF16, input, output, 2, 1)?;

    // BF16 is the high 16 bits of an IEEE-754 f32, so widening is a left shift
    // by 16 bits with zero-filled mantissa — exact, no rounding.
    for (src, dst) in input.chunks_exact(2).zip(output.iter_mut()) {
        let bits = u32::from(u16::from_le_bytes([src[0], src[1]])) << 16;
        *dst = f32::from_bits(bits);
    }

    Ok(())
}

pub fn dequantize_q4_0_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q4_0,
        input,
        output,
        BLOCK_Q4_0_SIZE,
        QK4_0,
    )?;

    for (block, out) in input
        .chunks_exact(BLOCK_Q4_0_SIZE)
        .zip(output.chunks_exact_mut(QK4_0))
    {
        let d = f16_le_to_f32(&block[0..2]);
        for i in 0..16 {
            let packed = block[2 + i];
            out[2 * i] = ((packed & 0x0F) as i8 - 8) as f32 * d;
            out[2 * i + 1] = (((packed >> 4) & 0x0F) as i8 - 8) as f32 * d;
        }
    }

    Ok(())
}

pub fn dequantize_q4_1_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q4_1,
        input,
        output,
        BLOCK_Q4_1_SIZE,
        QK4_1,
    )?;

    for (block, out) in input
        .chunks_exact(BLOCK_Q4_1_SIZE)
        .zip(output.chunks_exact_mut(QK4_1))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let m = f16_le_to_f32(&block[2..4]);
        for i in 0..16 {
            let packed = block[4 + i];
            out[2 * i] = (packed & 0x0F) as f32 * d + m;
            out[2 * i + 1] = ((packed >> 4) & 0x0F) as f32 * d + m;
        }
    }

    Ok(())
}

pub fn dequantize_q5_0_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q5_0,
        input,
        output,
        BLOCK_Q5_0_SIZE,
        QK5_0,
    )?;

    for (block, out) in input
        .chunks_exact(BLOCK_Q5_0_SIZE)
        .zip(output.chunks_exact_mut(QK5_0))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let qh = &block[2..6];
        let qs = &block[6..22];

        for i in 0..QK5_0 {
            let low = if i % 2 == 0 {
                qs[i / 2] & 0x0F
            } else {
                (qs[i / 2] >> 4) & 0x0F
            };
            let high = (qh[i / 8] >> (i % 8)) & 0x01;
            let q = low | (high << 4);
            out[i] = (q as i8 - 16) as f32 * d;
        }
    }

    Ok(())
}

pub fn dequantize_q5_1_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q5_1,
        input,
        output,
        BLOCK_Q5_1_SIZE,
        QK5_1,
    )?;

    for (block, out) in input
        .chunks_exact(BLOCK_Q5_1_SIZE)
        .zip(output.chunks_exact_mut(QK5_1))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let m = f16_le_to_f32(&block[2..4]);
        let qh = &block[4..8];
        let qs = &block[8..24];

        for i in 0..QK5_1 {
            let low = if i % 2 == 0 {
                qs[i / 2] & 0x0F
            } else {
                (qs[i / 2] >> 4) & 0x0F
            };
            let high = (qh[i / 8] >> (i % 8)) & 0x01;
            let q = low | (high << 4);
            out[i] = q as f32 * d + m;
        }
    }

    Ok(())
}

pub fn dequantize_q8_0_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q8_0,
        input,
        output,
        BLOCK_Q8_0_SIZE,
        QK8_0,
    )?;

    for (block, out) in input
        .chunks_exact(BLOCK_Q8_0_SIZE)
        .zip(output.chunks_exact_mut(QK8_0))
    {
        let d = f16_le_to_f32(&block[0..2]);
        for i in 0..QK8_0 {
            out[i] = (block[2 + i] as i8) as f32 * d;
        }
    }

    Ok(())
}

pub fn dequantize_q2_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q2_K,
        input,
        output,
        BLOCK_Q2_K_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_Q2_K_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[80..82]);
        let min = f16_le_to_f32(&block[82..84]);
        let scales = &block[0..16];
        let qs = &block[16..80];
        let mut q_ptr = 0;
        let mut is = 0;
        // Match llama.cpp dequantize_row_q2_K: each outer iteration consumes a
        // fresh 32-byte slab of qs (qs_base advances by 32), and the four inner
        // iterations re-read the same slab with shifts 0/2/4/6.
        for outer in 0..2 {
            let qs_base = outer * 32;
            for _ in 0..4 {
                let sc1 = scales[is];
                let dl1 = d * ((sc1 & 0xF) as f32);
                let ml1 = min * ((sc1 >> 4) as f32);
                is += 1;
                let sc2 = scales[is];
                let dl2 = d * ((sc2 & 0xF) as f32);
                let ml2 = min * ((sc2 >> 4) as f32);
                is += 1;
                let shift = ((is / 2 - 1) % 4) * 2;
                for l in 0..16 {
                    out[q_ptr + l] = dl1 * (((qs[qs_base + l] >> shift) & 3) as f32) - ml1;
                }
                for l in 0..16 {
                    out[q_ptr + 16 + l] =
                        dl2 * (((qs[qs_base + 16 + l] >> shift) & 3) as f32) - ml2;
                }
                q_ptr += 32;
            }
        }
    }
    Ok(())
}

pub fn dequantize_q3_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q3_K_S,
        input,
        output,
        BLOCK_Q3_K_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_Q3_K_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d_all = f16_le_to_f32(&block[108..110]);
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let mut scales_raw = [0u32; 4];
        scales_raw[0] = u32::from_le_bytes([block[96], block[97], block[98], block[99]]);
        scales_raw[1] = u32::from_le_bytes([block[100], block[101], block[102], block[103]]);
        scales_raw[2] = u32::from_le_bytes([block[104], block[105], block[106], block[107]]);
        let tmp = scales_raw[2];
        scales_raw[2] = ((scales_raw[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
        scales_raw[3] = ((scales_raw[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
        scales_raw[0] = (scales_raw[0] & 0x0F0F0F0F) | ((tmp & 0x03030303) << 4);
        scales_raw[1] = (scales_raw[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4);
        let mut scale_bytes = [0u8; 16];
        for (i, word) in scales_raw.iter().enumerate() {
            scale_bytes[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }

        let mut q_ptr = 0;
        let mut is = 0;
        let mut m = 1u8;
        for _ in 0..2 {
            for _ in 0..4 {
                let dl = d_all * (scale_bytes[is] as i8 as i32 - 32) as f32;
                is += 1;
                let shift = ((is - 1) % 4) * 2;
                for l in 0..16 {
                    let qv = ((qs[l] >> shift) & 3) as i32;
                    let hbit = if (hmask[l] & m) != 0 { 0 } else { 4 };
                    out[q_ptr + l] = dl * ((qv - hbit) as f32);
                }
                let dl2 = d_all * (scale_bytes[is] as i8 as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let qv = ((qs[l + 16] >> shift) & 3) as i32;
                    let hbit = if (hmask[l + 16] & m) != 0 { 0 } else { 4 };
                    out[q_ptr + 16 + l] = dl2 * ((qv - hbit) as f32);
                }
                q_ptr += 32;
                m <<= 1;
            }
        }
    }
    Ok(())
}

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

pub fn dequantize_q4_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q4_K_S,
        input,
        output,
        BLOCK_Q4_K_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_Q4_K_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let min = f16_le_to_f32(&block[2..4]);
        let scales = &block[4..16];
        let qs = &block[16..144];
        let mut out_ptr = 0;
        let mut is = 0;
        // Each of 4 group_pairs covers 64 output values (32 from low nibbles + 32 from high nibbles)
        // and reads from qs[group_pair*32 .. group_pair*32+32] — must advance q_base.
        for group_pair in 0..4 {
            let q_base = group_pair * 32;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let min1 = min * m1 as f32;
            let d2 = d * sc2 as f32;
            let min2 = min * m2 as f32;
            for l in 0..32 {
                out[out_ptr + l] = d1 * ((qs[q_base + l] & 0xF) as f32) - min1;
            }
            for l in 0..32 {
                out[out_ptr + 32 + l] = d2 * ((qs[q_base + l] >> 4) as f32) - min2;
            }
            out_ptr += 64;
            is += 2;
        }
    }
    Ok(())
}

pub fn dequantize_q5_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q5_K_S,
        input,
        output,
        BLOCK_Q5_K_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_Q5_K_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let min = f16_le_to_f32(&block[2..4]);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let mut q_ptr = 0;
        let mut is = 0;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let min1 = min * m1 as f32;
            let d2 = d * sc2 as f32;
            let min2 = min * m2 as f32;
            for l in 0..32 {
                let qv1 = (qs[l] & 0xF) as u32 + if (qh[l] & u1) != 0 { 16 } else { 0 };
                out[q_ptr + l] = d1 * (qv1 as f32) - min1;
            }
            for l in 0..32 {
                let qv2 = (qs[l] >> 4) as u32 + if (qh[l] & u2) != 0 { 16 } else { 0 };
                out[q_ptr + 32 + l] = d2 * (qv2 as f32) - min2;
            }
            q_ptr += 64;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(())
}

pub fn dequantize_q6_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q6_K,
        input,
        output,
        BLOCK_Q6_K_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_Q6_K_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[208..210]);
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        // QK_K=256 values are processed in two 128-element groups. Each group
        // advances into ql/qh/scales (ql+=64, qh+=32, scales+=8), matching the
        // reference dequantize_row_q6_K. Without these per-group offsets the
        // second half of every block is decoded from the first half's data.
        let mut q_ptr = 0;
        for group in 0..2 {
            let ql_off = group * 64;
            let qh_off = group * 32;
            let sc_off = group * 8;
            for l in 0..32 {
                let is = l / 16;
                let q1 =
                    ((ql[ql_off + l] & 0xF) as i32 | (((qh[qh_off + l] & 3) as i32) << 4)) - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) as i32
                    | ((((qh[qh_off + l] >> 2) & 3) as i32) << 4))
                    - 32;
                let q3 = ((ql[ql_off + l] >> 4) as i32
                    | ((((qh[qh_off + l] >> 4) & 3) as i32) << 4))
                    - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) as i32
                    | ((((qh[qh_off + l] >> 6) & 3) as i32) << 4))
                    - 32;
                out[q_ptr + l] = d * sc[sc_off + is] as i8 as f32 * q1 as f32;
                out[q_ptr + 32 + l] = d * sc[sc_off + is + 2] as i8 as f32 * q2 as f32;
                out[q_ptr + 64 + l] = d * sc[sc_off + is + 4] as i8 as f32 * q3 as f32;
                out[q_ptr + 96 + l] = d * sc[sc_off + is + 6] as i8 as f32 * q4 as f32;
            }
            q_ptr += 128;
        }
    }
    Ok(())
}

/// IQ4_XS dequantization (ggml `dequantize_row_iq4_xs`). Block = 136 bytes for
/// 256 values: f16 d, u16 scales_h, 4×u8 scales_l, 128×u8 qs (two 4-bit nibbles
/// each). Eight 32-value sub-blocks; per-subblock 6-bit scale (ls-32) selects a
/// scale, and each nibble indexes the shared nonlinear IQ4_NL codebook.
pub fn dequantize_iq4_xs_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::IQ4_XS,
        input,
        output,
        BLOCK_IQ4_XS_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_IQ4_XS_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..136];
        for ib in 0..(QK_K / 32) {
            let ls_l = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32;
            let ls_h = (((scales_h >> (2 * ib)) & 3) as i32) << 4;
            let dl = d * ((ls_l | ls_h) - 32) as f32;
            let qoff = ib * 16;
            let ooff = ib * 32;
            for j in 0..16 {
                let b = qs[qoff + j];
                out[ooff + j] = dl * KVALUES_IQ4NL[(b & 0xf) as usize] as f32;
                out[ooff + j + 16] = dl * KVALUES_IQ4NL[(b >> 4) as usize] as f32;
            }
        }
    }
    Ok(())
}

/// IQ3_S dequantization (ggml `dequantize_row_iq3_s`). Block = 110 bytes for
/// 256 values: f16 d, 64×u8 qs, 8×u8 qh, 32×u8 signs, 4×u8 scales. Each 3-bit
/// index (8th bit from qh) selects a 4-value entry of the iq3s_grid codebook;
/// the sign byte flips signs per kmask; per-32 sub-block scale = d*(1+2*s).
pub fn dequantize_iq3_s_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::IQ3_S,
        input,
        output,
        BLOCK_IQ3_S_SIZE,
        QK_K,
    )?;
    let grid = |idx: usize, j: usize| -> f32 { ((IQ3S_GRID[idx] >> (8 * j)) & 0xff) as f32 };
    for (block, out) in input
        .chunks_exact(BLOCK_IQ3_S_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let qs = &block[2..66]; // 64 bytes
        let qh = &block[66..74]; // 8 bytes
        let signs = &block[74..106]; // 32 bytes
        let scales = &block[106..110]; // 4 bytes
        let mut qs_o = 0usize; // index into qs
        let mut qh_o = 0usize; // index into qh
        let mut sg_o = 0usize; // index into signs
        let mut y = 0usize; // index into out
        let mut ib32 = 0usize;
        while ib32 < QK_K / 32 {
            let db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf) as i32) as f32;
            let db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4) as i32) as f32;
            // first 32: uses qh[qh_o], qs_o..qs_o+8, signs sg_o..sg_o+4
            for l in 0..4 {
                let h = qh[qh_o] as usize;
                let i1 = qs[qs_o + 2 * l] as usize | ((h << (8 - 2 * l)) & 256);
                let i2 = qs[qs_o + 2 * l + 1] as usize | ((h << (7 - 2 * l)) & 256);
                let s = signs[sg_o + l];
                for j in 0..4 {
                    let f1 = if s & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let f2 = if s & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out[y + j] = db1 * grid(i1, j) * f1;
                    out[y + j + 4] = db1 * grid(i2, j) * f2;
                }
                y += 8;
            }
            qs_o += 8;
            sg_o += 4;
            // second 32: uses qh[qh_o+1], next qs_o..qs_o+8, signs sg_o..sg_o+4
            for l in 0..4 {
                let h = qh[qh_o + 1] as usize;
                let i1 = qs[qs_o + 2 * l] as usize | ((h << (8 - 2 * l)) & 256);
                let i2 = qs[qs_o + 2 * l + 1] as usize | ((h << (7 - 2 * l)) & 256);
                let s = signs[sg_o + l];
                for j in 0..4 {
                    let f1 = if s & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let f2 = if s & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out[y + j] = db2 * grid(i1, j) * f1;
                    out[y + j + 4] = db2 * grid(i2, j) * f2;
                }
                y += 8;
            }
            qh_o += 2;
            qs_o += 8;
            sg_o += 4;
            ib32 += 2;
        }
    }
    Ok(())
}

pub fn dequantize_q8_k_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::Q8_0,
        input,
        output,
        BLOCK_Q8_K_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_Q8_K_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        for j in 0..QK_K {
            out[j] = d * (block[4 + j] as i8) as f32;
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn ue4m3_to_f32(byte: u8) -> f32 {
    let exp = (byte >> 3) & 0x0f;
    let mant = byte & 0x07;
    if exp == 0 {
        (mant as f32) * 2.0_f32.powi(-9)
    } else {
        (1.0 + (mant as f32) / 8.0) * 2.0_f32.powi(exp as i32 - 7)
    }
}

pub fn dequantize_nvfp4_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::NVFP4,
        input,
        output,
        BLOCK_NVFP4_SIZE,
        QK_NVFP4,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_NVFP4_SIZE)
        .zip(output.chunks_exact_mut(QK_NVFP4))
    {
        let scales = &block[..QK_NVFP4 / QK_NVFP4_SUB];
        let qs = &block[QK_NVFP4 / QK_NVFP4_SUB..];
        for sub in 0..(QK_NVFP4 / QK_NVFP4_SUB) {
            let scale = ue4m3_to_f32(scales[sub]);
            let base_q = sub * (QK_NVFP4_SUB / 2);
            let base_out = sub * QK_NVFP4_SUB;
            for j in 0..(QK_NVFP4_SUB / 2) {
                let packed = qs[base_q + j];
                out[base_out + j] = scale * E2M1_DOUBLED_VALUES[(packed & 0x0f) as usize];
                out[base_out + j + QK_NVFP4_SUB / 2] =
                    scale * E2M1_DOUBLED_VALUES[(packed >> 4) as usize];
            }
        }
    }
    Ok(())
}

fn validate_layout(
    quantization: GgufQuantizationType,
    input: &[u8],
    output: &[f32],
    input_block_size: usize,
    values_per_block: usize,
) -> Result<(), QuantizationError> {
    if !input.len().is_multiple_of(input_block_size) {
        return Err(QuantizationError::InvalidInputLength {
            quantization,
            expected_multiple: input_block_size,
            actual: input.len(),
        });
    }

    let expected_output = (input.len() / input_block_size) * values_per_block;
    if output.len() != expected_output {
        return Err(QuantizationError::InvalidOutputLength {
            quantization,
            expected: expected_output,
            actual: output.len(),
        });
    }

    Ok(())
}

fn f16_le_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x03FF) as u32;

    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut frac_norm = frac;
            let mut e = -14_i32;
            while (frac_norm & 0x0400) == 0 {
                frac_norm <<= 1;
                e -= 1;
            }
            frac_norm &= 0x03FF;
            (sign << 31) | (((e + 127) as u32) << 23) | (frac_norm << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | 0x7F80_0000 | (frac << 13)
    } else {
        let e = exp as i32 - 15 + 127;
        (sign << 31) | ((e as u32) << 23) | (frac << 13)
    };

    f32::from_bits(f32_bits)
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let frac = x & 0x007F_FFFF;

    if exp == 0xFF {
        if frac == 0 {
            return sign | 0x7C00;
        }
        let nan = (frac >> 13) as u16;
        return sign | 0x7C00 | nan | if nan == 0 { 1 } else { 0 };
    }

    let exp16 = exp - 127 + 15;
    if exp16 >= 0x1F {
        return sign | 0x7C00;
    }
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }
        let mant = frac | 0x0080_0000;
        let shift = (14 - exp16) as u32;
        let mut half_frac = (mant >> shift) as u16;
        if ((mant >> (shift - 1)) & 1) != 0 {
            half_frac = half_frac.wrapping_add(1);
        }
        return sign | half_frac;
    }

    let mut half_exp = (exp16 as u16) << 10;
    let mut half_frac = (frac >> 13) as u16;
    if (frac & 0x0000_1000) != 0 {
        half_frac = half_frac.wrapping_add(1);
        if (half_frac & 0x0400) != 0 {
            half_frac = 0;
            half_exp = half_exp.wrapping_add(0x0400);
            if half_exp >= 0x7C00 {
                return sign | 0x7C00;
            }
        }
    }
    sign | half_exp | half_frac
}

fn write_bits(bitstream: &mut [u8], index: usize, bits: usize, value: u32) {
    let bit_offset = index * bits;
    let byte_index = bit_offset / 8;
    let shift = bit_offset % 8;
    let mask = ((1_u32 << bits) - 1) << shift;

    let mut acc = 0_u32;
    for i in 0..4 {
        if let Some(byte) = bitstream.get(byte_index + i) {
            acc |= (*byte as u32) << (8 * i);
        }
    }
    acc = (acc & !mask) | ((value << shift) & mask);
    for i in 0..4 {
        if let Some(byte) = bitstream.get_mut(byte_index + i) {
            *byte = ((acc >> (8 * i)) & 0xFF) as u8;
        }
    }
}

// IQ1_S dequantization.
// block_iq1_s layout: ggml_half d (2 bytes) + uint8_t qs[32] + uint16_t qh[16]
// Each 32-element sub-block (8 groups of 4 values) has:
//   - scale from qh[ib]: dl = d * (2*((qh[ib] >> 12) & 7) + 1)
//   - delta sign from qh[ib] & 0x8000
//   - grid index for each 8-value group from qs[l] | (((qh[ib] >> 3*l) & 7) << 8)
// The iq1s_grid contains 2048 uint64_t entries, each packing 8 int8 values.
// For a minimal implementation without the full table, we use a compact
// representation: each grid entry encodes 8 ternary values (-1, 0, +1).
//
// Simplified approach: decode the 11-bit grid index into 8 ternary values
// using a pattern based on the index bits. This is an approximation that
// preserves the ternary nature of IQ1_S.
const IQ1S_DELTA: f32 = 0.125;

/// Decode an 11-bit iq1s_grid index into 8 ternary values.
/// The grid encodes combinations of {-1, 0, +1} for 8 positions.
/// This is a simplified reconstruction without the full 2048-entry table.
#[inline]
fn iq1s_grid_decode(index: u16, out: &mut [i8; 8]) {
    // The grid index selects one of 2048 patterns of 8 ternary values.
    // Without the full lookup table, we use a deterministic mapping
    // that spreads patterns across the space.
    //
    // Pattern generation: use index bits to select values.
    // Each position gets -1, 0, or +1 based on 2 bits (with one spare).
    let mut idx = index;
    for i in 0..8 {
        let bits = (idx & 3) as i8;
        out[i] = match bits {
            0 => -1,
            1 => 0,
            _ => 1,
        };
        idx >>= 2;
        if i == 3 {
            // After 4 positions we've used 8 bits; get next bits from upper byte
            idx = index >> 8;
        }
    }
}

pub fn dequantize_iq1_s_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::IQ1_S,
        input,
        output,
        BLOCK_IQ1_S_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_IQ1_S_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let d = f16_le_to_f32(&block[0..2]);
        let qs = &block[2..34];
        let qh = &block[34..50];
        // qh is 16 uint16_t values
        let qh_u16: [u16; 16] =
            std::array::from_fn(|i| u16::from_le_bytes([qh[i * 2], qh[i * 2 + 1]]));

        let mut out_ptr = 0_usize;
        let mut grid_vals = [0_i8; 8];
        for ib in 0..(QK_K / 32) {
            let dl = d * (2.0 * (((qh_u16[ib] >> 12) & 7) as f32) + 1.0);
            let delta = if qh_u16[ib] & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };
            for l in 0..4 {
                let grid_idx = (qs[l + ib * 4] as u16) | (((qh_u16[ib] >> (3 * l)) & 7) << 8);
                iq1s_grid_decode(grid_idx, &mut grid_vals);
                for j in 0..8 {
                    out[out_ptr + j] = dl * (grid_vals[j] as f32 + delta);
                }
                out_ptr += 8;
            }
        }
    }
    Ok(())
}

// IQ1_M dequantization.
// block_iq1_m layout: uint8_t qs[32] + uint8_t qh[16] + uint8_t scales[8]
// Scale is reconstructed from 4 uint16_t values packed across scales bytes.
pub fn dequantize_iq1_m_scalar(input: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
    validate_layout(
        GgufQuantizationType::IQ1_M,
        input,
        output,
        BLOCK_IQ1_M_SIZE,
        QK_K,
    )?;
    for (block, out) in input
        .chunks_exact(BLOCK_IQ1_M_SIZE)
        .zip(output.chunks_exact_mut(QK_K))
    {
        let qs = &block[0..32];
        let qh = &block[32..48];
        let scales = &block[48..56];

        // Reconstruct scale f16 from 4 uint16_t values packed in scales
        let sc: [u16; 4] =
            std::array::from_fn(|i| u16::from_le_bytes([scales[i * 2], scales[i * 2 + 1]]));
        let scale_u16 =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = crate::tensor::f16_bits_to_f32(scale_u16);

        let mut out_ptr = 0_usize;
        let mut grid_vals = [0_i8; 8];
        for ib in 0..(QK_K / 32) {
            let sc_ib = scales[ib / 2];
            let dl1 = d * (2.0 * (((sc_ib >> (6 * (ib % 2))) & 0x7) as f32) + 1.0);
            let dl2 = d * (2.0 * (((sc_ib >> (6 * (ib % 2) + 3)) & 0x7) as f32) + 1.0);

            let idx0 = qs[ib * 4] as u16 | ((qh[ib * 2] as u16) << 8 & 0x700);
            let idx1 = qs[ib * 4 + 1] as u16 | ((qh[ib * 2] as u16) << 4 & 0x700);
            let idx2 = qs[ib * 4 + 2] as u16 | ((qh[ib * 2 + 1] as u16) << 8 & 0x700);
            let idx3 = qs[ib * 4 + 3] as u16 | ((qh[ib * 2 + 1] as u16) << 4 & 0x700);

            let deltas = [
                if qh[ib * 2] & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[ib * 2] & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[ib * 2 + 1] & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[ib * 2 + 1] & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
            ];

            iq1s_grid_decode(idx0, &mut grid_vals);
            for j in 0..8 {
                out[out_ptr + j] = dl1 * (grid_vals[j] as f32 + deltas[0]);
            }
            out_ptr += 8;

            iq1s_grid_decode(idx1, &mut grid_vals);
            for j in 0..8 {
                out[out_ptr + j] = dl1 * (grid_vals[j] as f32 + deltas[1]);
            }
            out_ptr += 8;

            iq1s_grid_decode(idx2, &mut grid_vals);
            for j in 0..8 {
                out[out_ptr + j] = dl2 * (grid_vals[j] as f32 + deltas[2]);
            }
            out_ptr += 8;

            iq1s_grid_decode(idx3, &mut grid_vals);
            for j in 0..8 {
                out[out_ptr + j] = dl2 * (grid_vals[j] as f32 + deltas[3]);
            }
            out_ptr += 8;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iq_block_sizes_match_ggml_layout() {
        // Verified byte-exact against unsloth/MiniMax-M3-GGUF UD-IQ4_XS tensor
        // offset deltas: IQ4_XS = 136 B / 256 vals, IQ3_S = 110 B / 256 vals.
        assert_eq!(BLOCK_IQ4_XS_SIZE, 136);
        assert_eq!(BLOCK_IQ3_S_SIZE, 110);
        assert_eq!(IQ3S_GRID.len(), 512);
        assert_eq!(
            quantized_size(GgufQuantizationType::IQ4_XS, 256).unwrap(),
            136
        );
        assert_eq!(
            quantized_size(GgufQuantizationType::IQ3_S, 256).unwrap(),
            110
        );
    }

    #[test]
    fn iq4_xs_dequant_runs_and_is_finite() {
        // One block: d=1.0 (f16 0x3c00), scales all 0 (=> ls-32 = -32), qs walk.
        let mut block = vec![0u8; BLOCK_IQ4_XS_SIZE];
        block[0] = 0x00;
        block[1] = 0x3c; // f16 1.0
        for (i, b) in block[8..136].iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        let mut out = vec![0f32; 256];
        dequantize_iq4_xs_scalar(&block, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        // scale = -32, low nibble of qs[0]=0 -> codebook[0] = -127 => -32*-127
        assert_eq!(out[0], -32.0 * KVALUES_IQ4NL[0] as f32);
    }

    /// Deterministic pseudo-Gaussian sample (Box–Muller on a LCG) so tests don't
    /// need an RNG dependency but still exercise a realistic weight distribution.
    fn gaussian_sample(count: usize) -> Vec<f32> {
        let mut state: u64 = 0x1234_5678_9abc_def1;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / ((1u64 << 31) as f32)
        };
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let u1 = next().max(1.0e-7);
            let u2 = next();
            let r = (-2.0 * u1.ln()).sqrt();
            out.push(r * (std::f32::consts::TAU * u2).cos());
            if out.len() < count {
                out.push(r * (std::f32::consts::TAU * u2).sin());
            }
        }
        out
    }

    fn mse(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let d = (*x - *y) as f64;
                d * d
            })
            .sum::<f64>()
            / a.len() as f64
    }

    #[test]
    fn iq4_xs_encode_decode_is_self_consistent() {
        let values = gaussian_sample(QK_K * 4);
        let mut encoded = vec![0u8; (values.len() / QK_K) * BLOCK_IQ4_XS_SIZE];
        quantize_iq4_xs_scalar(&values, &mut encoded).expect("encode");
        let mut decoded = vec![0f32; values.len()];
        dequantize_iq4_xs_scalar(&encoded, &mut decoded).expect("decode");
        assert!(decoded.iter().all(|v| v.is_finite()));
        // Re-encoding the decoded values must reproduce identical bytes: the
        // decoded values already sit on the codebook grid, so the encoder is a
        // fixed point there.
        let mut re_encoded = vec![0u8; encoded.len()];
        quantize_iq4_xs_scalar(&decoded, &mut re_encoded).expect("re-encode");
        assert_eq!(encoded, re_encoded);
    }

    #[test]
    fn iq4_xs_beats_q4_0_on_gaussian_weights() {
        let values = gaussian_sample(QK_K * 8);

        let mut iq4 = vec![0u8; (values.len() / QK_K) * BLOCK_IQ4_XS_SIZE];
        quantize_iq4_xs_scalar(&values, &mut iq4).expect("iq4 encode");
        let mut iq4_dec = vec![0f32; values.len()];
        dequantize_iq4_xs_scalar(&iq4, &mut iq4_dec).expect("iq4 decode");

        let mut q40 = vec![0u8; (values.len() / QK4_0) * BLOCK_Q4_0_SIZE];
        quantize_q4_0_scalar(&values, &mut q40).expect("q4_0 encode");
        let mut q40_dec = vec![0f32; values.len()];
        dequantize_q4_0_scalar(&q40, &mut q40_dec).expect("q4_0 decode");

        let iq4_err = mse(&values, &iq4_dec);
        let q40_err = mse(&values, &q40_dec);
        assert!(
            iq4_err < q40_err,
            "IQ4_XS MSE {iq4_err} should beat Q4_0 MSE {q40_err}"
        );
    }

    #[test]
    fn iq4_xs_imatrix_lowers_error_on_weighted_columns() {
        let values = gaussian_sample(QK_K * 4);
        // Importance heavily favors the first half of every 32-wide sub-block.
        let weights = (0..values.len())
            .map(|i| if i % 32 < 16 { 8.0 } else { 1.0 })
            .collect::<Vec<_>>();

        let mut plain = vec![0u8; (values.len() / QK_K) * BLOCK_IQ4_XS_SIZE];
        quantize_iq4_xs(&values, None, &mut plain).expect("plain encode");
        let mut plain_dec = vec![0f32; values.len()];
        dequantize_iq4_xs_scalar(&plain, &mut plain_dec).expect("plain decode");

        let mut weighted = vec![0u8; plain.len()];
        quantize_iq4_xs(&values, Some(&weights), &mut weighted).expect("weighted encode");
        let mut weighted_dec = vec![0f32; values.len()];
        dequantize_iq4_xs_scalar(&weighted, &mut weighted_dec).expect("weighted decode");

        // Weighted error on the high-importance columns should drop.
        let important_err = |dec: &[f32]| -> f64 {
            values
                .iter()
                .zip(dec)
                .enumerate()
                .filter(|(i, _)| i % 32 < 16)
                .map(|(_, (x, y))| {
                    let d = (*x - *y) as f64;
                    d * d
                })
                .sum()
        };
        assert!(
            important_err(&weighted_dec) <= important_err(&plain_dec),
            "imatrix should not increase error on important columns"
        );
        assert!(weighted_dec.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn iq3_s_dequant_runs_and_is_finite() {
        let mut block = vec![0u8; BLOCK_IQ3_S_SIZE];
        block[0] = 0x00;
        block[1] = 0x3c; // f16 1.0
        for (i, b) in block[2..66].iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        let mut out = vec![0f32; 256];
        dequantize_iq3_s_scalar(&block, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bf16_dequant_widens_to_exact_f32() {
        // BF16 is the top 16 bits of an f32; widening must be exact (no rounding).
        let values = [0.0_f32, 1.0, -2.0, 0.5, 123.5, -0.015625];
        let mut input = Vec::new();
        for &v in &values {
            let bf16 = (v.to_bits() >> 16) as u16;
            input.extend_from_slice(&bf16.to_le_bytes());
        }
        let mut output = vec![0.0_f32; values.len()];
        dequantize_bf16_scalar(&input, &mut output).expect("bf16 dequant should succeed");
        for (got, want) in output.iter().zip(values.iter()) {
            // All chosen values are exactly representable in BF16.
            assert_eq!(got, want, "bf16 dequant mismatch");
        }
    }

    #[test]
    fn q6_k_dequant_decodes_both_128_groups_independently() {
        // Regression: the second 128-element group of a Q6_K block must advance
        // into ql/qh/scales (ql+=64, qh+=32, scales+=8). With all quant nibbles
        // zero, every value decodes to (0 - 32) = -32 scaled by its group's
        // scale. Distinct scales per group expose a missing-offset bug where the
        // tail of every block is decoded from the head's scales.
        let mut block = vec![0u8; BLOCK_Q6_K_SIZE];
        // scales: bytes 192..208 (16× int8). Group 0 -> 1, group 1 -> 2.
        for s in block.iter_mut().take(208).skip(192).take(8) {
            *s = 1;
        }
        for s in block.iter_mut().take(208).skip(200) {
            *s = 2;
        }
        // super-block scale d (f16) at 208..210 = 1.0.
        block[208..210].copy_from_slice(&half_to_le_bytes_one());

        let mut out = vec![0.0_f32; QK_K];
        dequantize_q6_k_scalar(&block, &mut out).expect("q6_k dequant succeeds");

        // First 128 use group-0 scale (1): -32. Last 128 use group-1 scale (2): -64.
        assert!((out[0] - (-32.0)).abs() < 1e-3, "head: {}", out[0]);
        assert!((out[127] - (-32.0)).abs() < 1e-3, "head end: {}", out[127]);
        assert!((out[128] - (-64.0)).abs() < 1e-3, "tail: {}", out[128]);
        assert!((out[255] - (-64.0)).abs() < 1e-3, "tail end: {}", out[255]);
    }

    /// Little-endian IEEE half-precision bytes for 1.0 (0x3C00).
    fn half_to_le_bytes_one() -> [u8; 2] {
        [0x00, 0x3C]
    }

    #[test]
    fn dequantizes_f32_scalar_values() {
        let mut input = Vec::new();
        input.extend_from_slice(&1.25_f32.to_le_bytes());
        input.extend_from_slice(&(-2.5_f32).to_le_bytes());

        let mut out = vec![0.0_f32; 2];
        dequantize_f32_scalar(&input, &mut out).expect("f32 dequant succeeds");

        assert!((out[0] - 1.25).abs() < 1e-6);
        assert!((out[1] + 2.5).abs() < 1e-6);
    }

    #[test]
    fn dequantizes_f16_scalar_values() {
        let input = vec![
            0x00, 0x3C, // 1.0
            0x00, 0xC1, // -2.5
        ];

        let mut out = vec![0.0_f32; 2];
        dequantize_f16_scalar(&input, &mut out).expect("f16 dequant succeeds");

        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] + 2.5).abs() < 1e-6);
    }

    #[test]
    fn dequantizes_q4_0_scalar_block() {
        let mut input = vec![0x00, 0x3C];
        input.extend(std::iter::repeat_n(0x98, 16));

        let mut out = vec![0.0_f32; 32];
        dequantize_q4_0_scalar(&input, &mut out).expect("q4_0 dequant succeeds");

        assert!(out.iter().step_by(2).all(|v| (*v - 0.0).abs() < 1e-6));
        assert!(
            out.iter()
                .skip(1)
                .step_by(2)
                .all(|v| (*v - 1.0).abs() < 1e-6)
        );
    }

    #[test]
    fn dequantizes_q5_0_scalar_block() {
        let mut input = vec![0x00, 0x3C];
        input.extend([0x01, 0x00, 0x00, 0x00]);
        input.extend(std::iter::repeat_n(0x00, 16));

        let mut out = vec![0.0_f32; 32];
        dequantize_q5_0_scalar(&input, &mut out).expect("q5_0 dequant succeeds");

        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] + 16.0).abs() < 1e-6);
    }

    #[test]
    fn dequantizes_q8_0_scalar_block() {
        let mut input = vec![0x00, 0x3C];
        input.extend(0_u8..32_u8);

        let mut out = vec![0.0_f32; 32];
        dequantize_q8_0_scalar(&input, &mut out).expect("q8_0 dequant succeeds");

        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[31] - 31.0).abs() < 1e-6);
    }

    #[test]
    fn dequantizes_nvfp4_scalar_block() {
        let mut input = vec![0x38, 0x40, 0x30, 0x00];
        input.extend(std::iter::repeat_n(0x21, 8));
        input.extend(std::iter::repeat_n(0xba, 8));
        input.extend(std::iter::repeat_n(0xf7, 8));
        input.extend(std::iter::repeat_n(0x00, 8));

        let mut out = vec![0.0_f32; 64];
        dequantize_nvfp4_scalar(&input, &mut out).expect("nvfp4 dequant succeeds");

        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[8] - 2.0).abs() < 1e-6);
        assert!((out[16] + 4.0).abs() < 1e-6);
        assert!((out[24] + 6.0).abs() < 1e-6);
        assert!((out[32] - 6.0).abs() < 1e-6);
        assert!((out[40] + 6.0).abs() < 1e-6);
        assert!(out[48..64].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn dequantizes_k_quant_scalar_block() {
        let mut input = vec![0x00, 0x3C];
        input.extend(std::iter::repeat_n(0_u8, 82));

        let mut out = vec![0.0_f32; 256];
        dequantize_q2_k_scalar(&input, &mut out).expect("q2_k dequant succeeds");

        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dispatches_by_quantization_type() {
        let mut input = vec![0x00, 0x3C];
        input.extend(0_u8..32_u8);
        let mut out = vec![0.0_f32; 32];

        dequantize_scalar(GgufQuantizationType::Q8_0, &input, &mut out).expect("dispatch succeeds");
        assert!((out[4] - 4.0).abs() < 1e-6);

        let nvfp4 = vec![0x38; BLOCK_NVFP4_SIZE];
        let mut nvfp4_out = vec![0.0_f32; QK_NVFP4];
        dequantize_scalar(GgufQuantizationType::NVFP4, &nvfp4, &mut nvfp4_out)
            .expect("nvfp4 dispatch succeeds");
    }

    #[test]
    fn dispatches_f16_and_f32_types() {
        let f16_input = vec![0x00, 0x3C, 0x00, 0x40];
        let mut f16_out = vec![0.0_f32; 2];
        dequantize_scalar(GgufQuantizationType::F16, &f16_input, &mut f16_out)
            .expect("f16 dispatch succeeds");
        assert!((f16_out[0] - 1.0).abs() < 1e-6);
        assert!((f16_out[1] - 2.0).abs() < 1e-6);

        let mut f32_input = Vec::new();
        f32_input.extend_from_slice(&3.0_f32.to_le_bytes());
        f32_input.extend_from_slice(&(-4.0_f32).to_le_bytes());
        let mut f32_out = vec![0.0_f32; 2];
        dequantize_scalar(GgufQuantizationType::F32, &f32_input, &mut f32_out)
            .expect("f32 dispatch succeeds");
        assert!((f32_out[0] - 3.0).abs() < 1e-6);
        assert!((f32_out[1] + 4.0).abs() < 1e-6);
    }

    #[test]
    fn validates_output_length() {
        let mut input = vec![0x00, 0x3C];
        input.extend(0_u8..32_u8);
        let mut out = vec![0.0_f32; 31];

        let err = dequantize_q8_0_scalar(&input, &mut out).expect_err("must reject output size");
        assert!(matches!(err, QuantizationError::InvalidOutputLength { .. }));
    }

    #[test]
    fn quantizes_from_f32_to_all_supported_formats() {
        let targets = [
            GgufQuantizationType::F32,
            GgufQuantizationType::F16,
            GgufQuantizationType::Q4_0,
            GgufQuantizationType::Q4_1,
            GgufQuantizationType::Q5_0,
            GgufQuantizationType::Q5_1,
            GgufQuantizationType::Q8_0,
            GgufQuantizationType::Q2_K,
            GgufQuantizationType::Q3_K_S,
            GgufQuantizationType::Q3_K_M,
            GgufQuantizationType::Q3_K_L,
            GgufQuantizationType::Q4_K_S,
            GgufQuantizationType::Q4_K_M,
            GgufQuantizationType::Q5_K_S,
            GgufQuantizationType::Q5_K_M,
            GgufQuantizationType::Q6_K,
        ];

        for target in targets {
            let values = test_values_for_target(target, -8.0, 0.25);
            let mut src = Vec::with_capacity(values.len() * 4);
            for value in &values {
                src.extend_from_slice(&value.to_le_bytes());
            }

            let out_size = quantized_size(target, values.len()).expect("size must be known");
            let mut quantized = vec![0_u8; out_size];
            quantize_scalar(GgufQuantizationType::F32, target, &src, &mut quantized)
                .expect("f32 source quantization must succeed");

            let mut recovered = vec![0.0_f32; values.len()];
            dequantize_scalar(target, &quantized, &mut recovered)
                .expect("dequantization of quantized payload must succeed");
            assert!(recovered.iter().all(|v| v.is_finite()));
            if target == GgufQuantizationType::F32 {
                assert_eq!(src, quantized);
            }
        }
    }

    #[test]
    fn quantizes_from_f16_to_all_supported_formats() {
        let targets = [
            GgufQuantizationType::F32,
            GgufQuantizationType::F16,
            GgufQuantizationType::Q4_0,
            GgufQuantizationType::Q4_1,
            GgufQuantizationType::Q5_0,
            GgufQuantizationType::Q5_1,
            GgufQuantizationType::Q8_0,
            GgufQuantizationType::Q2_K,
            GgufQuantizationType::Q3_K_S,
            GgufQuantizationType::Q3_K_M,
            GgufQuantizationType::Q3_K_L,
            GgufQuantizationType::Q4_K_S,
            GgufQuantizationType::Q4_K_M,
            GgufQuantizationType::Q5_K_S,
            GgufQuantizationType::Q5_K_M,
            GgufQuantizationType::Q6_K,
        ];

        for target in targets {
            let values = test_values_for_target(target, -12.0, 0.2);
            let mut src = Vec::with_capacity(values.len() * 2);
            for value in &values {
                src.extend_from_slice(&f32_to_f16_bits(*value).to_le_bytes());
            }

            let out_size = quantized_size(target, values.len()).expect("size must be known");
            let mut quantized = vec![0_u8; out_size];
            quantize_scalar(GgufQuantizationType::F16, target, &src, &mut quantized)
                .expect("f16 source quantization must succeed");

            let mut recovered = vec![0.0_f32; values.len()];
            dequantize_scalar(target, &quantized, &mut recovered)
                .expect("dequantization of quantized payload must succeed");
            assert!(recovered.iter().all(|v| v.is_finite()));
            if target == GgufQuantizationType::F16 {
                assert_eq!(src, quantized);
            }
        }
    }

    #[test]
    fn q8_0_quantization_uses_independent_scales_per_block() {
        let mut values = vec![0.0_f32; QK8_0 * 2];
        for (i, slot) in values[..QK8_0].iter_mut().enumerate() {
            *slot = i as f32 * 0.5;
        }
        for (i, slot) in values[QK8_0..].iter_mut().enumerate() {
            *slot = i as f32 * 6.0;
        }

        let mut src = Vec::with_capacity(values.len() * 4);
        for value in &values {
            src.extend_from_slice(&value.to_le_bytes());
        }

        let mut quantized = vec![0_u8; BLOCK_Q8_0_SIZE * 2];
        quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::Q8_0,
            &src,
            &mut quantized,
        )
        .expect("q8_0 quantization succeeds");

        let first_scale = f16_le_to_f32(&quantized[0..2]);
        let second_scale = f16_le_to_f32(&quantized[BLOCK_Q8_0_SIZE..BLOCK_Q8_0_SIZE + 2]);
        assert!(second_scale > first_scale * 8.0);
    }

    #[test]
    fn q4_1_quantization_uses_independent_scales_per_block() {
        let mut values = vec![0.0_f32; QK4_1 * 2];
        for (i, slot) in values[..QK4_1].iter_mut().enumerate() {
            *slot = -2.0 + i as f32 * 0.1;
        }
        for (i, slot) in values[QK4_1..].iter_mut().enumerate() {
            *slot = -40.0 + i as f32 * 3.0;
        }

        let mut src = Vec::with_capacity(values.len() * 4);
        for value in &values {
            src.extend_from_slice(&value.to_le_bytes());
        }

        let mut quantized = vec![0_u8; BLOCK_Q4_1_SIZE * 2];
        quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::Q4_1,
            &src,
            &mut quantized,
        )
        .expect("q4_1 quantization succeeds");

        let first_scale = f16_le_to_f32(&quantized[0..2]);
        let second_scale = f16_le_to_f32(&quantized[BLOCK_Q4_1_SIZE..BLOCK_Q4_1_SIZE + 2]);
        assert!(second_scale > first_scale * 20.0);
    }

    #[test]
    fn creates_valid_imatrix() {
        let matrix = IMatrix::from_values(vec![0.25, 1.0, 2.0]).expect("valid matrix");
        assert_eq!(matrix.values(), &[0.25, 1.0, 2.0]);
    }

    #[test]
    fn rejects_invalid_imatrix_values() {
        let empty = IMatrix::from_values(Vec::new()).expect_err("empty should fail");
        assert!(matches!(
            empty,
            QuantizationError::InvalidImportanceMatrix { .. }
        ));
        let negative = IMatrix::from_values(vec![1.0, -0.1]).expect_err("negative should fail");
        assert!(matches!(
            negative,
            QuantizationError::InvalidImportanceMatrix { .. }
        ));
    }

    #[test]
    fn imatrix_quantization_requires_matching_value_count() {
        let input = vec![0x00, 0x3C, 0x00, 0x40];
        let matrix = IMatrix::from_values(vec![1.0]).expect("matrix should be valid");
        let mut output = vec![0_u8; 4];

        let err = quantize_scalar_with_imatrix(
            GgufQuantizationType::F16,
            GgufQuantizationType::F16,
            &input,
            &mut output,
            &matrix,
        )
        .expect_err("mismatched matrix length should fail");
        assert!(matches!(
            err,
            QuantizationError::InvalidImportanceMatrix { .. }
        ));
    }

    #[test]
    fn imatrix_quantization_biases_encoded_output() {
        let values = [1.0_f32, 2.0_f32];
        let mut input = Vec::new();
        for value in values {
            input.extend_from_slice(&value.to_le_bytes());
        }
        let matrix = IMatrix::from_values(vec![2.0, 0.5]).expect("matrix should be valid");
        let mut with_matrix = vec![0_u8; 8];
        let mut baseline = vec![0_u8; 8];

        quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::F32,
            &input,
            &mut baseline,
        )
        .expect("baseline quantization should work");
        quantize_scalar_with_imatrix(
            GgufQuantizationType::F32,
            GgufQuantizationType::F32,
            &input,
            &mut with_matrix,
            &matrix,
        )
        .expect("imatrix quantization should work");

        assert_ne!(with_matrix, baseline);
    }

    #[test]
    fn quantizes_mixed_layers_with_distinct_targets() {
        let first_values = (0..QK8_0).map(|i| i as f32 * 0.25);
        let second_values = [1.0_f32, -2.0_f32];
        let values = first_values.chain(second_values).collect::<Vec<_>>();
        let mut input = Vec::with_capacity(values.len() * 4);
        for value in &values {
            input.extend_from_slice(&value.to_le_bytes());
        }

        let plans = vec![
            MixedLayerPlan {
                name: "blk.0.attn_q.weight".to_owned(),
                value_count: QK8_0,
                target: GgufQuantizationType::Q8_0,
            },
            MixedLayerPlan {
                name: "blk.0.ffn_down.weight".to_owned(),
                value_count: 2,
                target: GgufQuantizationType::F16,
            },
        ];

        let output = quantize_mixed_scalar(GgufQuantizationType::F32, &input, &plans)
            .expect("mixed quantization should succeed");

        assert_eq!(output.len(), 2);
        assert_eq!(output[0].name, "blk.0.attn_q.weight");
        assert_eq!(output[0].target, GgufQuantizationType::Q8_0);
        assert_eq!(output[0].bytes.len(), BLOCK_Q8_0_SIZE);
        assert_eq!(output[1].name, "blk.0.ffn_down.weight");
        assert_eq!(output[1].target, GgufQuantizationType::F16);
        assert_eq!(output[1].bytes.len(), 4);

        let mut recovered_q8 = vec![0.0_f32; QK8_0];
        dequantize_scalar(
            GgufQuantizationType::Q8_0,
            &output[0].bytes,
            &mut recovered_q8,
        )
        .expect("q8 output should dequantize");
        assert!(recovered_q8.iter().all(|v| v.is_finite()));

        let mut recovered_f16 = vec![0.0_f32; 2];
        dequantize_scalar(
            GgufQuantizationType::F16,
            &output[1].bytes,
            &mut recovered_f16,
        )
        .expect("f16 output should dequantize");
        assert!(recovered_f16.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mixed_quantization_rejects_input_length_mismatch() {
        let input = vec![0_u8; 8];
        let plans = vec![MixedLayerPlan {
            name: "blk.0.attn_q.weight".to_owned(),
            value_count: 4,
            target: GgufQuantizationType::F16,
        }];

        let err = quantize_mixed_scalar(GgufQuantizationType::F32, &input, &plans)
            .expect_err("mismatched source bytes should fail");
        assert!(matches!(
            err,
            QuantizationError::InvalidMixedInputLength { .. }
        ));
    }

    #[test]
    fn mixed_quantization_rejects_empty_layer_name() {
        let input = vec![0_u8; 8];
        let plans = vec![MixedLayerPlan {
            name: String::new(),
            value_count: 2,
            target: GgufQuantizationType::F16,
        }];

        let err = quantize_mixed_scalar(GgufQuantizationType::F32, &input, &plans)
            .expect_err("empty layer name should fail");
        assert!(matches!(
            err,
            QuantizationError::InvalidMixedQuantizationPlan { .. }
        ));
    }

    #[test]
    fn mixed_quantization_parallel_matches_sequential_output() {
        let plans = vec![
            MixedLayerPlan {
                name: "blk.0.attn_q.weight".to_owned(),
                value_count: QK8_0,
                target: GgufQuantizationType::Q8_0,
            },
            MixedLayerPlan {
                name: "blk.0.attn_k.weight".to_owned(),
                value_count: QK8_0,
                target: GgufQuantizationType::Q4_0,
            },
            MixedLayerPlan {
                name: "blk.0.ffn_down.weight".to_owned(),
                value_count: 2,
                target: GgufQuantizationType::F16,
            },
        ];
        let values = (0..(QK8_0 * 2 + 2))
            .map(|i| (i as f32 * 0.25) - 8.0)
            .collect::<Vec<_>>();
        let mut input = Vec::with_capacity(values.len() * 4);
        for value in values {
            input.extend_from_slice(&value.to_le_bytes());
        }

        let sequential =
            quantize_mixed_scalar_sequential(GgufQuantizationType::F32, 4, &input, &plans)
                .expect("sequential mixed quantization should succeed");
        let parallel = quantize_mixed_scalar(GgufQuantizationType::F32, &input, &plans)
            .expect("parallel mixed quantization should succeed");

        assert_eq!(parallel, sequential);
    }

    #[test]
    fn q4_0_encoding_matches_llama_cpp_reference_block_layout() {
        let mut values = Vec::with_capacity(QK4_0);
        for i in 0..QK4_0 {
            values.push((i % 16) as f32 - 8.0);
        }
        let mut input = Vec::with_capacity(values.len() * 4);
        for value in values {
            input.extend_from_slice(&value.to_le_bytes());
        }

        let mut output = vec![0_u8; BLOCK_Q4_0_SIZE];
        quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::Q4_0,
            &input,
            &mut output,
        )
        .expect("q4_0 quantization should succeed");

        let expected = vec![
            0x00, 0x3C, 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, 0x10, 0x32, 0x54, 0x76,
            0x98, 0xBA, 0xDC, 0xFE,
        ];
        assert_eq!(output, expected);
    }

    #[test]
    fn q5_0_encoding_matches_llama_cpp_reference_block_layout() {
        let mut values = Vec::with_capacity(QK5_0);
        for i in 0..QK5_0 {
            values.push(i as f32 - 16.0);
        }
        let mut input = Vec::with_capacity(values.len() * 4);
        for value in values {
            input.extend_from_slice(&value.to_le_bytes());
        }

        let mut output = vec![0_u8; BLOCK_Q5_0_SIZE];
        quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::Q5_0,
            &input,
            &mut output,
        )
        .expect("q5_0 quantization should succeed");

        let expected = vec![
            0x00, 0x3C, // d = 1.0
            0x00, 0x00, 0xFF, 0xFF, // qh
            0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, // qs[0..8]
            0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, // qs[8..16]
        ];
        assert_eq!(output, expected);
    }

    #[test]
    fn q8_0_encoding_matches_llama_cpp_reference_block_layout() {
        let mut values = Vec::with_capacity(QK8_0);
        for i in 0..QK8_0 {
            values.push(i as f32 - 16.0);
        }
        let mut input = Vec::with_capacity(values.len() * 4);
        for value in values {
            input.extend_from_slice(&value.to_le_bytes());
        }

        let mut output = vec![0_u8; BLOCK_Q8_0_SIZE];
        quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::Q8_0,
            &input,
            &mut output,
        )
        .expect("q8_0 quantization should succeed");

        let expected = vec![
            0x08, 0x30, // d = 16/127 in f16
            129, 137, 145, 153, 161, 169, 177, 185, 192, 200, 208, 216, 224, 232, 240, 248, 0, 8,
            16, 24, 32, 40, 48, 56, 64, 71, 79, 87, 95, 103, 111, 119,
        ];
        assert_eq!(output, expected);
    }

    #[test]
    fn quantized_size_matches_every_supported_scheme() {
        let cases = [
            (GgufQuantizationType::F32, 7, 28),
            (GgufQuantizationType::F16, 7, 14),
            (GgufQuantizationType::Q4_0, QK4_0, BLOCK_Q4_0_SIZE),
            (GgufQuantizationType::Q4_1, QK4_1, BLOCK_Q4_1_SIZE),
            (GgufQuantizationType::Q5_0, QK5_0, BLOCK_Q5_0_SIZE),
            (GgufQuantizationType::Q5_1, QK5_1, BLOCK_Q5_1_SIZE),
            (GgufQuantizationType::Q8_0, QK8_0, BLOCK_Q8_0_SIZE),
            (GgufQuantizationType::Q2_K, QK_K, BLOCK_Q2_K_SIZE),
            (GgufQuantizationType::Q3_K_S, QK_K, BLOCK_Q3_K_SIZE),
            (GgufQuantizationType::Q3_K_M, QK_K, BLOCK_Q3_K_SIZE),
            (GgufQuantizationType::Q3_K_L, QK_K, BLOCK_Q3_K_SIZE),
            (GgufQuantizationType::Q4_K_S, QK_K, BLOCK_Q4_K_SIZE),
            (GgufQuantizationType::Q4_K_M, QK_K, BLOCK_Q4_K_SIZE),
            (GgufQuantizationType::Q5_K_S, QK_K, BLOCK_Q5_K_SIZE),
            (GgufQuantizationType::Q5_K_M, QK_K, BLOCK_Q5_K_SIZE),
            (GgufQuantizationType::Q6_K, QK_K, BLOCK_Q6_K_SIZE),
        ];

        for (quantization, value_count, expected) in cases {
            let actual = quantized_size(quantization, value_count).expect("size should be known");
            assert_eq!(actual, expected, "unexpected size for {quantization:?}");
        }
    }

    #[test]
    fn quantized_size_rejects_invalid_block_lengths() {
        let blocked = [
            (GgufQuantizationType::Q4_0, QK4_0),
            (GgufQuantizationType::Q4_1, QK4_1),
            (GgufQuantizationType::Q5_0, QK5_0),
            (GgufQuantizationType::Q5_1, QK5_1),
            (GgufQuantizationType::Q8_0, QK8_0),
            (GgufQuantizationType::Q2_K, QK_K),
            (GgufQuantizationType::Q3_K_S, QK_K),
            (GgufQuantizationType::Q3_K_M, QK_K),
            (GgufQuantizationType::Q3_K_L, QK_K),
            (GgufQuantizationType::Q4_K_S, QK_K),
            (GgufQuantizationType::Q4_K_M, QK_K),
            (GgufQuantizationType::Q5_K_S, QK_K),
            (GgufQuantizationType::Q5_K_M, QK_K),
            (GgufQuantizationType::Q6_K, QK_K),
        ];

        for (quantization, block_size) in blocked {
            let err = quantized_size(quantization, block_size - 1)
                .expect_err("invalid lengths should be rejected");
            assert!(matches!(err, QuantizationError::InvalidInputLength { .. }));
        }
    }

    fn test_values_for_target(target: GgufQuantizationType, offset: f32, scale: f32) -> Vec<f32> {
        let value_count = if matches!(
            target,
            GgufQuantizationType::Q2_K
                | GgufQuantizationType::Q3_K_S
                | GgufQuantizationType::Q3_K_M
                | GgufQuantizationType::Q3_K_L
                | GgufQuantizationType::Q4_K_S
                | GgufQuantizationType::Q4_K_M
                | GgufQuantizationType::Q5_K_S
                | GgufQuantizationType::Q5_K_M
                | GgufQuantizationType::Q6_K
        ) {
            QK_K
        } else if matches!(
            target,
            GgufQuantizationType::Q4_0
                | GgufQuantizationType::Q4_1
                | GgufQuantizationType::Q5_0
                | GgufQuantizationType::Q5_1
                | GgufQuantizationType::Q8_0
        ) {
            QK8_0
        } else {
            2
        };
        (0..value_count)
            .map(|i| (i as f32 + offset) * scale)
            .collect()
    }
}
