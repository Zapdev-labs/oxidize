#![allow(clippy::manual_checked_ops, clippy::needless_range_loop)]

use crate::gguf::GgufQuantizationType;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK4_NL: usize = 32;
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
const BLOCK_IQ2_XXS_SIZE: usize = sizeof_of_f16() + QK_K / 4;
const BLOCK_IQ2_XS_SIZE: usize = sizeof_of_f16() + QK_K / 8 * sizeof_of_i16() + QK_K / 32;
const BLOCK_IQ2_S_SIZE: usize = sizeof_of_f16() + QK_K / 4 + QK_K / 32 + QK_K / 32;
const BLOCK_IQ3_XXS_SIZE: usize = sizeof_of_f16() + 3 * (QK_K / 8);
pub const BLOCK_IQ4_NL_SIZE: usize = sizeof_of_f16() + QK4_NL / 2;
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

pub fn quant_block_layout(
    quantization: GgufQuantizationType,
) -> Result<(usize, usize), QuantizationError> {
    match quantization {
        GgufQuantizationType::F32 => Ok((1, 4)),
        GgufQuantizationType::F16 => Ok((1, 2)),
        GgufQuantizationType::I8 => Ok((1, 1)),
        GgufQuantizationType::I16 => Ok((1, 2)),
        GgufQuantizationType::I32 => Ok((1, 4)),
        GgufQuantizationType::I64 => Ok((1, 8)),
        GgufQuantizationType::F64 => Ok((1, 8)),
        GgufQuantizationType::BF16 => Ok((1, 2)),
        GgufQuantizationType::Q4_0 => Ok((QK4_0, BLOCK_Q4_0_SIZE)),
        GgufQuantizationType::Q4_O => Ok((QK4_0, BLOCK_Q4_0_SIZE)),
        GgufQuantizationType::Q4_1 => Ok((QK4_1, BLOCK_Q4_1_SIZE)),
        GgufQuantizationType::Q5_0 => Ok((QK5_0, BLOCK_Q5_0_SIZE)),
        GgufQuantizationType::Q5_1 => Ok((QK5_1, BLOCK_Q5_1_SIZE)),
        GgufQuantizationType::Q8_0 => Ok((QK8_0, BLOCK_Q8_0_SIZE)),
        GgufQuantizationType::Q2_K => Ok((QK_K, BLOCK_Q2_K_SIZE)),
        GgufQuantizationType::Q3_K_S
        | GgufQuantizationType::Q3_K_M
        | GgufQuantizationType::Q3_K_L => Ok((QK_K, BLOCK_Q3_K_SIZE)),
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => Ok((QK_K, BLOCK_Q4_K_SIZE)),
        GgufQuantizationType::Q5_K_S | GgufQuantizationType::Q5_K_M => Ok((QK_K, BLOCK_Q5_K_SIZE)),
        GgufQuantizationType::Q6_K => Ok((QK_K, BLOCK_Q6_K_SIZE)),
        GgufQuantizationType::IQ1_S => Ok((QK_K, BLOCK_IQ1_S_SIZE)),
        GgufQuantizationType::IQ1_M => Ok((QK_K, BLOCK_IQ1_M_SIZE)),
        GgufQuantizationType::NVFP4 => Ok((QK_NVFP4, BLOCK_NVFP4_SIZE)),
        GgufQuantizationType::IQ2_XXS => Ok((QK_K, BLOCK_IQ2_XXS_SIZE)),
        GgufQuantizationType::IQ2_XS => Ok((QK_K, BLOCK_IQ2_XS_SIZE)),
        GgufQuantizationType::IQ2_S => Ok((QK_K, BLOCK_IQ2_S_SIZE)),
        GgufQuantizationType::IQ3_S => Ok((QK_K, BLOCK_IQ3_S_SIZE)),
        GgufQuantizationType::IQ4_XS => Ok((QK_K, BLOCK_IQ4_XS_SIZE)),
        GgufQuantizationType::IQ3_XXS => Ok((QK_K, BLOCK_IQ3_XXS_SIZE)),
        GgufQuantizationType::IQ4_NL => Ok((QK4_NL, BLOCK_IQ4_NL_SIZE)),
        other => Err(QuantizationError::UnsupportedQuantizationType(other)),
    }
}

pub fn quantized_size(
    quantization: GgufQuantizationType,
    value_count: usize,
) -> Result<usize, QuantizationError> {
    let (values_per_block, bytes_per_block) = quant_block_layout(quantization)?;

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

#[path = "quantization/quant_utils.rs"]
mod quant_utils;
use quant_utils::*;

#[path = "quantization/quant_simple.rs"]
mod quant_simple;
pub use quant_simple::*;

#[path = "quantization/quant_k_blocks.rs"]
mod quant_k_blocks;
pub use quant_k_blocks::*;

#[path = "quantization/quant_nvfp4.rs"]
mod quant_nvfp4;
pub use quant_nvfp4::*;

#[path = "quantization/iq_grids.rs"]
pub(crate) mod iq_grids;
pub(crate) use iq_grids::iq1s_grid_decode;

#[path = "quantization/quant_iq_series.rs"]
mod quant_iq_series;
pub use quant_iq_series::*;

#[path = "quantization/quant_dispatch.rs"]
mod quant_dispatch;
pub use quant_dispatch::*;

#[cfg(test)]
#[path = "quantization/tests.rs"]
mod tests;
