use super::*;

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
