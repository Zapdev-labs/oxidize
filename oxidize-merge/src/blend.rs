/// Element-wise linear interpolation: `(1 - t) * a + t * b`.
pub fn linear_f32(a: &[f32], b: &[f32], t: f32, out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let one_minus_t = 1.0 - t;
    for ((o, &left), &right) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = left.mul_add(one_minus_t, right * t);
    }
}

/// Spherical linear interpolation treating `a` and `b` as one vector.
pub fn slerp_f32(a: &[f32], b: &[f32], t: f32, out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    if a.is_empty() {
        return;
    }

    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (&left, &right) in a.iter().zip(b.iter()) {
        let left = f64::from(left);
        let right = f64::from(right);
        dot += left * right;
        norm_a += left * left;
        norm_b += right * right;
    }

    if norm_a == 0.0 && norm_b == 0.0 {
        out.fill(0.0);
        return;
    }
    if norm_a == 0.0 {
        out.copy_from_slice(b);
        return;
    }
    if norm_b == 0.0 {
        out.copy_from_slice(a);
        return;
    }

    let cos_theta = (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    if theta < 1e-8 {
        linear_f32(a, b, t, out);
        return;
    }

    let sin_theta = theta.sin();
    // Near-antipodal inputs: theta → π, sin_theta → 0, so the slerp weight
    // division blows up to NaN/Inf. The great-circle direction is undefined
    // there, so fall back to a stable linear blend.
    if sin_theta < 1e-8 {
        linear_f32(a, b, t, out);
        return;
    }
    let w0 = ((1.0 - f64::from(t)) * theta).sin() / sin_theta;
    let w1 = (f64::from(t) * theta).sin() / sin_theta;
    for ((o, &left), &right) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = (w0 * f64::from(left) + w1 * f64::from(right)) as f32;
    }
}

pub fn linear_bytes(
    dtype: safetensors::tensor::Dtype,
    a: &[u8],
    b: &[u8],
    t: f32,
    out: &mut [u8],
) -> anyhow::Result<()> {
    match dtype {
        safetensors::tensor::Dtype::F32 => {
            blend_slice(a, b, t, out, linear_f32)?;
        }
        safetensors::tensor::Dtype::F16 => {
            blend_slice_f16(a, b, t, out, linear_f32)?;
        }
        safetensors::tensor::Dtype::BF16 => {
            blend_slice_bf16(a, b, t, out, linear_f32)?;
        }
        other => anyhow::bail!("linear blend does not support dtype {other:?}"),
    }
    Ok(())
}

pub fn slerp_bytes(
    dtype: safetensors::tensor::Dtype,
    a: &[u8],
    b: &[u8],
    t: f32,
    out: &mut [u8],
) -> anyhow::Result<()> {
    match dtype {
        safetensors::tensor::Dtype::F32 => {
            blend_slice(a, b, t, out, slerp_f32)?;
        }
        safetensors::tensor::Dtype::F16 => {
            blend_slice_f16(a, b, t, out, slerp_f32)?;
        }
        safetensors::tensor::Dtype::BF16 => {
            blend_slice_bf16(a, b, t, out, slerp_f32)?;
        }
        other => anyhow::bail!("slerp blend does not support dtype {other:?}"),
    }
    Ok(())
}

fn blend_slice<F>(a: &[u8], b: &[u8], t: f32, out: &mut [u8], blend_fn: F) -> anyhow::Result<()>
where
    F: Fn(&[f32], &[f32], f32, &mut [f32]),
{
    let elem = size_of::<f32>();
    if !a.len().is_multiple_of(elem) || a.len() != b.len() || a.len() != out.len() {
        anyhow::bail!("tensor byte length mismatch for f32 blend");
    }
    let count = a.len() / elem;
    let a_vals = bytes_to_f32(a);
    let b_vals = bytes_to_f32(b);
    let mut tmp = vec![0.0_f32; count];
    blend_fn(&a_vals, &b_vals, t, &mut tmp);
    write_f32(out, &tmp);
    Ok(())
}

fn blend_slice_f16<F>(a: &[u8], b: &[u8], t: f32, out: &mut [u8], blend_fn: F) -> anyhow::Result<()>
where
    F: Fn(&[f32], &[f32], f32, &mut [f32]),
{
    let elem = 2;
    if !a.len().is_multiple_of(elem) || a.len() != b.len() || a.len() != out.len() {
        anyhow::bail!("tensor byte length mismatch for f16 blend");
    }
    let count = a.len() / elem;
    let a_vals = f16_bytes_to_f32(a);
    let b_vals = f16_bytes_to_f32(b);
    let mut tmp = vec![0.0_f32; count];
    blend_fn(&a_vals, &b_vals, t, &mut tmp);
    write_f16(out, &tmp);
    Ok(())
}

fn blend_slice_bf16<F>(
    a: &[u8],
    b: &[u8],
    t: f32,
    out: &mut [u8],
    blend_fn: F,
) -> anyhow::Result<()>
where
    F: Fn(&[f32], &[f32], f32, &mut [f32]),
{
    let elem = 2;
    if !a.len().is_multiple_of(elem) || a.len() != b.len() || a.len() != out.len() {
        anyhow::bail!("tensor byte length mismatch for bf16 blend");
    }
    let count = a.len() / elem;
    let a_vals = bf16_bytes_to_f32(a);
    let b_vals = bf16_bytes_to_f32(b);
    let mut tmp = vec![0.0_f32; count];
    blend_fn(&a_vals, &b_vals, t, &mut tmp);
    write_bf16(out, &tmp);
    Ok(())
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn write_f32(out: &mut [u8], values: &[f32]) {
    for (chunk, value) in out.as_chunks_mut::<4>().0.iter_mut().zip(values) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
}

fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| f16_to_f32(u16::from_le_bytes(*chunk)))
        .collect()
}

fn write_f16(out: &mut [u8], values: &[f32]) {
    for (chunk, value) in out.as_chunks_mut::<2>().0.iter_mut().zip(values) {
        chunk.copy_from_slice(&f32_to_f16(*value).to_le_bytes());
    }
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| {
            let bits = u16::from_le_bytes(*chunk);
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect()
}

fn write_bf16(out: &mut [u8], values: &[f32]) {
    for (chunk, value) in out.as_chunks_mut::<2>().0.iter_mut().zip(values) {
        let bits = (value.to_bits() >> 16) as u16;
        chunk.copy_from_slice(&bits.to_le_bytes());
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            u32::from(sign) << 31
        } else {
            let mut e = -1_i32;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            let exp = (127 - 15 + 1 + e) as u32;
            (u32::from(sign) << 31) | (exp << 23) | (u32::from(f) << 13)
        }
    } else if exp == 0x1f {
        (u32::from(sign) << 31) | (0xff << 23) | (u32::from(frac) << 13)
    } else {
        let exp = exp as u32 + 127 - 15;
        (u32::from(sign) << 31) | (exp << 23) | (u32::from(frac) << 13)
    };
    f32::from_bits(f32_bits)
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;
    if exp == 255 {
        return (sign << 15) | (0x1f << 10) | ((frac != 0) as u16) << 9;
    }
    let mut new_exp = exp - 127 + 15;
    let mut new_frac = frac >> 13;
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign << 15;
        }
        new_frac |= 0x400;
        new_frac >>= 1 - new_exp;
        return (sign << 15) | new_frac as u16;
    }
    if new_exp >= 0x1f {
        return (sign << 15) | (0x1f << 10);
    }
    if (frac >> 12) & 1 == 1 && ((frac & 0xfff) != 0 || (new_frac & 1) == 1) {
        new_frac += 1;
        if new_frac == 0x400 {
            new_frac = 0;
            new_exp += 1;
            if new_exp >= 0x1f {
                return (sign << 15) | (0x1f << 10);
            }
        }
    }
    (sign << 15) | ((new_exp as u16) << 10) | (new_frac as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_midpoint() {
        let a = [0.0_f32, 1.0, 2.0];
        let b = [2.0_f32, 3.0, 4.0];
        let mut out = [0.0; 3];
        linear_f32(&a, &b, 0.5, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn slerp_endpoints() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        let mut out = [0.0; 2];
        slerp_f32(&a, &b, 0.0, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!(out[1].abs() < 1e-5);
        slerp_f32(&a, &b, 1.0, &mut out);
        assert!(out[0].abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn slerp_angle_is_sane() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        let mut out = [0.0; 2];
        slerp_f32(&a, &b, 0.5, &mut out);
        let norm = (out[0] * out[0] + out[1] * out[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
        // Midpoint between two orthogonal unit vectors sits at exactly 45°,
        // so both components must equal cos(45°) = 1/sqrt(2). Checking the
        // angle (not just norm + sign) pins down the actual interpolation.
        let half = std::f32::consts::FRAC_1_SQRT_2;
        assert!((out[0] - half).abs() < 1e-4, "out[0]={}", out[0]);
        assert!((out[1] - half).abs() < 1e-4, "out[1]={}", out[1]);
    }
}
