use super::*;
use rayon::prelude::*;

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

pub(super) fn quantize_from_f32_scalar(
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

pub(super) fn quantize_mixed_scalar_sequential(
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
