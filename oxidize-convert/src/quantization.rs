use oxidize_core::gguf::GgufQuantizationType;

pub fn parse_target(value: &str) -> Result<GgufQuantizationType, String> {
    match value.to_ascii_uppercase().as_str() {
        "F32" => Ok(GgufQuantizationType::F32),
        "F16" => Ok(GgufQuantizationType::F16),
        "Q4_0" => Ok(GgufQuantizationType::Q4_0),
        "Q4_K_S" => Ok(GgufQuantizationType::Q4_K_S),
        "Q4_K_M" => Ok(GgufQuantizationType::Q4_K_M),
        "Q6_K" => Ok(GgufQuantizationType::Q6_K),
        "Q8_0" => Ok(GgufQuantizationType::Q8_0),
        "AL5" => Ok(GgufQuantizationType::AL5),
        "AL5_XS" => Ok(GgufQuantizationType::AL5_XS),
        "AL6" => Ok(GgufQuantizationType::AL6),
        "AL8" => Ok(GgufQuantizationType::AL8),
        _ => Err(format!("unsupported --target quantization: {value}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_target_case_insensitively() {
        assert_eq!(parse_target("q4_k_m"), Ok(GgufQuantizationType::Q4_K_M));
        assert_eq!(parse_target("F16"), Ok(GgufQuantizationType::F16));
    }

    #[test]
    fn rejects_unknown_target() {
        let err = parse_target("wat").expect_err("unknown target must fail");
        assert!(err.contains("unsupported"));
    }
}
