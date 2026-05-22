#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationSuite {
    VulkanDflashCpu,
    FullPipeline,
    SmokeCheck,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationResult {
    pub suite: ValidationSuite,
    pub max_abs_diff: f32,
    pub tolerance: f32,
}

impl ValidationResult {
    pub fn passed(&self) -> bool {
        self.max_abs_diff <= self.tolerance
    }
}

pub fn compare_outputs(
    suite: ValidationSuite,
    expected: &[f32],
    actual: &[f32],
    tolerance: f32,
) -> Result<ValidationResult, String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "validation length mismatch: expected {}, actual {}",
            expected.len(),
            actual.len()
        ));
    }
    let max_abs_diff = expected
        .iter()
        .zip(actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    Ok(ValidationResult {
        suite,
        max_abs_diff,
        tolerance,
    })
}

pub fn implemented_validation_suites() -> &'static [ValidationSuite] {
    &[
        ValidationSuite::VulkanDflashCpu,
        ValidationSuite::FullPipeline,
        ValidationSuite::SmokeCheck,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_reports_pass_and_failure() {
        let pass = compare_outputs(
            ValidationSuite::SmokeCheck,
            &[1.0, 2.0],
            &[1.001, 2.0],
            0.01,
        )
        .unwrap();
        assert!(pass.passed());

        let fail = compare_outputs(ValidationSuite::FullPipeline, &[1.0], &[1.2], 0.01).unwrap();
        assert!(!fail.passed());
    }
}
