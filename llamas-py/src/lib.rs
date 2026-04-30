use pyo3::prelude::*;

fn health_status() -> &'static str {
    llamas_core::workspace_health().status
}

#[pyfunction]
fn workspace_health() -> &'static str {
    health_status()
}

#[pymodule]
fn llamas(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(workspace_health, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_status_matches_core_workspace_health() {
        assert_eq!(health_status(), "ready");
    }
}
