use pyo3::prelude::*;

const PYTHON_PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");
const PYTHON_PACKAGE_EXPORTS: [&str; 2] = ["workspace_health", "version"];

fn health_status() -> &'static str {
    llamas_core::workspace_health().status
}

#[pyfunction]
fn workspace_health() -> &'static str {
    health_status()
}

#[pyfunction]
fn version() -> &'static str {
    PYTHON_PACKAGE_VERSION
}

#[pymodule]
fn llamas(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(workspace_health, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add("__version__", PYTHON_PACKAGE_VERSION)?;
    module.add("__all__", PYTHON_PACKAGE_EXPORTS)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_status_matches_core_workspace_health() {
        assert_eq!(health_status(), "ready");
    }

    #[test]
    fn python_package_has_workspace_version() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn python_package_exports_expected_symbols() {
        assert_eq!(PYTHON_PACKAGE_EXPORTS, ["workspace_health", "version"]);
    }
}
