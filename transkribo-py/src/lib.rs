use pyo3::prelude::*;

#[pymodule]
fn transkribo(_py: Python, _m: &Bound<PyModule>) -> PyResult<()> {
    // TODO: expose Python API
    Ok(())
}
