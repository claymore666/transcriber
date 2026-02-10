use pyo3::prelude::*;

#[pymodule]
fn transcriber(_py: Python, _m: &Bound<PyModule>) -> PyResult<()> {
    // TODO: expose Python API
    Ok(())
}
