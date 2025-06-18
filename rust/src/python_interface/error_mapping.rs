use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use anyhow::Error as AnyhowError;

// 1. Definimos un struct local que envuelva a anyhow::Error
pub struct MyError(AnyhowError);

// 2. Implementamos `From<anyhow::Error> for MyError` para facilitar el wrapping
impl From<AnyhowError> for MyError {
    fn from(err: AnyhowError) -> MyError {
        MyError(err)
    }
}

// 3. Ahora podemos implementar `From<MyError> for PyErr` en nuestro crate
impl From<MyError> for PyErr {
    fn from(err: MyError) -> PyErr {
        PyRuntimeError::new_err(err.0.to_string())
    }
}
