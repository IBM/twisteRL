// -*- coding: utf-8 -*-
/* 
(C) Copyright 2025 IBM. All Rights Reserved.

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
*/

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use anyhow::Error as AnyhowError;

// Code to map Rust errors to Python exceptions
// Wrapper for anyhow::Error
pub struct MyError(AnyhowError);

// `From<anyhow::Error> for MyError` for wrapping
impl From<AnyhowError> for MyError {
    fn from(err: AnyhowError) -> MyError {
        MyError(err)
    }
}

impl From<MyError> for PyErr {
    fn from(err: MyError) -> PyErr {
        PyRuntimeError::new_err(err.0.to_string())
    }
}
