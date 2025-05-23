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

use crate::nn::modules::Sequential;
use super::layers::PyLinear;

#[pyclass(name="Sequential")]
#[derive(Clone)]
pub struct PySequential {
    pub seq: Box<Sequential>,
}

#[pymethods]
impl PySequential {
    #[new]
    pub fn new(layers: Vec<PyLinear>) -> Self {
        let rs_layers = layers.iter().map(|layer| layer.linear.clone()).collect();
        let seq = Box::new(Sequential::new(rs_layers));
        PySequential { seq }
    }
}
