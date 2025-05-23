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

use crate::nn::layers::{EmbeddingBag, Linear};


#[pyclass(name="Linear")]
#[derive(Clone)]
pub struct PyLinear {
    pub linear: Box<Linear>
}


#[pymethods]
impl PyLinear {
    #[new]
    pub fn new(weights_vector: Vec<f32>, bias_vector: Vec<f32>, apply_relu: bool) -> Self {
        let linear = Box::new(Linear::new(weights_vector, bias_vector, apply_relu));
        PyLinear { linear }
    }
}


#[pyclass(name="EmbeddingBag")]
#[derive(Clone)]
pub struct PyEmbeddingBag {
    pub embedding : Box<EmbeddingBag>
}

#[pymethods]
impl PyEmbeddingBag {
    #[new]
    pub fn new(vec_vectors: Vec<Vec<f32>>, bias_vector: Vec<f32>, apply_relu: bool, obs_shape: Vec<usize>, conv_dim: usize) -> Self {
        let embedding = Box::new(EmbeddingBag::new(vec_vectors, bias_vector, apply_relu, obs_shape, conv_dim));
        PyEmbeddingBag { embedding }
    }
}
