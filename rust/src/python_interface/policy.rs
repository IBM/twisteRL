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

use crate::python_interface::modules::PySequential;
use crate::python_interface::layers::PyEmbeddingBag;
use crate::nn::policy::Policy;

#[pyclass(name="Policy")]
pub struct PyPolicy {
    pub policy: Box<Policy>,
}

#[pymethods]
impl PyPolicy {
    #[new]
    pub fn new(embeddings: PyEmbeddingBag, common: PySequential, action_net: PySequential, value_net: PySequential, obs_perms: Vec<Vec<usize>>, act_perms: Vec<Vec<usize>>) -> Self {
        let policy = Box::new(Policy::new(embeddings.embedding, common.seq, action_net.seq, value_net.seq, obs_perms, act_perms));
        PyPolicy { policy }
    }

    pub fn predict(&self, obs: Vec<usize>, masks: Vec<bool>) -> (Vec<f32>, f32) {
        self.policy.predict(obs, masks)
    }


    pub fn forward(&self, obs: Vec<usize>, masks: Vec<bool>) -> (Vec<f32>, f32) {
        self.policy.forward(obs, masks)
    }

    pub fn full_predict(&self, obs: Vec<usize>, masks: Vec<bool>) -> (Vec<f32>, f32) {
        self.policy.full_predict(obs, masks)
    }

}
