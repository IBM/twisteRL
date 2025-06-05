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

use std::collections::HashMap;
use crate::nn::policy::Policy;
use crate::rl::env::Env;


/// Container for collected rollout data using plain Rust vectors.
pub struct CollectedData {
    /// Observations at each timestep: Vec of feature Vecs
    pub obs: Vec<Vec<usize>>,
    /// Logits (action probabilities) at each timestep
    pub logits: Vec<Vec<f32>>,
    /// Value estimates at each timestep
    pub values: Vec<f32>,
    /// Rewards received at each timestep
    pub rewards: Vec<f32>,
    /// Actions taken at each timestep
    pub actions: Vec<usize>,
    /// Additional data (e.g., GAE advantages, returns)
    pub additional_data: HashMap<String, Vec<f32>>,
}

impl CollectedData {
    /// Construct a new CollectedData from raw rollout vectors.
    pub fn new(
        obs: Vec<Vec<usize>>,
        logits: Vec<Vec<f32>>,
        values: Vec<f32>,
        rewards: Vec<f32>,
        actions: Vec<usize>,
    ) -> Self {
        CollectedData {
            obs,
            logits,
            values,
            rewards,
            actions,
            additional_data: HashMap::new(),
        }
    }

    /// Merge another CollectedData into this one by appending all vectors.
    pub fn merge(&mut self, other: &CollectedData) {
        // Append observations and logits (2D vectors)
        self.obs.extend(other.obs.iter().cloned());
        self.logits.extend(other.logits.iter().cloned());

        // Append 1D vectors
        self.values.extend(&other.values);
        self.rewards.extend(&other.rewards);
        self.actions.extend(&other.actions);

        // Merge additional_data: append Vec<f32> values
        for (key, value_vec) in &other.additional_data {
            self.additional_data
                .entry(key.clone())
                .and_modify(|existing| existing.extend(value_vec.iter().cloned()))
                .or_insert_with(|| value_vec.clone());
        }
    }
}

/// A generic trait for collecting data.
pub trait Collector: Send + Sync{
    /// Runs the collection process and returns accumulated data.
    fn collect(&self, env: &Box<dyn Env>, policy: &Policy) -> CollectedData;
}
