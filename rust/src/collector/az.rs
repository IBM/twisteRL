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
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use anyhow::Result;

use crate::rl::env::Env;
use crate::nn::policy::{Policy, sample};
use crate::collector::collector::{CollectedData, Collector, merge};
use crate::rl::search::predict_probs_mcts;

#[derive(Clone)]
pub struct AZCollector {
    pub num_episodes: usize,
    pub num_mcts_searches: usize,
    pub C: f32,
    pub max_expand_depth: usize,
    pub num_cores: usize,
}

impl AZCollector {
    pub fn new(
        num_episodes: usize,
        num_mcts_searches: usize,
        C: f32,
        max_expand_depth: usize,
        num_cores: usize,
    ) -> Self {
        AZCollector {
            num_episodes,
            num_mcts_searches,
            C,
            max_expand_depth,
            num_cores,
        }
    }
}

impl AZCollector {
    /// Runs one episode, returns its `CollectedData`
    fn single_collect(
        &self,
        env: &Box<dyn Env>,
        policy: &Policy,
    ) -> CollectedData {
        let mut env = env.clone();
        env.reset();

        // Init data vecs
        let mut obs: Vec<Vec<usize>> = vec![];
        let mut probs: Vec<Vec<f32>> = vec![];
        let mut vals: Vec<f32> = vec![];
        let mut total_vals: Vec<f32> = vec![];

        let mut total_val = 0.0;

        // Loop until a final state
        loop {
            // Calculate MCTS probs for current state
            let mcts_probs = predict_probs_mcts(env.clone(), policy, self.num_mcts_searches, self.C, self.max_expand_depth);

            // Select next action and get current value
            let action = sample(&mcts_probs);
            let val = env.reward();
            total_vals.push(total_val);

            total_val += val;

            // Store data
            obs.push(env.observe());
            probs.push(mcts_probs);
            vals.push(val);

            // Break if we are in a final state
            if env.is_final() {
                break;
            }

            // Move to next state
            env.step(action);
        }

        // Post process rewards
        let remaining_vals: Vec<f32> = total_vals.iter().map(|&v| total_val - v).collect();

        let mut data = CollectedData::new(
            obs,
            probs,
            vec![],
            vec![],
            vec![],
        );
        data.additional_data
            .insert("remaining_values".into(), remaining_vals);

        data
    }
}

impl Collector for AZCollector {
    fn collect(&self, env: &Box<dyn Env>, policy: &Policy) -> Result<CollectedData> {
        if self.num_cores == 1 {
            Ok(merge(
                (0..self.num_episodes).into_iter()  
                    .map(|_| self.single_collect(env, policy)) 
                    .collect()
            )?)
        } else {
            let pool: rayon::ThreadPool = ThreadPoolBuilder::new().num_threads(self.num_cores).build()?;
            // Use the thread pool to run the generation in parallel
            Ok(merge(pool.install(|| {
                (0..self.num_episodes).into_par_iter()  // Create a parallel iterator over the range 0..num_episodes
                    .map(|_| self.single_collect(env, policy)) 
                    .collect()
            }))?)
        }
    }
}

