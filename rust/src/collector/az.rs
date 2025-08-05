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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::layers::{EmbeddingBag, Linear};
    use crate::nn::modules::Sequential;
    use crate::nn::policy::Policy;
    use crate::rl::env::Env;

    #[derive(Clone)]
    struct DummyEnv { step: usize }

    impl DummyEnv {
        fn new() -> Self { Self { step: 0 } }
    }

    impl Env for DummyEnv {
        fn as_any(&self) -> &dyn std::any::Any { self }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        fn num_actions(&self) -> usize { 1 }
        fn obs_shape(&self) -> Vec<usize> { vec![1] }
        fn set_state(&mut self, state: Vec<i64>) { self.step = state[0] as usize; }
        fn reset(&mut self) { self.step = 0; }
        fn step(&mut self, _action: usize) { self.step += 1; }
        fn masks(&self) -> Vec<bool> { vec![true] }
        fn is_final(&self) -> bool { self.step >= 1 }
        fn reward(&self) -> f32 { 1.0 }
        fn observe(&self) -> Vec<usize> { vec![0] }
    }

    fn dummy_policy() -> Policy {
        let emb = EmbeddingBag::new(vec![vec![1.0]], vec![0.0], false, vec![1], 0);
        let lin = Linear::new(vec![1.0], vec![0.0], false);
        let seq_a = Sequential::new(vec![Box::new(lin.clone())]);
        let seq_v = Sequential::new(vec![Box::new(lin)]);
        Policy::new(
            Box::new(emb),
            Box::new(Sequential::new(vec![])),
            Box::new(seq_a),
            Box::new(seq_v),
            vec![],
            vec![],
        )
    }

    #[test]
    fn test_azcollector_collect() {
        let env: Box<dyn Env> = Box::new(DummyEnv::new());
        let policy = dummy_policy();
        let collector = AZCollector::new(1, 1, 1.0, 1, 1);

        let data = collector.collect(&env, &policy).unwrap();
        assert_eq!(data.obs.len(), 2);
        assert!(data.additional_data.contains_key("remaining_values"));
    }
}

