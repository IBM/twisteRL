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

use crate::collector::collector::{Collector, CollectedData, merge};
use crate::nn::policy::{Policy, sample_from_logits};
use crate::rl::env::Env;

#[derive(Clone)]
pub struct PPOCollector {
    pub num_episodes: usize,
    pub gamma: f32,
    pub lambda: f32,
    pub num_cores: usize,
}

impl PPOCollector {
    pub fn new(
        num_episodes: usize,
        gamma: f32,
        lambda: f32,
        num_cores: usize,
    ) -> Self {
        PPOCollector { num_episodes, gamma, lambda, num_cores }
    }
}

impl PPOCollector {
    fn get_step_data(
        &self,
        env: &dyn Env,
        policy: &Policy,
    ) -> (Vec<usize>, Vec<f32>, usize, f32, f32) {
        let obs = env.observe();      // Vec<f32> or whatever your Env returns
        let masks   = env.masks();
        let reward  = env.reward();
        let (logits, value) = policy.forward(obs.clone(), masks);  
        let action = sample_from_logits(&logits);
        (obs, logits, action, value, reward)
    }

    fn single_collect(
        & self,
        env: &Box<dyn Env>,
        policy: &Policy,
    ) -> CollectedData {
        let mut env = env.clone();
        env.reset(); // We do not care about the original env in the collect

        let mut obss = Vec::new();
        let mut log_probs  = Vec::new();
        let mut vals  = Vec::new();
        let mut rews = Vec::new();
        let mut acts = Vec::new();

        loop {
            let (obs, log_prob, act, val, rew) = self.get_step_data(&*env, policy);
            obss.push(obs);
            log_probs.push(log_prob);
            vals.push(val);
            rews.push(rew);
            acts.push(act);

            if env.is_final() { break; }
            env.step(act);
        }

        // compute GAE advs/rets
        let n = rews.len();
        let mut advs = vec![0.0; n];
        let mut rets = vec![0.0; n];
        advs[n-1] = rews[n-1] - vals[n-1];
        rets[n-1] = rews[n-1];
        for t in (0..n-1).rev() {
            rets[t] = rews[t]
                    + self.gamma * (vals[t+1] + self.lambda * advs[t+1]);
            advs[t] = rets[t] - vals[t];
        }

        let mut data = CollectedData::new(
            obss,
            log_probs,
            vals,
            rews,
            acts,
        );
        data.additional_data.insert("advs".into(), advs);
        data.additional_data.insert("rets".into(), rets);
        data
    }
}

impl Collector for PPOCollector {
    fn collect(&self, env: &Box<dyn Env>, policy: &Policy) -> Result<CollectedData> {
        if self.num_cores == 1 {
            Ok(merge(
                (0..self.num_episodes).into_iter()  // Create a parallel iterator over the range 0..num_episodes
                    .map(|_| self.single_collect(env, policy)) // For each item in the range run a collection
                    .collect())?
            )
        } else {
            let pool: rayon::ThreadPool = ThreadPoolBuilder::new().num_threads(self.num_cores).build()?;
            // Use the thread pool to run the generation in parallel
            Ok(merge(pool.install(|| {
                (0..self.num_episodes).into_par_iter()  // Create a parallel iterator over the range 0..num_episodes
                    .map(|_| self.single_collect(env, policy)) // For each item in the range run a collection
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
    fn test_ppocollector_collect() {
        let env: Box<dyn Env> = Box::new(DummyEnv::new());
        let policy = dummy_policy();
        let collector = PPOCollector::new(1, 0.9, 0.95, 1);

        let data = collector.collect(&env, &policy).unwrap();
        assert_eq!(data.obs.len(), 2);
        assert!(data.additional_data.contains_key("rets"));
    }
}

