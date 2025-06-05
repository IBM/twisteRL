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

use crate::rl::env::Env;
use crate::nn::policy::Policy;
use super::solve::solve; 

pub fn evaluate(
    env: &Box<dyn Env>,
    policy: &Policy,
    num_episodes: usize,
    deterministic: bool,
    num_searches: usize,
    num_mcts_searches: usize,
    _seed: usize,          // unused for now
    C: f32,
    max_expand_depth: usize,
    num_cores: usize,
) -> (f32, f32) {
    let mut env = env.clone();
    if num_cores <= 1 {
        let mut successes = 0.0;
        let mut rewards  = 0.0;
        for _ in 0..num_episodes {
            env.reset();
            let ((success, reward), _path) = solve(
                &env,
                &policy,
                deterministic,
                num_searches,
                num_mcts_searches,
                C,
                max_expand_depth,
            );
            successes += success;
            rewards  += reward;
        }
        (successes/(num_episodes as f32), rewards/(num_episodes as f32))
    } else {
        // Parallel evaluation via Rayon
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_cores)
            .build()
            .unwrap();

        let (successes, total_vals) = pool.install(|| {
            (0..num_episodes)
                .into_par_iter()
                .map(|_| {
                    let mut env = env.clone();
                    env.reset();
                    let ((success, reward), _path) = solve(
                        &env,
                        &policy,
                        deterministic,
                        num_searches,
                        num_mcts_searches,
                        C,
                        max_expand_depth,
                    );
                    (success, reward)
                })
                .reduce(
                    || (0.0, 0.0),  // Initialize total successes and rewards to to zero
                    |mut acc, (success, total_val)| {         
                        acc.0 += success; // add the successes
                        acc.1 += total_val;  // add the rewards
                        acc // return the values
                    }
                )
        });

        (successes / (num_episodes as f32), total_vals / (num_episodes as f32))
    }
}