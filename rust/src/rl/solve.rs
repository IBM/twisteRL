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
use crate::rl::env::Env;
use crate::nn::policy::{Policy, sample, argmax};
use super::search::predict_probs_mcts;

pub fn single_solve(
    env: &mut Box<dyn Env>,
    policy: &Policy,
    deterministic: bool,
    num_mcts_searches: usize,
    C: f32,
    max_expand_depth: usize,
) -> ((f32, f32), Vec<usize>) {
    let mut total_val = 0.0;
    let mut solution = Vec::new();

    // step until final
    while !env.is_final() {
        let val = env.reward();
        let obs = env.observe();
        let masks = env.masks();
        total_val += val;
        
        // choose probs via either policy or MCTS
        let probs = if num_mcts_searches == 0 {
            policy.predict(obs, masks).0
        } else {
            // this will internally clone/reset the env for search
            predict_probs_mcts(
                env.clone(), 
                policy, 
                num_mcts_searches, 
                C, 
                max_expand_depth,
            )
        };

        let action = if deterministic {
            argmax(&probs)
        } else {
            sample(&probs)
        };

        env.step(action);
        solution.push(action);
    }

    let val = env.reward();
    total_val += val;

    (((val == 1.0) as usize as f32, total_val), solution)
}

pub fn solve(
    env: &Box<dyn Env>,
    policy: &Policy,
    deterministic: bool,
    num_searches: usize,
    num_mcts_searches: usize,
    C: f32,
    max_expand_depth: usize,
) -> ((f32, f32), Vec<usize>) {
    let mut best: ((f32, f32), Vec<usize>) = ((0.0, f32::NEG_INFINITY), Vec::new());

    for _ in 0..num_searches {
        let mut cloned_env = env.clone(); // Clone to avoid changing the original env
        let next_val = single_solve(
            &mut cloned_env,
            policy,
            deterministic,
            num_mcts_searches,
            C,
            max_expand_depth,
        );

        if next_val.0 > best.0 {
            best = next_val;
        }
    }

    best
}
