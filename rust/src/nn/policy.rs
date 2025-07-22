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

use rand::{prelude::Distribution, Rng};

use crate::nn::modules::Sequential;
use crate::nn::layers::EmbeddingBag;

#[derive(Clone)]
pub struct Policy {
    embeddings: Box<EmbeddingBag>,
    common: Box<Sequential>,
    action_net: Box<Sequential>,
    value_net: Box<Sequential>,
    obs_perms: Vec<Vec<usize>>,
    act_perms: Vec<Vec<usize>>
}

impl Policy {
    pub fn new(embeddings: Box<EmbeddingBag>, common: Box<Sequential>, action_net: Box<Sequential>, value_net: Box<Sequential>, obs_perms: Vec<Vec<usize>>, act_perms: Vec<Vec<usize>>) -> Self {
        Self { embeddings: embeddings, common, action_net, value_net, obs_perms, act_perms }
    }

    pub fn predict(&self, obs: Vec<usize>, masks: Vec<bool>) -> (Vec<f32>, f32) {
        // Forward of the action net
        let (action_logits, value) = self._raw_predict(obs, self.get_perm_id());

        // Apply masks to the actions
        let mut exp_masked_probs: Vec<f32> = action_logits.iter().zip(masks.iter()).map(|(&a, &m)| if m {a.exp()} else {0.0}).collect();

        // TODO: apply noise to the actions

        // Normalize actions
        let action_probs_sum: f32 = exp_masked_probs.iter().sum();
        exp_masked_probs = exp_masked_probs.iter().map(|&v| v / (action_probs_sum + 0.000001)).collect();
        (exp_masked_probs, value)
    }


    pub fn forward(&self, obs: Vec<usize>, masks: Vec<bool>) -> (Vec<f32>, f32) {
        // Similar to predict but outputs unnormalized logits instead of probabilities

        // Forward of the action net
        let (action_logits, value) = self._raw_predict(obs, self.get_perm_id());

        // Apply masks to the actions
        let masked_logits: Vec<f32> = action_logits.iter().zip(masks.iter()).map(|(&a, &m)| if m {a} else {-1e10}).collect();

        (masked_logits, value)
    }

    fn get_perm_id(&self) -> Option<usize> {
        let mut n_perm: Option<usize> = None;

        // Select a random permutation (if there are perms)
        if self.obs_perms.len() > 0 {
            let mut rng = rand::thread_rng();
            n_perm = Some(rand::distributions::Uniform::new(0, self.obs_perms.len()).sample(&mut rng));
        }

        n_perm
    }

    fn _raw_predict(&self, mut obs: Vec<usize>, n_perm: Option<usize>) -> (Vec<f32>, f32) {
        // Permute the obs according to the obs_perm
        if let Some(pi) = n_perm {
            obs = obs.iter().map(|&v| self.obs_perms[pi][v]).collect();
        }

        // Do forward pass of the shared nn part
        let common_out = self.common.forward(self.embeddings.forward(&obs));

        // Forward of the value net
        let value = self.value_net.forward(common_out.clone()).sum(); // This only has one element

        // Forward of the action net
        let mut action_logits  = self.action_net.forward(common_out).data.as_vec().to_owned();

        // Permute logits according to the corresponding act_perm
        if let Some(pi) = n_perm {
            action_logits = self.act_perms[pi].iter().map(|&v| action_logits[v]).collect();
        }

        (action_logits, value)
    }

    pub fn full_predict(&self, obs: Vec<usize>, masks: Vec<bool>) -> (Vec<f32>, f32) {
        if self.obs_perms.len() == 0 {return self.predict(obs, masks);};

        // Forward of the action net for each perm
        let mut action_logits = vec![0.0f32; self.act_perms[0].len()];
        let mut value = 0.0f32;

        for pi in 0..self.obs_perms.len() {
            let (action_logits_pi, value_pi) = self._raw_predict(obs.clone(), Some(pi));
            value += value_pi / (self.obs_perms.len() as f32);
            for i in 0..action_logits_pi.len() {
                action_logits[i] += action_logits_pi[i] / (self.obs_perms.len() as f32);
            }
        }

        // Apply masks to the actions
        let mut exp_masked_probs: Vec<f32> = action_logits.iter().zip(masks.iter()).map(|(&a, &m)| if m {a.exp()} else {0.0}).collect();

        // TODO: apply noise to the actions

        // Normalize actions
        let action_probs_sum: f32 = exp_masked_probs.iter().sum();
        exp_masked_probs = exp_masked_probs.iter().map(|&v| v / (action_probs_sum + 0.000001)).collect();
        (exp_masked_probs, value)
    }

}

pub fn argmax(values: &Vec<f32>) -> usize {
    // If the vector is empty, return 0 by default.
    if values.is_empty() {
        return 0;
    }

    let mut max_idx = 0;
    let mut max_val = values[0];

    // We start iterating from the second element because we already took
    // the first as the initial `max_val`.
    for (i, &val) in values.iter().enumerate().skip(1) {
        // Using a direct comparison (`val > max_val`) will simply ignore NaNs,
        // because any comparison with NaN is false.
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx
}

pub fn sample(probs: &Vec<f32>) -> usize {
    let mut rng = rand::thread_rng();  // Random number generator

    match rand::distributions::WeightedIndex::new(probs) {
        Ok(dist) => {
            dist.sample(&mut rng)
        }
        Err(err) => {
            // Handle the error and print the `probs` value
            println!("Failed to create WeightedIndex: {:?}", err);
            println!("The problematic probs were: {:?}", probs);
            0
        }
    }
}

pub fn sample_from_logits(probs: &Vec<f32>) -> usize {
    let mut rng = rand::thread_rng();  // Random number generator
    argmax(&probs.iter().map(|&v| v - rng.gen::<f32>().ln().abs().ln()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_basic() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(argmax(&v), 2);
    }

    #[test]
    fn test_argmax_with_nan() {
        let v = vec![f32::NAN, 1.0, 0.5];
        assert_eq!(argmax(&v), 0);
    }

    #[test]
    fn test_sample_range() {
        let probs = vec![0.2, 0.3, 0.5];
        let idx = sample(&probs);
        assert!(idx < probs.len());
    }

    #[test]
    fn test_sample_from_logits_range() {
        let logits = vec![0.1, 2.0, 0.3];
        let idx = sample_from_logits(&logits);
        assert!(idx < logits.len());
    }
}
