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

use crate::rl::tree::Tree;
use crate::nn::policy::{sample, Policy};

pub struct MCTSNode {
    pub state: Box<dyn Env>,
    pub action_taken: Option<usize>,
    pub prior: f32,
    pub visit_count: u32,
    pub value_sum: f32,
}

impl MCTSNode {
    pub fn ucb(&self, child: &MCTSNode, C: f32) -> f32 {
        let q = if child.visit_count == 0 {
            0.0
        } else {
            child.value_sum / (child.visit_count as f32)
        };
        q + C
            * ((self.visit_count as f32).sqrt()
               / (child.visit_count as f32 + 1.0))
            * child.prior
    }
}

pub type MCTSTree = Tree<MCTSNode>;

impl MCTSTree {
    pub fn backpropagate(&mut self, node_idx: usize, value: f32) {
        let node = &mut self.nodes[node_idx].val;
        node.value_sum += value;
        node.visit_count += 1;

        if let Some(parent) = self.nodes[node_idx].parent {
            self.backpropagate(parent, value);
        }
    }

    // Expand the node by adding child nodes based on the policy probabilities
    pub fn expand(&mut self, node_idx: usize, action_priors: Vec<f32>) {
        
        for (action, &prob) in action_priors.iter().enumerate() {
            if prob <= 0.0 { continue; }
            
            let mut next_state = self.nodes[node_idx].val.state.clone();
            next_state.step(action);

            self.add_child_to_node(
            MCTSNode{
                    state: next_state,
                    action_taken: Some(action),
                    prior: prob,
                    visit_count: 0,
                    value_sum: 0.0
                }, 
                node_idx
            );
        }
    }

    pub fn next(&self, node_idx: usize, C: f32) -> usize {
        let mut best_child = None;
        let mut best_ucb = f32::NEG_INFINITY;

        for child_idx in self.nodes[node_idx].children.iter() {
            let ucb = self.nodes[node_idx].val.ucb(&self.nodes[*child_idx].val, C);

            if ucb > best_ucb {
                best_child = Some(*child_idx);
                best_ucb = ucb;
            }
        }

        best_child.expect("No child selected in next()")
    }

    // Select the best child node based on UCB score
    pub fn next_sample(&self, node_idx: usize) -> usize {
        let probs: Vec<f32> = self.nodes[node_idx].children.iter().map(
            |&child_idx| self.nodes[child_idx].val.prior
        ).collect();
        let child_num = sample(&probs);
        self.nodes[node_idx].children[child_num]
    }
}


pub fn predict_probs_mcts(
    env: Box<dyn Env>,
    policy: &Policy,
    num_mcts_searches: usize,
    C: f32,
    max_expand_depth: usize,
) -> Vec<f32> {
    // Perform the MCTS search starting from the given state
    let root_state = env.clone();

    // Get the initial policy and value from the neural network
    let (action_probs, _) = policy.full_predict(root_state.observe(), root_state.masks());

    // Create the tree and root node
    let mut tree: Tree<MCTSNode> = Tree::new();

    let root_idx = tree.new_node(MCTSNode {
        state: root_state,
        action_taken: None,
        prior: 0.0,
        visit_count: 1,
        value_sum: 0.0,
    });

    // Expand the root node
    tree.expand(root_idx, action_probs);

    // Perform the search iterations
    for _ in 0..num_mcts_searches {
        let mut node_idx = root_idx.clone();
        
        // First expand until leaf node
        while !tree.nodes[node_idx].children.is_empty() {
            node_idx = tree.next(node_idx, C);
        }
        let mut value = 0.0f32;
        let mut expanded_depth = 0;  // This counts how many steps we expand at most
        
        // Then add nodes until an end state
        while expanded_depth < max_expand_depth {
            // Get value
            let node_state = &*tree.nodes[node_idx].val.state;
            value = node_state.reward();

            // Break if is_final
            if node_state.is_final() {
                break;
            }

            // If not, predict actions, expand tree and select by sampling
            let (action_probs, new_value) = 
                policy.full_predict(node_state.observe(), node_state.masks());
            tree.expand(node_idx, action_probs);
            node_idx = tree.next_sample(node_idx);
            value = new_value;
            expanded_depth += 1;
        }

        // Backpropagation
        tree.backpropagate(node_idx, value);
    }

    let n_actions = env.num_actions() as usize;

    // Calculate the action probabilities from the root's children
    let mut mcts_action_probs = vec![0.0f32; n_actions];
    
    for &child_idx in &tree.nodes[root_idx].children {
        let act = tree.nodes[child_idx]
            .val
            .action_taken
            .expect("expanded nodes must have an action") as usize;
        mcts_action_probs[act] = tree.nodes[child_idx].val.visit_count as f32;
    }

    let sum_probs: f32 = mcts_action_probs.iter().sum();
    if sum_probs > 0.0 {
        for p in mcts_action_probs.iter_mut() {
            *p /= sum_probs;
        }
    } else {
        mcts_action_probs = vec![1.0/(n_actions as f32); n_actions];
    }

    mcts_action_probs
}
