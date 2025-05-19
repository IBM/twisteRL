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

use std::vec;
use dyn_clone::DynClone;

pub trait Env : DynClone + Send + Sync {
    // Returns the number of possible actions
    fn num_actions(&self) -> usize;

    // Returns the size of the observations 
    fn obs_shape(&self) -> Vec<usize>;

    // Sets the current difficulty
    fn set_difficulty(&mut self, difficulty: usize){}

    // Returns current difficulty
    fn get_difficulty(&self) -> usize {1}

    // Sets itself a given input state (constructed from a Vec<usize>)
    fn set_state(&mut self, state: Vec<i64>);

    // Sets itself to a random initial state
    fn reset(&mut self);

    // Evolves the current state by an action
    fn step(&mut self, action: usize);

    // Returns an array with the action masks (True if an action is allowed, False if not)
    fn masks(&self) -> Vec<bool> {vec![true; self.num_actions()]}

    // Returns True if the given state is a terminal state
    fn is_final(&self) -> bool;

    // Returns the value of current state
    fn reward(&self) -> f32;

    // Returns current state encoded in a sparse format
    fn observe(&self) -> Vec<usize>;

    // Returns a list of possible permutations on the observations and the corresponding permutations on actions
    fn twists(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {(vec![], vec![])}
}

dyn_clone::clone_trait_object!(Env);