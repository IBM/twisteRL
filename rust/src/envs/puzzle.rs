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

use rand::distributions::{Distribution, Uniform};
use crate::rl::env::Env;


// This is the Env definition
#[derive(Clone)]
pub struct Puzzle {
    pub state: Vec<usize>,
    pub zero_location: (usize, usize),
    pub depth: usize,

    pub width: usize,
    pub height: usize,
    pub difficulty: usize,
    pub depth_slope: usize,
    pub max_depth: usize,
}


impl Puzzle {
    pub fn new(
        width: usize,
        height: usize,
        difficulty: usize,
        depth_slope: usize,
        max_depth: usize,
    ) -> Self {
        Puzzle {state: (0..(width*height)).collect(), zero_location: (0,0), depth:1, width, height, difficulty, depth_slope, max_depth}
    }

    pub fn solved(&self) -> bool {
        for i in 0..self.state.len() {
            if self.state[i] != i {return false}
        }

        true
    }

    pub fn get_state(&self) -> Vec<usize> {
        self.state.clone()
    }

    pub fn display(&self) {
        for (i, &v) in self.state.iter().enumerate() {
            if v == 0 {
                print!("   ");
            } else if v < 10 {
                print!("  {} ", v);
            } else {
                print!(" {} ", v);
            }
            if (i+1)%self.width == 0 {
                print!("\n")
            }
        }
    }

    pub fn set_position(&mut self, x: usize, y: usize, val: usize) {
        self.state[y*self.width + x] = val;
    }

    pub fn get_position(&self, x: usize, y: usize) -> usize {
        self.state[y*self.width + x]
    }
}

// This implements the necessary functions for the environment
impl Env for Puzzle {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn num_actions(&self) -> usize {
        4
    }

    fn obs_shape(&self) -> Vec<usize> {
        vec![self.state.len(), self.state.len()]
        //vec![4, 6*self.tmax]
    }

    fn set_difficulty(&mut self, difficulty: usize) {
        self.difficulty = difficulty;
    }

    fn get_difficulty(&self) -> usize {
        self.difficulty
    }

    fn set_state(&mut self, state: Vec<i64>) {
        self.state = state.iter().map(|&x| x as usize).collect();
        self.depth = self.max_depth;

        for (i, &s) in state.iter().enumerate() {
            if s == 0 {
                self.zero_location = (i % self.width, i / self.width);
                break;
            }
        }
    }

    fn reset(&mut self) {
        // Reset the state to the target
        self.state = (0..(self.width * self.height)).collect();
        self.zero_location = (0,0);

        let mut rng = rand::thread_rng();
        let action_range = Uniform::new(0, self.num_actions());

        // Apply random actions based on the difficulty
        for _ in 0..self.difficulty {
            let action = action_range.sample(&mut rng);
            self.step(action);
        }
        self.depth = self.depth_slope * self.difficulty;  
    }

    fn step(&mut self, action: usize)  {
        let (zx, zy) = self.zero_location;
        if (action == 0) && (zx > 0) {
            let new_val = self.get_position(zx-1, zy);
            self.set_position(zx, zy, new_val);
            self.set_position(zx-1, zy, 0);
            self.zero_location = (zx-1, zy);
        } else if (action == 1) && (zy > 0) {
            let new_val = self.get_position(zx, zy-1);
            self.set_position(zx, zy, new_val);
            self.set_position(zx, zy-1, 0);
            self.zero_location = (zx, zy-1);
        } else if (action == 2) && (zx < (self.width-1)) {
            let new_val = self.get_position(zx+1, zy);
            self.set_position(zx, zy, new_val);
            self.set_position(zx+1, zy, 0);
            self.zero_location = (zx+1, zy);
        } else if (action == 3) && (zy < (self.height-1)) {
            let new_val = self.get_position(zx, zy+1);
            self.set_position(zx, zy, new_val);
            self.set_position(zx, zy+1, 0);
            self.zero_location = (zx, zy+1);
        }

        self.depth = self.depth.saturating_sub(1); 
    }
    
    fn masks(&self) -> Vec<bool> {
        let (zx, zy) = self.zero_location;
        vec![(zx > 0), (zy > 0), (zx < (self.width-1)), (zy < (self.height-1))]
    }

    fn is_final(&self) -> bool {
        self.depth == 0 || self.solved()
    }

    fn reward(&self) -> f32 {
        if self.solved() {
            1.0
        } else {
            if self.depth == 0 { -0.5 } else { -0.5/(self.max_depth as f32) }
        }
    }

    fn observe(&self,) -> Vec<usize> {    
        self.state.iter().enumerate().map(|(i, v)| i * self.height * self.width + v).collect()  
    }
    
}
