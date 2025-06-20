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

use pyo3::prelude::*;
use crate::rl::env::Env;
use crate::envs::puzzle::Puzzle;
use crate::python_interface::policy::PyPolicy;
use crate::python_interface::error_mapping::MyError;
use crate::rl::solve::solve;
use crate::rl::evaluate::evaluate;
use std::any::Any;

/// Generic helper functions for extracting concrete environment types from PyBaseEnv
pub fn get_env_ref<T: Any>(base_env: &PyBaseEnv) -> PyResult<&T> {
    base_env.env.as_any().downcast_ref::<T>()
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err(
            format!("Expected environment of type {}", std::any::type_name::<T>())
        ))
}

pub fn get_env_mut<T: Any>(base_env: &mut PyBaseEnv) -> PyResult<&mut T> {
    base_env.env.as_any_mut().downcast_mut::<T>()
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err(
            format!("Expected environment of type {}", std::any::type_name::<T>())
        ))
}


#[pyclass(subclass)]
pub struct PyBaseEnv {
    pub env: Box<dyn Env>,
}

#[pymethods]
impl PyBaseEnv {
    // Returns the number of possible actions
    fn num_actions(&self) -> PyResult<usize> {
        Ok(self.env.num_actions())
    }

    // Returns the size of the observations 
    fn obs_shape(&self) -> PyResult<Vec<usize>>{
        Ok(self.env.obs_shape())
    }

    // Sets the current difficulty
    #[setter]
    fn set_difficulty(&mut self, difficulty: usize) -> PyResult<()> {
        Ok(self.env.set_difficulty(difficulty))
    }

    // Returns current difficulty
    #[getter]
    fn get_difficulty(&self) -> PyResult<usize> {
        Ok(self.env.get_difficulty())
    }

    // Sets itself a given input state (constructed from a Vec<usize>)
    fn set_state(&mut self, state: Vec<i64>) -> PyResult<()> {
        Ok(self.env.set_state(state))
    }

    // Sets itself to a random initial state
    fn reset(&mut self) -> PyResult<()>{
        Ok(self.env.reset())
    }

    // Evolves the current state by an action
    fn step(&mut self, action: usize) -> PyResult<()>{
        Ok(self.env.step(action))
    }

    // Returns an array with the action masks (True if an action is allowed, False if not)
    fn masks(&self) -> PyResult<Vec<bool>> {
        Ok(self.env.masks())
    }

    // Returns True if the given state is a terminal state
    fn is_final(&self) -> PyResult<bool>{
        Ok(self.env.is_final())
    }

    // Returns the value of current state
    fn reward(&self) -> PyResult<f32> {
        Ok(self.env.reward())
    }

    // Returns current state encoded in a sparse format
    fn observe(&self) -> PyResult<Vec<usize>> {
        Ok(self.env.observe())
    }

    // Returns a list of possible permutations on the observations and the corresponding permutations on actions
    fn twists(&self) -> PyResult<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        Ok(self.env.twists())
    }

    // Hidden method to extract the Rust env as a pointer
    fn __extract_env__(&self) -> PyResult<usize> {
        // Return the Box pointer itself as a usize
        let box_ptr = &self.env as *const Box<dyn Env> as usize;
        Ok(box_ptr)
    }
}


#[pyclass(name="Puzzle", extends=PyBaseEnv)]
pub struct PyPuzzleEnv;


#[pymethods]
impl PyPuzzleEnv {
    #[new]
    pub fn new(
        width: usize,
        height: usize,
        difficulty: usize,
        depth_slope: usize,
        max_depth: usize,
    ) -> (Self, PyBaseEnv) {
        let puzzle = Puzzle::new(width, height, difficulty, depth_slope, max_depth);
        let env = Box::new(puzzle);
        (PyPuzzleEnv, PyBaseEnv { env: env })
    }

    pub fn solved(slf: PyRef<'_, Self>) -> PyResult<bool> {
        let puzzle = get_env_ref::<Puzzle>(slf.as_ref())?;
        Ok(puzzle.solved())
    }

    pub fn get_state(slf: PyRef<'_, Self>) -> PyResult<Vec<usize>> {
        let puzzle = get_env_ref::<Puzzle>(slf.as_ref())?;
        Ok(puzzle.get_state())
    }

    pub fn display(slf: PyRef<'_, Self>) -> PyResult<()> {
        let puzzle = get_env_ref::<Puzzle>(slf.as_ref())?;
        Ok(puzzle.display())
    }

    pub fn set_position(mut slf: PyRefMut<'_, Self>, x: usize, y: usize, val: usize) -> PyResult<()> {
        let puzzle = get_env_mut::<Puzzle>(slf.as_mut())?;
        Ok(puzzle.set_position(x, y, val))
    }

    pub fn get_position(slf: PyRef<'_, Self>, x: usize, y: usize) -> PyResult<usize> {
        let puzzle = get_env_ref::<Puzzle>(slf.as_ref())?;
        Ok(puzzle.get_position(x, y))
    }
}


pub fn get_env<'a>(py_env: &'a Bound<'_, PyAny>) -> PyResult<&'a Box<dyn Env>> {
    // try to call __extract_env__ on the Python side
    let ptr_val = match py_env.call_method0("__extract_env__") {
        Ok(val) => val,
        Err(_) => {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Object must implement __extract_env__ method",
            ));
        }
    };
    // extract the usize
    let ptr: usize = ptr_val.extract()?;
    // turn it back into a &Box<dyn Env>
    unsafe { Ok(&*(ptr as *const Box<dyn Env>)) }
}


#[pyfunction(name = "solve")]
pub fn solve_py(
    py_env: &Bound<'_, PyAny>,
    policy: &PyPolicy,
    deterministic: bool,
    num_searches: usize,
    num_mcts_searches: usize,
    C: f32,
    max_expand_depth: usize) -> PyResult<((f32, f32), Vec<usize>)> {
        let env_ref = get_env(py_env)?;
        Ok(solve(env_ref, &*policy.policy, deterministic, num_searches, num_mcts_searches, C, max_expand_depth))
}


#[pyfunction(name = "evaluate")]
pub fn evaluate_py(py_env: &Bound<'_, PyAny>,
    policy: &PyPolicy,
    num_episodes: usize,
    deterministic: bool,
    num_searches: usize,
    num_mcts_searches: usize,
    seed: usize,          // unused for now
    C: f32,
    max_expand_depth: usize,
    num_cores: usize) -> PyResult<(f32, f32)> {
    let env_ref = get_env(py_env)?;
    Ok(evaluate(env_ref, &*policy.policy, num_episodes, deterministic, num_searches, num_mcts_searches, seed, C, max_expand_depth, num_cores).map_err(MyError::from)?)
}