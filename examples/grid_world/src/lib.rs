use pyo3::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::cmp;

use twisterl::rl::env::Env;
use twisterl::python_interface::env::{PyBaseEnv, get_env_ref, get_env_mut};

#[derive(Clone)]
pub struct GridWorld {
    width: usize,
    height: usize,
    max_steps: usize,
    difficulty: usize,
    steps_left: usize,
    agent: (usize, usize),
    goal: (usize, usize),
    trap: (usize, usize),
}

impl GridWorld {
    pub fn new(
        width: usize,
        height: usize,
        max_steps: usize,
        difficulty: usize,
    ) -> Self {
        let mut env = Self {
            width,
            height,
            max_steps,
            difficulty: cmp::min(width+height, difficulty),
            steps_left: max_steps,
            agent: (0, 0),
            goal: (0, 0),
            trap: (0,0)
        };
        // env.reset();
        env
    }

    fn random_pos(&self) -> (usize, usize) {
        let mut rng = thread_rng();
        let range = Uniform::new(0, self.width * self.height);
        let idx = range.sample(&mut rng);
        (idx % self.width, idx / self.width)
    }

    fn random_pos_near(&self, center: (usize, usize), max_dist: usize) -> (usize, usize) {
        let mut rng = thread_rng();
        let mut candidates = Vec::new();
        for x in 0..self.width {
            for y in 0..self.height {
                let dist = ((x as isize - center.0 as isize).abs()
                    + (y as isize - center.1 as isize).abs()) as usize;
                if dist <= max_dist {
                    candidates.push((x, y));
                }
            }
        }
        *candidates
            .choose(&mut rng)
            .expect("GridWorld: no valid positions found")
    }

    pub fn at_goal(&self) -> bool { self.agent == self.goal }
    pub fn at_trap(&self) -> bool { self.agent == self.trap }

    pub fn get_positions(&self) -> ((usize, usize), (usize, usize), (usize, usize)) {
        (self.agent, self.goal, self.trap)
    }

    pub fn get_state(&self) -> Vec<usize> {
        let mut board = vec![0usize; self.width * self.height];
        let idx = |p: (usize, usize)| -> usize { p.1 * self.width + p.0 };
        board[idx(self.goal)] = 2;
        board[idx(self.trap)] = 3;
        board[idx(self.agent)] = 1;
        board
    }
}

impl Env for GridWorld {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn num_actions(&self) -> usize { 4 }

    fn obs_shape(&self) -> Vec<usize> { vec![self.width*self.height, self.width*self.height] }

    fn set_difficulty(&mut self, difficulty: usize) {
        self.difficulty = cmp::min(self.width+self.height, difficulty);
    }

    fn get_difficulty(&self) -> usize {
        self.difficulty
    }

    fn set_state(&mut self, board: Vec<i64>) {
        let pos = |idx: usize| -> (usize, usize) { (idx % self.width, idx / self.width) };

        for (idx, &value) in board.iter().enumerate() {
            match value {
                1 => self.agent = pos(idx),
                2 => self.goal = pos(idx),
                3 => self.trap = pos(idx),
                _ => {} // 0 or any other value = empty space, do nothing
            }
        }
        self.steps_left = self.max_steps;
    }

    fn reset(&mut self) {
        self.agent = self.random_pos();
        self.goal = loop {
            let g = self.random_pos_near(self.agent, self.difficulty);
            if g != self.agent { break g; }
        };
        self.trap = loop {
            let t = self.random_pos();
            if t != self.agent && t != self.goal { break t; }
        };
        self.steps_left = self.max_steps;
    }

    fn step(&mut self, action: usize) {
        match action {
            0 if self.agent.1 > 0 => self.agent.1 -= 1,        // up
            1 if self.agent.1 + 1 < self.height => self.agent.1 += 1, // down
            2 if self.agent.0 > 0 => self.agent.0 -= 1,        // left
            3 if self.agent.0 + 1 < self.width => self.agent.0 += 1, // right
            _ => {}
        }
        self.steps_left = self.steps_left.saturating_sub(1);
    }

    fn masks(&self) -> Vec<bool> {
        vec![
            self.agent.1 > 0,
            self.agent.1 + 1 < self.height,
            self.agent.0 > 0,
            self.agent.0 + 1 < self.width,
        ]
    }

    fn is_final(&self) -> bool {
        self.steps_left == 0 || self.at_goal() || self.at_trap()
    }

    fn reward(&self) -> f32 {
        if self.at_goal() { 1.0 } else if self.at_trap() { -0.5 } else {
            if self.steps_left == 0 { -0.5 } else { -0.5/(self.steps_left as f32) }
        }
    }

    fn observe(&self) -> Vec<usize> {
        self.get_state().iter().enumerate().map(|(i, v)| i * self.height * self.width + v).collect()  
    }
}

#[pyclass(name="GridWorld", extends=PyBaseEnv)]
pub struct PyGridWorldEnv;

#[pymethods]
impl PyGridWorldEnv {
    #[new]
    pub fn new(width: usize, height: usize, max_steps: usize, difficulty: usize) -> (Self, PyBaseEnv) {
        let env = GridWorld::new(width, height, max_steps, difficulty);
        let env = Box::new(env);
        (PyGridWorldEnv, PyBaseEnv { env })
    }

    pub fn get_positions(
        slf: PyRef<'_, Self>,
    ) -> PyResult<((usize, usize), (usize, usize), (usize, usize))> {
        let env = get_env_ref::<GridWorld>(slf.as_ref())?;
        Ok(env.get_positions())
    }

    pub fn at_goal(slf: PyRef<'_, Self>) -> PyResult<bool> {
        let env = get_env_ref::<GridWorld>(slf.as_ref())?;
        Ok(env.at_goal())
    }

    pub fn at_trap(slf: PyRef<'_, Self>) -> PyResult<bool> {
        let env = get_env_ref::<GridWorld>(slf.as_ref())?;
        Ok(env.at_trap())
    }

    pub fn get_state(slf: PyRef<'_, Self>) -> PyResult<Vec<usize>> {
        let env = get_env_ref::<GridWorld>(slf.as_ref())?;
        Ok(env.get_state())
    }

    pub fn set_state(mut slf: PyRefMut<'_, Self>, board: Vec<i64>) -> PyResult<()> {
        let env = get_env_mut::<GridWorld>(slf.as_mut())?;
        env.set_state(board);
        Ok(())
    }
}

#[pymodule]
fn grid_world(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGridWorldEnv>()?;
    Ok(())
}