use pyo3::prelude::*;
use crate::rl::env::Env;
use crate::envs::puzzle::Puzzle;
use crate::python_interface::policy::PyPolicy;
use crate::rl::solve::solve;
use crate::rl::evaluate::evaluate;

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
}


#[pyclass(name="Puzzle", extends=PyBaseEnv)]
pub struct PyPuzzleEnv {
    env: Box<Puzzle>,
}

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
        (PyPuzzleEnv { env: env.clone() }, PyBaseEnv { env: env })
    }

    pub fn solved(&self) -> PyResult<bool> {
        Ok(self.env.solved())
    }

    pub fn get_state(&self) -> PyResult<Vec<usize>> {
        Ok(self.env.get_state())
    }

    pub fn display(&self) -> PyResult<()> {
        Ok(self.env.display())
    }

    pub fn set_position(&mut self, x: usize, y: usize, val: usize) -> PyResult<()> {
        Ok(self.env.set_position(x, y, val))
    }

    pub fn get_position(&self, x: usize, y: usize) -> PyResult<usize> {
        Ok(self.env.get_position(x, y))
    }
}


#[pyfunction(name = "solve")]
pub fn solve_py(py_env: PyRef<PyBaseEnv>,
    policy: &PyPolicy,
    deterministic: bool,
    num_searches: usize,
    num_mcts_searches: usize,
    C: f32,
    max_expand_depth: usize) -> ((f32, f32), Vec<usize>) {
        solve(&py_env.env, &*policy.policy, deterministic, num_searches, num_mcts_searches, C, max_expand_depth)
}


#[pyfunction(name = "evaluate")]
pub fn evaluate_py(py_env: PyRef<PyBaseEnv>,
    policy: &PyPolicy,
    num_episodes: usize,
    deterministic: bool,
    num_searches: usize,
    num_mcts_searches: usize,
    seed: usize,          // unused for now
    C: f32,
    max_expand_depth: usize,
    num_cores: usize) -> (f32, f32) {
    evaluate(&py_env.env, &*policy.policy, num_episodes, deterministic, num_searches, num_mcts_searches, seed, C, max_expand_depth, num_cores)
}