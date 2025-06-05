// Now create a Python-facing proxy struct using PyO3
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::collector::collector::{Collector, CollectedData};
use crate::collector::ppo::PPOCollector;
use crate::collector::az::AZCollector;
use crate::nn::policy::Policy;
use crate::python_interface::env::PyBaseEnv;


#[pyclass(name="CollectedData")]
pub struct PyCollectedData {
    /// The internal Rust implementation
    inner: CollectedData,
}

#[pymethods]
impl PyCollectedData {
    /// Construct a new PyCollectedData from raw rollout vectors.
    #[new]
    pub fn new(
        obs: Vec<Vec<usize>>,
        logits: Vec<Vec<f32>>,
        values: Vec<f32>,
        rewards: Vec<f32>,
        actions: Vec<usize>,
    ) -> Self {
        PyCollectedData {
            inner: CollectedData::new(obs, logits, values, rewards, actions),
        }
    }

    /// Merge another PyCollectedData into this one by appending all vectors.
    pub fn merge(&mut self, other: &PyCollectedData) {
        self.inner.merge(&other.inner);
    }

    // Getter and setter methods for the Python interface
    
    #[getter]
    fn get_obs(&self) -> Vec<Vec<usize>> {
        self.inner.obs.clone()
    }
    
    #[setter]
    fn set_obs(&mut self, obs: Vec<Vec<usize>>) {
        self.inner.obs = obs;
    }
    
    #[getter]
    fn get_logits(&self) -> Vec<Vec<f32>> {
        self.inner.logits.clone()
    }
    
    #[setter]
    fn set_logits(&mut self, logits: Vec<Vec<f32>>) {
        self.inner.logits = logits;
    }
    
    #[getter]
    fn get_values(&self) -> Vec<f32> {
        self.inner.values.clone()
    }
    
    #[setter]
    fn set_values(&mut self, values: Vec<f32>) {
        self.inner.values = values;
    }
    
    #[getter]
    fn get_rewards(&self) -> Vec<f32> {
        self.inner.rewards.clone()
    }
    
    #[setter]
    fn set_rewards(&mut self, rewards: Vec<f32>) {
        self.inner.rewards = rewards;
    }
    
    #[getter]
    fn get_actions(&self) -> Vec<usize> {
        self.inner.actions.clone()
    }
    
    #[setter]
    fn set_actions(&mut self, actions: Vec<usize>) {
        self.inner.actions = actions;
    }
    
    #[getter]
    fn get_additional_data(&self) -> HashMap<String, Vec<f32>> {
        self.inner.additional_data.clone()
    }
    
    #[setter]
    fn set_additional_data(&mut self, additional_data: HashMap<String, Vec<f32>>) {
        self.inner.additional_data = additional_data;
    }
    
    // Add a method to access specific additional data by key
    pub fn get_additional_data_item(&self, key: &str) -> Option<Vec<f32>> {
        self.inner.additional_data.get(key).cloned()
    }
    
    // Add a method to set specific additional data by key
    pub fn set_additional_data_item(&mut self, key: String, value: Vec<f32>) {
        self.inner.additional_data.insert(key, value);
    }
}

#[pyclass(subclass)]
pub struct PyBaseCollector {
    collector: Box<dyn Collector>,
}

#[pymethods]
impl PyBaseCollector {
    // Collects Data
    fn collect(&self, env: PyRef<PyBaseEnv>, policy: &Policy) -> PyCollectedData{
        let collected_data = self.collector.collect(&env.env, policy);
        PyCollectedData { inner: collected_data }
    }
}

#[pyclass(name="PPOCollector", extends=PyBaseCollector)]
pub struct PyPPOCollector {}

#[pymethods]
impl PyPPOCollector {
    #[new]
    pub fn new(
        num_episodes: usize,
        gamma: f32,
        lambda: f32,
        num_cores: usize,
    ) -> (Self, PyBaseCollector) {
        let collector = Box::new(PPOCollector::new(num_episodes, gamma, lambda, num_cores));
        (PyPPOCollector {}, PyBaseCollector { collector: collector })
        // (PyPPOCollector { collector: collector.clone() }, PyBaseCollector { collector: collector })
    }
}

#[pyclass(name="AZCollector", extends=PyBaseCollector)]
pub struct PyAZCollector {}

#[pymethods]
impl PyAZCollector {
    #[new]
    pub fn new(
        num_episodes: usize,
        num_mcts_searches: usize,
        C: f32,
        max_expand_depth: usize,
        num_cores: usize,
    ) -> (Self, PyBaseCollector) {
        let collector = Box::new(AZCollector::new(num_episodes, num_mcts_searches, C, max_expand_depth, num_cores));
        (PyAZCollector { }, PyBaseCollector { collector: collector })
    }
}
