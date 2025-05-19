# -*- coding: utf-8 -*-

# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import time

from abc import abstractmethod
from functools import wraps

import torch
from torch.utils.tensorboard import SummaryWriter
from twisterl.defaults import make_config

from twisterl import twisterl_rs
evaluate = twisterl_rs.collector.evaluate
solve = twisterl_rs.collector.solve

from loguru import logger


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter_ns()
        result = func(*args, **kwargs)
        elapsed_time = (time.perf_counter_ns() - t0) / 1e9  # Convert nanoseconds to seconds
        return result, elapsed_time
    return wrapper


class Algorithm:
    def __init__(self, env, policy, config, run_path=None):
        self.run_path = run_path
        
        # Setup tensorboard writer
        self.tb_writer = SummaryWriter(run_path) if self.run_path is not None else None

        # Init env
        self.env = env
        self.obs_size = policy.obs_size
        self.num_actions = self.env.num_actions()
        
        # Make full config
        self.config = make_config(type(self).__name__, config)

        # Make policy
        self.policy = policy.to(self.config["device"])
        self.policy.device = self.config["device"]
        self.rs_pol = self.policy.to_rust()

        # Make optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), **self.config["optimizer"])

    @timed
    @abstractmethod
    def data_to_torch(self, data):
        """Receives data in python and returns the data in torch, in the format the `train` method needs."""
        pass

    @timed
    @abstractmethod
    def train_step(self, torch_data):        
        """Receives data torch, performs one step of gradient descent, and returns any relevant metrics."""
        pass

    @timed
    def train(self, torch_data):
        losses = []
        for epoch in range(self.config["training"]['num_epochs']):
            step_losses, step_time = self.train_step(torch_data)
            losses.append(step_losses)
        # Returning the final loss for now
        return losses[-1]

    @timed
    def sync_rs_policy(self):
        """Updates self.rs_pol with current python policy."""
        self.rs_pol = self.policy.to_rust()

    @timed
    def evaluate(self, kwargs):
        """Receives python-rust object and evaluates the policy with some parameters."""
        return evaluate(self.env, self.rs_pol, **kwargs)
    
    @timed
    def collect(self):
        """Collects data and returns the dataset"""
        return self.collector.collect(self.env, self.rs_pol)

    @timed
    def learn_step(self):
        times_dict = {}
        bench_dict = {}
        train_dict = {}

        # Policy to rust
        _, times_dict["to_rust"]  = self.sync_rs_policy()

        # Run eval (multiple evals)
        bench_dict["successes"] = dict()
        bench_dict["rewards"] = dict()
        for eval_name, eval_kwargs in self.config["evals"].items():
            (bench_dict["successes"][eval_name], bench_dict["rewards"][eval_name]), times_dict[f"eval_{eval_name}"] = self.evaluate(eval_kwargs)
        bench_dict["difficulty"] = self.env.difficulty
        bench_dict["success"] = bench_dict["successes"][self.config["learning"]["diff_metric"]]
        bench_dict["reward"] = bench_dict["rewards"][self.config["learning"]["diff_metric"]]

        # Collect data
        data, times_dict["collect"] = self.collect()

        # Transform data and move to torch/GPU
        torch_data, times_dict["data_to_torch"] = self.data_to_torch(data)

        # Train
        self.policy.train()
        train_dict, times_dict["train"] = self.train(torch_data)

        # Return metrics
        return times_dict, bench_dict, train_dict


    def learn(self, num_steps, best_metrics=None):
        # Init best metrics with a benchmark
        if best_metrics is None:
            (success, reward), _ = self.evaluate(self.config["evals"][self.config["learning"]["diff_metric"]])
            best_metrics = (self.env.difficulty, success, reward)

        # Loop for the given number of iterations
        for iteration in range(num_steps):
            (times_dict, bench_dict, train_dict), total_step_time = self.learn_step()
            times_dict["total"] = total_step_time
            current_metrics = (bench_dict["difficulty"], bench_dict["success"], bench_dict["reward"])
            improved = current_metrics >= best_metrics
            if improved:
                best_metrics = current_metrics

            # Maybe increase difficulty
            if (bench_dict["success"] >= self.config["learning"]["diff_threshold"]) and self.env.difficulty < self.config["learning"]["diff_max"]:
                self.env.difficulty += 1
                logger.info(f"({self.env.difficulty}/{iteration}) Diff increased to {self.env.difficulty}, {current_metrics}")

            # Pring logs
            if (self.config["logging"]["log_freq"] > 0) and (iteration % self.config["logging"]["log_freq"] == 0):
                logger.info(f"({self.env.difficulty}/{iteration}) {str(bench_dict)} | {str(times_dict)}")

            # Save checkpoints
            if self.run_path and (self.config["logging"]["checkpoint_freq"] > 0) and ((iteration % self.config["logging"]["checkpoint_freq"] == 0)):
                torch.save(self.policy.state_dict(), open(f"{self.run_path}/checkpoint_last.pt", "wb"))
            
            if self.run_path and improved:
                torch.save(self.policy.state_dict(), open(f"{self.run_path}/checkpoint_best.pt", "wb"))
                logger.info(f"({self.env.difficulty}/{iteration}) Improved, saved checkpoint!")

            # Write to tensorboard
            # Benchmarks
            if (self.tb_writer is not None) and ((iteration % self.config["logging"]["log_freq"] == 0)):
                for bname in ["difficulty", "success", "reward"]:
                    self.tb_writer.add_scalar(f"Benchmark/{bname}", bench_dict[bname], iteration, new_style=True)
                self.tb_writer.add_scalars(f"Benchmark/Detail/Success", bench_dict["successes"], iteration)
                self.tb_writer.add_scalars(f"Benchmark/Detail/Reward", bench_dict["rewards"], iteration)

                # Algorithm
                self.tb_writer.add_scalars("Losses", train_dict, iteration)

                # Times
                self.tb_writer.add_scalars("Times", times_dict, iteration)

    def solve(self, state, deterministic=False, num_searches=1, num_mcts_searches=0, C=(2**0.5), max_expand_depth=1):
        self.env.set_state(state)

        # Then solve 
        (success, reward), actions = solve(
            self.env,
            self.rs_pol, 
            deterministic=deterministic, 
            num_searches=num_searches, 
            num_mcts_searches=num_mcts_searches, 
            C=C, 
            max_expand_depth=max_expand_depth,
        )

        if success:
            return actions