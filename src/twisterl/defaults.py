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

import copy
import torch

# Policy

POLICY_CONFIG = {
    "embedding_size": 512,
    "common_layers": [256],
    "policy_layers": [],
    "value_layers": [],
}


# Evaluation

BASE_EVAL_CONFIG = {
    "num_episodes": 100,
    "deterministic": True,
    "num_searches": 1,
    "num_mcts_searches": 0,
    "seed": 0,
    "num_cores": 32,
    "C": 1.41,
    "max_expand_depth": 1,
}

EVALS_CONFIG = {
    "ppo_deterministic": {
        "deterministic": True,
        "num_searches": 1,
    },
    "ppo_1": {
        "deterministic": False,
        "num_searches": 1,
    },
    "ppo_10": {
        "deterministic": False,
        "num_searches": 10,
    },
    "mcts_100": {
        "deterministic": True,
        "num_searches": 1,
        "num_mcts_searches": 100,
    },
}


# Training

PPO_CONFIG = {
    # Collect params
    "collecting": {
        "num_cores": 32,
        "num_episodes": 512 * 32,
        "lambda": 0.995,
        "gamma": 0.995,
    },
    # Train params
    "training": {
        "num_epochs": 10,
        "vf_coef": 0.8,
        "ent_coef": 0.01,  # 0.05,
        "clip_ratio": 0.1,
        "normalize_advantage": False,
    },
    # Optimizer
    "optimizer": {"lr": 0.0003},
}

AZ_CONFIG = {
    # Collect params
    "collecting": {
        "num_cores": 32,
        "num_episodes": 512,
        "num_mcts_searches": 1000,
        "C": 1.41,
        "max_expand_depth": 1,
        "seed": 123,
    },
    # Train params
    "training": {
        "num_epochs": 10,
    },
    # Optimizer
    "optimizer": {"lr": 0.0003},
}

ALGO_CONFIG = {"PPO": PPO_CONFIG, "AZ": AZ_CONFIG}

# Learning

LEARNING_CONFIG = {"diff_threshold": 0.85, "diff_metric": "ppo_deterministic"}


# Logging and checkpoints

LOGGING_CONFIG = {
    "log_freq": 1,
    "checkpoint_freq": 10,
}


def make_config(algo_name, input_config):
    input_config = copy.deepcopy(input_config)
    conf = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "policy": copy.deepcopy(POLICY_CONFIG),
        "evals": copy.deepcopy(EVALS_CONFIG),
        "learning": copy.deepcopy(LEARNING_CONFIG),
        "logging": copy.deepcopy(LOGGING_CONFIG),
        **copy.deepcopy(ALGO_CONFIG[algo_name]),
    }
    conf.update(input_config)

    for k in conf["evals"].keys():
        tmp = conf["evals"][k]
        conf["evals"][k] = copy.deepcopy(BASE_EVAL_CONFIG)
        conf["evals"][k].update(tmp)

    return conf
