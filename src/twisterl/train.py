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


import argparse
import json
import shutil

from twisterl.utils import prepare_algorithm, load_config


def main(config_path: str, run_path: str, load_checkpoint_path: str, num_steps: int):
    # Make run_path if None
    if run_path is None:
        run_path = f"runs/{config_path.split('/')[-1][:-5]}"

    # Load base config
    config = load_config(config_path)

    # Prepare algorithm
    algorithm = prepare_algorithm(config, run_path, load_checkpoint_path)

    # Save config files to run_path
    shutil.copyfile(config_path, f"{run_path}/base_config.json")
    json.dump(
        algorithm.config, open(f"{run_path}/full_algorithm_config.json", "w"), indent=4
    )

    # Learn
    algorithm.learn(num_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train RL algorithm.")

    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file (required)"
    )
    parser.add_argument(
        "--run_path",
        type=str,
        required=False,
        default=None,
        help="Path to save checkpoings and tensorboard (optional)",
    )
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        required=False,
        default=None,
        help="Path to checkpoint to load (optional)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        required=False,
        default=int(1e9),
        help="Number of training steps (optional)",
    )

    # Parse arguments
    args = parser.parse_args()
    main(
        config_path=args.config,
        run_path=args.run_path,
        load_checkpoint_path=args.load_checkpoint_path,
        num_steps=args.num_steps,
    )
