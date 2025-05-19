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

import torch
import numpy as np

from twisterl.rl.algorithm import Algorithm, timed
from twisterl import twisterl_rs

class AZ(Algorithm):
    def __init__(self, env, policy, config, run_path=None):
        super().__init__(env, policy, config, run_path)
        self.collector = twisterl_rs.collector.AZCollector(**self.config["collecting"])

    @timed
    def data_to_torch(self, data):
        obs, probs, vals = data.obs, data.logits, data.additional_data["remaining_values"]

        np_obs = np.zeros((len(obs), self.obs_size), dtype=float)
        for i, obs_i in enumerate(obs):
            np_obs[i, obs_i] = 1.0
        
        pt_obs = torch.tensor(np_obs, dtype=torch.float, device=self.config["device"])
        pt_probs = torch.tensor(probs, dtype=torch.float, device=self.config["device"])
        pt_vals = torch.tensor(vals, dtype=torch.float, device=self.config["device"]).unsqueeze(1)

        return pt_obs, pt_probs, pt_vals 

    @timed
    def train_step(self, torch_data):        
        pt_obs, pt_probs, pt_vals = torch_data

        pred_probs, pred_vals = self.policy(pt_obs)
        
        policy_loss = torch.nn.functional.cross_entropy(pred_probs, pt_probs)
        value_loss = torch.nn.functional.mse_loss(pred_vals, pt_vals)
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

        return {
            "value": value_loss.item(),
            "policy": policy_loss.item(),
            "total": loss.item(),
        }
