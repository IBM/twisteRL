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

class PPO(Algorithm):
    def __init__(self, env, policy, config, run_path=None):
        super().__init__(env, policy, config, run_path)
        self.collector = twisterl_rs.collector.PPOCollector(**self.config["collecting"])

    @timed
    def data_to_torch(self, data):
        obs, logits, vals, rews, acts, rets, advs = (
            data.obs,
            data.logits,
            data.values,
            data.rewards,
            data.actions,
            data.additional_data["rets"],
            data.additional_data["advs"],
        )
        np_obs = np.zeros((len(obs), self.obs_size), dtype=float)
        for i, obs_i in enumerate(obs):
            np_obs[i, obs_i] = 1.0

        pt_obs = torch.tensor(np_obs, dtype=torch.float, device=self.config["device"])
        pt_logits = torch.tensor(logits, dtype=torch.float, device=self.config["device"])
        # pt_vals = torch.tensor(vals, dtype=torch.float, device=self.config["device"])
        # pt_rews = torch.tensor(rews, dtype=torch.float, device=self.config["device"])
        pt_acts = torch.tensor(acts, dtype=torch.long, device=self.config["device"])
        pt_rets = torch.tensor(rets, dtype=torch.float, device=self.config["device"])
        pt_advs = torch.tensor(advs, dtype=torch.float, device=self.config["device"])

        with torch.no_grad():
            if self.config["training"].get("normalize_advantage", False):
                pt_advs = (pt_advs - pt_advs.mean()) / (pt_advs.std() + 1e-8)
            pt_log_probs = torch.distributions.Categorical(logits=pt_logits).log_prob(pt_acts)

        return pt_obs, pt_log_probs, pt_acts, pt_advs, pt_rets

    @timed
    def train_step(self, torch_data):        
        pt_obs, pt_log_probs, pt_acts, pt_advs, pt_rets = torch_data

        # Forward pass to get logits and values
        pred_logits, pred_vals = self.policy(pt_obs)

        # Get log-probabilities of the actions actually taken
        dist = torch.distributions.Categorical(logits=pred_logits)
        pred_log_probs = dist.log_prob(pt_acts)

        # Entropy of the action distribution
        entropy = dist.entropy()
        entropy_loss = entropy.mean()

        # Compute the ratio between new and old policy probabilities
        ratios = torch.exp(pred_log_probs - pt_log_probs)

        # Compute the PPO clipped objective
        surr1 = ratios * pt_advs
        surr2 = torch.clamp(ratios, 1 - self.config["training"]["clip_ratio"], 1 + self.config["training"]["clip_ratio"]) * pt_advs
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss
        value_loss = torch.nn.functional.mse_loss(pred_vals, pt_rets.unsqueeze(1))

        # Combined loss
        loss = policy_loss + self.config["training"]["vf_coef"] * value_loss - self.config["training"]["ent_coef"] * entropy_loss

        # Update the policy via gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "value": value_loss.item(),
            "policy": policy_loss.item(),
            "entropy": entropy_loss.item(),
            "total": loss.item(),
        }
