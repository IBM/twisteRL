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
from twisterl import twisterl


def sequential_to_rust(seq):
    """Exports a Sequential of Linears & ReLUs to rust"""
    py_layers = list(seq)
    rs_linears = []

    for layer_idx in range(len(py_layers)):
        if type(py_layers[layer_idx]).__name__ == "Linear":
            rs_linears.append(
                twisterl.nn.Linear(
                    py_layers[layer_idx]
                    .weight.cpu()
                    .detach()
                    .numpy()
                    .T.flatten()
                    .tolist(),
                    py_layers[layer_idx].bias.cpu().detach().numpy().tolist(),
                    ((layer_idx + 1) < len(py_layers))
                    and (type(py_layers[layer_idx + 1]).__name__ == "ReLU"),
                )
            )
        elif type(py_layers[layer_idx]).__name__ not in ("ReLU", "Linear"):
            raise TypeError(
                f"Layer of type {type(py_layers[layer_idx]).__name__} not supported in Sequential."
            )

    return twisterl.nn.Sequential(rs_linears)


def embeddingbag_to_rust(eb, obs_shape, conv_dim):
    """Exports an EmbeddingBag module to rust (followed by a ReLU)"""

    if type(eb).__name__ == "Linear":
        return twisterl.nn.EmbeddingBag(
            eb.weight.cpu().detach().numpy().T.tolist(),
            (
                eb.bias.cpu().detach().numpy().tolist()
                if hasattr(eb, "bias")
                else ([0.0] * eb.weight.shape[0])
            ),
            True,
            obs_shape,
            conv_dim,
        )
    elif type(eb).__name__ == "EmbeddingBag":
        return twisterl.nn.EmbeddingBag(
            eb.weight.cpu().detach().numpy().tolist(),
            [0.0] * eb.weight.shape[1],
            True,
            obs_shape,
            conv_dim,
        )
    elif type(eb).__name__ == "Conv1d":
        return twisterl.nn.EmbeddingBag(
            eb.weight.squeeze(2).cpu().detach().numpy().T.tolist(),
            [0.0] * (eb.weight.shape[0] * obs_shape[1 - conv_dim]),
            True,
            obs_shape,
            conv_dim,
        )
    else:
        raise TypeError(
            f"Layer of type {type(eb).__name__} not supported as EmbeddingBag."
        )


def make_sequential(in_size, layers, final_relu=True):
    layer_list = []

    for out_size in layers:
        layer_list.append(torch.nn.Linear(in_size, out_size))
        layer_list.append(torch.nn.ReLU())
        in_size = out_size

    if not final_relu:
        layer_list = layer_list[:-1]

    return torch.nn.Sequential(*layer_list)
