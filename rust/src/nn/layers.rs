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

use nalgebra::{DMatrix, DVector};

#[derive(Clone)]
pub struct Linear {
    weights: DMatrix<f32>,
    bias: DVector<f32>,
    apply_relu: bool
}


impl Linear {
    pub fn new(weights_vector: Vec<f32>, bias_vector: Vec<f32>, apply_relu: bool) -> Self {
        let weights = DMatrix::from_vec(bias_vector.len(), weights_vector.len() / bias_vector.len(), weights_vector);
        let bias = DVector::from_vec(bias_vector);
        Self { weights, bias, apply_relu }
    }
    
    pub fn forward(&self, input: &DVector<f32>) -> DVector<f32> {
        let mut out = (&self.weights * input) + &self.bias;
        if self.apply_relu {
            out =  out.map(relu);
        }
        out
    }
}

#[derive(Clone)]
pub struct EmbeddingBag {
    vectors: Vec<DVector<f32>>,
    bias: DVector<f32>,
    apply_relu: bool,
    obs_shape: Vec<usize>,
    conv_dim: usize
}

impl EmbeddingBag {
    pub fn new(vec_vectors: Vec<Vec<f32>>, bias_vector: Vec<f32>, apply_relu: bool, obs_shape: Vec<usize>, conv_dim: usize) -> Self {
        let vectors = vec_vectors.into_iter().map(|vec| DVector::from_vec(vec)).collect();
        let bias = DVector::from_vec(bias_vector);
        Self { vectors, bias, apply_relu, obs_shape, conv_dim }
    }

    pub fn forward(&self, input: &Vec<usize>) -> DVector<f32> {
        let mut out = self.bias.clone();
        if self.obs_shape.len() == 1 {
            // This is standard embeddings / linear
            for &i in input.iter() {
                out += &self.vectors[i];
            }
        } else if self.obs_shape.len() == 2 {
            let v_size = self.vectors[0].len();
            // This is conv1d
            for &i in input.iter() {
                // obs_shape[0] is the size of each col (i.e. the number of rows)
                let mut row = i / self.obs_shape[1]; 
                let mut col = i % self.obs_shape[1];

                // If conv_dim is 1 then we swap row and col
                if self.conv_dim == 1 {(row, col) = (col, row);}

                let mut out_slice = out.rows_mut(col * v_size, v_size);
                out_slice += &self.vectors[row];

            }

        } // TODO: add conv2d


        if self.apply_relu {
            out =  out.map(relu);
        }
        out
    }
}

fn relu(x: f32) -> f32 {
   if x > 0.0 { x } else { 0.0 }
}