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

pub struct Tree<T> {
    pub nodes: Vec<Node<T>>,
}

pub struct Node<T> {
    idx: usize,
    pub val: T,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
}

impl<T> Node<T> {
    pub fn new(idx: usize, val: T) -> Self {
        Self {
            idx,
            val,
            parent: None,
            children: vec![],
        }
    }
}


impl<T> Tree<T> {
    pub fn new() -> Self {
        Self {nodes: vec![]}
    }

    pub fn new_node(&mut self, val: T) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node::new(idx, val));
        idx
    }

    pub fn add_child_to_node(&mut self, val: T, idx: usize) -> usize {
        let child_idx = self.new_node(val);
        self.nodes[idx].children.push(child_idx);
        self.nodes[child_idx].parent = Some(idx);
        child_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_add_child() {
        let mut tree: Tree<i32> = Tree::new();
        let root = tree.new_node(1);
        let child = tree.add_child_to_node(2, root);

        assert_eq!(root, 0);
        assert_eq!(child, 1);
        assert_eq!(tree.nodes.len(), 2);
        assert_eq!(tree.nodes[root].children, vec![child]);
        assert_eq!(tree.nodes[child].parent, Some(root));
    }
}