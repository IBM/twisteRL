# Grid World Example

This example shows how to expose a simple Rust environment to Python using PyO3. The environment consists of a rectangular grid where an agent must reach a randomly placed goal while avoiding one or more traps.

The `difficulty` parameter limits how far the goal can be from the starting position in Manhattan distance. A difficulty of `3` means the goal is placed between 0 and 3 moves away from the agent at the start of an episode. Actions `0-3` correspond to up, down, left and right.

## Build

Compile the module with `maturin develop` or `python -m pip install -e .` if you have [maturin](https://github.com/PyO3/maturin) installed.

## Usage

```python
import grid_world

# width, height, maximum steps, difficulty, number of traps
env = grid_world.GridWorld(4, 4, 20, 3)
print("Start:", env.get_state())

# move diagonally down (action 1)
env.step(1)
print("After step:", env.get_state())
print("Reached goal?", env.at_goal())
```