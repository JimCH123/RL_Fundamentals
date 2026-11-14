import sys
from pathlib import Path

import numpy as np

# The course environment is a git submodule without packaging, so make it importable here.
ENV_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "DungeonMazeWorld"
sys.path.insert(0, str(ENV_ROOT))

from envs.simple_dungeonworld_env import DungeonMazeEnv  # noqa: E402

dim = 10
max_step = 100
num_actions = 3
num_directions = 4


def epsilon_greedy(epsilon, q_table, obs):
    x, y, d = obs_to_state(obs)
    if np.random.rand() < epsilon:
        action = np.random.randint(num_actions)
    else:
        action = np.argmax(q_table[x, y, d])
    return action


def obs_to_state(obs):
    pos = obs["robot_position"]
    d = obs["robot_direction"]
    x = int(pos[0])
    y = int(pos[1])
    return x, y, int(d)


def is_terminal_state(state, grid_size):
    x, y, d = state
    return x == (grid_size - 2) and y == (grid_size - 2)
