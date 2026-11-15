import time
import numpy as np
from rl_fundamentals.utils import DungeonMazeEnv, dim, max_step, obs_to_state


def show_greedy_policy(q_table, seed=124, max_steps=max_step):
    """Open a pygame window and let the trained agent walk the maze greedily (epsilon = 0)."""
    demo_env = DungeonMazeEnv(render_mode="human", grid_size=dim)
    obs, info = demo_env.reset(seed=seed)  # same seed as training -> same maze
    for step in range(max_steps):
        x, y, d = obs_to_state(obs)
        action = int(np.argmax(q_table[x, y, d]))
        obs, reward, done, truncated, info = demo_env.step(action)
        if done:
            print(f"Reached the target in {step + 1} steps")
            break
    else:
        print(f"Did not reach the target within {max_steps} steps")
    time.sleep(2)  # keep the final frame on screen for a moment
    demo_env.close()


if __name__ == "__main__":
    q_table = np.load("results/sarsa_q_table.npy")
    show_greedy_policy(q_table, seed=124)