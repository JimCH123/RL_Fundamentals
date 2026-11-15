from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl_fundamentals.sarsa import gamma, learning_rate, num_episodes, sarsa
from rl_fundamentals.utils import DungeonMazeEnv, dim, num_actions

FIGURE_DIR = Path("plots/sarsa")

if __name__ == "__main__":
    env = DungeonMazeEnv(render_mode=None, grid_size=dim)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


    #---------------------------Analysize for different gamma--------------------------
    gammas = [0.85 ,0.9, 0.95]
    reward_result = {}
    errors_result = {}
    for g in gammas:
        q_table, q_learning_rewards, avg_error = sarsa(env, num_actions, num_episodes, dim, learning_rate, gamma=g)
        reward_result[g] = q_learning_rewards
        errors_result[g] = avg_error
    print("Training finished!")
    
    plt.figure(figsize=(8, 6))
    for g, error in errors_result.items():
        episodes = range(1, len(error) + 1)
        plt.plot(episodes, error, label=f"gamma = {g}")
    plt.xlabel("Episode")
    plt.ylabel("Average Absolute Q-Value Change")
    plt.title("SARSA Convergence")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(FIGURE_DIR / "convergence_gamma.png", dpi=150, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 6))
    for g, rewards in reward_result.items():
        episodes = range(1, len(rewards) + 1)
        plt.plot(episodes, rewards, label=f"gamma = {g}")
    plt.xlabel("Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.title("SARSA Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / "reward_gamma.png", dpi=150, bbox_inches="tight")
    plt.show()


    #---------------------Analysize for different learning rate-------------------
    alpha = [0.1, 0.5, 0.9]
    reward_result = {}
    for a in alpha:
        q_table, sarsa_rewards, avg_error = sarsa(env, num_actions, num_episodes, dim=dim, learning_rate=a, gamma=gamma)
        reward_result[a] = sarsa_rewards
    print("Training finished!")

    plt.figure(figsize=(8,6))
    for lr, reward in reward_result.items():
        episodes = range(1, len(reward)+1)
        plt.plot(episodes, reward, label=f"lr = {lr}")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("SARSA Reward (Initialisation = zeros)")
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / "reward_learning_rate.png", dpi=150, bbox_inches="tight")
    plt.show()

    #---------------------Train once with the default settings and save the Q table-------------------
    q_table, _, _ = sarsa(env, num_actions, num_episodes, dim, learning_rate, gamma)
    Path("results").mkdir(exist_ok=True)
    np.save("results/sarsa_q_table.npy", q_table)
    print("Saved Q table to results/sarsa_q_table.npy")