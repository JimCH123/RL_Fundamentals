import numpy as np

from rl_fundamentals.utils import (
    epsilon_greedy,
    max_step,
    num_directions,
    obs_to_state,
)

gamma = 0.99
epsilon = 0.1
learning_rate = 0.5
num_episodes = 500
RUN = 10


def q_learning(env, num_actions, num_episodes, dim, learning_rate, gamma):
    error_per_run = []
    rewards_per_run = []

    for _ in range(RUN):
        q_table = np.zeros((dim, dim, num_directions, num_actions))
        rewards_per_episode = []
        errors = []

        for episode in range(num_episodes):
            current_epsilon = epsilon * (1 - episode / num_episodes)
            obs, info = env.reset(seed = 124)
            state = obs_to_state(obs)
            cumulated_reward = 0
            updated_pairs = []

            for step in range(max_step):  # cap episode length (the env itself never truncates)
                action = epsilon_greedy(current_epsilon, q_table, obs)
                next_obs, reward, done, truncated, info = env.step(action)
                # The env gives 0 reward for bumping into a wall, so with any gamma "bump forever"
                # beats reaching the goal and the agent gets stuck. Treat a bump as a normal -1 step.
                if reward == 0:
                    reward = -1
                cumulated_reward += reward

                next_state = obs_to_state(next_obs)

                x, y, d = state
                nx, ny, nd = next_state

                best_next_action = np.argmax(q_table[nx, ny, nd])
                old_value = q_table[x, y, d, action]
                q_table[x, y, d, action] += learning_rate * (
                    reward + gamma * q_table[nx, ny, nd, best_next_action]
                    - q_table[x, y, d, action]
                )
                updated_pairs.append((x, y, d, action, old_value))

                state = next_state
                obs = next_obs

                if done or truncated:
                    break

            rewards_per_episode.append(cumulated_reward)

            if updated_pairs:
                diff = np.mean([np.abs(q_table[x, y, d, a] - old_val) for x, y, d, a, old_val in updated_pairs])
            else:
                diff = 0
            errors.append(diff)

        error_per_run.append(errors)
        rewards_per_run.append(rewards_per_episode)

    avg_error = np.mean(error_per_run, axis=0)
    avg_rewards = np.mean(rewards_per_run, axis=0)
    return q_table, avg_rewards, avg_error