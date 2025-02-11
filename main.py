import gymnasium as gym  # Use gymnasium instead of gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from gymnasium.spaces import Box

# Create CartPole environment
env = gym.make("CartPole-v1")

# Initialize Q-table
state_bins = [10, 10, 10, 10]  # Discretization bins for state space
q_table = np.zeros(state_bins + [env.action_space.n])

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 500

# State bounds for discretization
state_bounds = [
    (-2.4, 2.4),  # Cart position
    (-3.0, 3.0),  # Cart velocity
    (-0.5, 0.5),  # Pole angle
    (-2.0, 2.0)   # Pole angular velocity
]

def discretize_state(state):
    """Convert continuous state into discrete bins"""
    state_indices = []
    for i in range(len(state)):
        # Clip state to bounds
        clipped_value = np.clip(state[i], state_bounds[i][0], state_bounds[i][1])
        # Scale to [0, 1] and then to [0, state_bins[i] - 1]
        scaling = (clipped_value - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0])
        index = int(scaling * (state_bins[i] - 1))
        index = min(max(index, 0), state_bins[i] - 1)
        state_indices.append(index)
    return tuple(state_indices)

# Training Loop
rewards_per_episode = []
for episode in range(episodes):
    observation, _ = env.reset()  # Handle new Gym API correctly
    state = discretize_state(observation)
    total_reward = 0
    done = False

    while not done:
        # Epsilon-Greedy Action Selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)  # Ensure Boolean conversion
        next_state = discretize_state(next_observation)

        # Update Q-value using Bellman Equation
        best_next_action = np.max(q_table[next_state])
        q_table[state][action] = (1 - learning_rate) * q_table[state][action] + learning_rate * (
                    reward + discount_factor * best_next_action)

        state = next_state
        total_reward += reward

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)

    if episode % 50 == 0:
        print(f"Episode {episode}: Total Reward = {total_reward}")

# Plot rewards over episodes
window_size = 50
moving_avg = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')
plt.plot(moving_avg)
plt.xlabel("Episodes")
plt.ylabel("Total Reward (Moving Avg)")
plt.title("Q-Learning Performance on CartPole")
plt.show()

env.close()