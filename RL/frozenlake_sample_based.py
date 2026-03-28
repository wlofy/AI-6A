import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
from tqdm import tqdm

# Environment setup
env = gym.make("FrozenLake-v1", render_mode="rgb_array")  # Default: is_slippery=True
env_unwrapped = env.unwrapped
n_states = env.observation_space.n
n_actions = env.action_space.n

# Hyperparameters
N_EPISODES = 10000
GAMMA = 0.99  # Discount factor
ALPHA = 0.1   # Learning rate
EPSILON = 0.1  # Fixed epsilon for epsilon-greedy exploration
ACTION_MAP = {0: "←", 1: "↓", 2: "→", 3: "↑"}

# Utility functions
def epsilon_greedy_policy(Q, state):
    if np.random.random() < EPSILON:
        return env.action_space.sample()
    return np.argmax(Q[state])

# Monte Carlo Control
def monte_carlo_control():
    Q = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))
    returns = []
    successes = 0
    
    start_time = time.time()
    for episode in tqdm(range(N_EPISODES), desc="Monte Carlo"):
        episode_data = []
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        # Generate episode
        while not done:
            action = epsilon_greedy_policy(Q, state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state
            total_reward += reward
            if reward == 1.0:
                successes += 1
        
        returns.append(total_reward)
        
        # Update Q-values
        G = 0
        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            G = GAMMA * G + reward
            returns_count[state, action] += 1
            Q[state, action] += (G - Q[state, action]) / returns_count[state, action]
    
    training_time = time.time() - start_time
    policy = np.argmax(Q, axis=1)
    return Q, policy, returns, successes / N_EPISODES, training_time

# SARSA
def sarsa():
    Q = np.zeros((n_states, n_actions))
    returns = []
    successes = 0
    
    start_time = time.time()
    for episode in tqdm(range(N_EPISODES), desc="SARSA"):
        state, _ = env.reset()
        action = epsilon_greedy_policy(Q, state)
        done = False
        total_reward = 0
        
        while not done:
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state)
            Q[state, action] += ALPHA * (reward + GAMMA * Q[next_state, next_action] - Q[state, action])
            state = next_state
            action = next_action
            total_reward += reward
            if reward == 1.0:
                successes += 1
        
        returns.append(total_reward)
    
    training_time = time.time() - start_time
    policy = np.argmax(Q, axis=1)
    return Q, policy, returns, successes / N_EPISODES, training_time

# Q-Learning
def q_learning():
    Q = np.zeros((n_states, n_actions))
    returns = []
    successes = 0
    
    start_time = time.time()
    for episode in tqdm(range(N_EPISODES), desc="Q-Learning"):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = epsilon_greedy_policy(Q, state)
            next_state, reward, done, truncated, _ = env.step(action)
            Q[state, action] += ALPHA * (reward + GAMMA * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            total_reward += reward
            if reward == 1.0:
                successes += 1
        
        returns.append(total_reward)
    
    training_time = time.time() - start_time
    policy = np.argmax(Q, axis=1)
    return Q, policy, returns, successes / N_EPISODES, training_time

# Visualize policy as 4x4 grid
def visualize_policy(policy, title, filename):
    policy_grid = policy.reshape(4, 4)
    fig, ax = plt.subplots()
    ax.matshow(np.zeros((4, 4)), cmap="Greys")
    
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            action = policy[state]
            ax.text(j, i, ACTION_MAP[action], ha="center", va="center", fontsize=12)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.savefig(filename)
    plt.close()

# Simulate and animate episodes
def animate_episodes(policy, filename, n_episodes=10):
    frames = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        frames.append(env.render())
        
        while not done:
            action = policy[state]
            next_state, reward, done, truncated, _ = env.step(action)
            frames.append(env.render())
            state = next_state
    
    imageio.mimsave(filename, frames, fps=2)
    env.close()

# Main execution
if __name__ == "__main__":
    # Train algorithms
    mc_Q, mc_policy, mc_returns, mc_success_rate, mc_time = monte_carlo_control()
    sarsa_Q, sarsa_policy, sarsa_returns, sarsa_success_rate, sarsa_time = sarsa()
    ql_Q, ql_policy, ql_returns, ql_success_rate, ql_time = q_learning()
    
    # Visualize policies
    visualize_policy(mc_policy, "Monte Carlo Policy", "mc_policy.png")
    visualize_policy(sarsa_policy, "SARSA Policy", "sarsa_policy.png")
    visualize_policy(ql_policy, "Q-Learning Policy", "ql_policy.png")
    
    # Animate episodes
    animate_episodes(mc_policy, "mc_animation.gif")
    animate_episodes(sarsa_policy, "sarsa_animation.gif")
    animate_episodes(ql_policy, "ql_animation.gif")
    
    # Compute metrics
    metrics = {
        "Monte Carlo": {
            "Avg Return": np.mean(mc_returns),
            "Success Rate": mc_success_rate * 100,
            "Training Time (s)": mc_time
        },
        "SARSA": {
            "Avg Return": np.mean(sarsa_returns),
            "Success Rate": sarsa_success_rate * 100,
            "Training Time (s)": sarsa_time
        },
        "Q-Learning": {
            "Avg Return": np.mean(ql_returns),
            "Success Rate": ql_success_rate * 100,
            "Training Time (s)": ql_time
        }
    }
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for algo, metric in metrics.items():
        print(f"\n{algo}:")
        for key, value in metric.items():
            print(f"  {key}: {value:.2f}")