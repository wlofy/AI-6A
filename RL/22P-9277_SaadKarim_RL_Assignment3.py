#import all needed libraries for this assignment

import numpy as np #For storing Q Table Values
import gymnasium as gym # For Frozen Lake environment
import matplotlib.pyplot as plt # For Policy Grid Evaluation
import imageio # Making GIFs for each algorithm's episode to be included into the report
import time #keeping track of time taken to train
from tqdm import tqdm #progress bar for training of each episode (looks cleaner)


# Setting up Environment
env = gym.make("FrozenLake-v1", render_mode = 'rgb_array')
env_unwrapped = env.unwrapped
n_states = env.observation_space.n
n_actions = env.action_space.n


# Hyperparameters
N_EPISODES = 10000
GAMMA = 0.99  # Discounting factor
ALPHA = 0.1   # Learning rate
EPSILON = 0.1  # Fixed epsilon for epsilon-greedy exploration
ACTION_MAP = {0: "←", 1: "↓", 2: "→", 3: "↑"} # will be used to make an action map for optimal actions

#As I will be making functions for each algorithm, I first need to define a function for epsilon greedy as I will be using it 

def epsilon_greedy_policy(Q, state):
    if np.random.rand() < EPSILON:
        return np.random.choice(n_actions)  # Explore
    else:
        return np.argmax(Q[state])  # Exploit
    

# Monte Carlo Implementation
def monte_carlo_control():
    Q = np.zeros((n_states, n_actions))# initialzing a Q table for each state and actions possible in this grid
    returns_count = np.zeros((n_states, n_actions))# counting the number of times a state has been visited initially all 0
    returns = []# creating an empty list of all returns
    successes = 0# count of how many times has the agent successfully reached the goal state

    start_time = time.time() # tracking time taken for experimenting
    for episode in tqdm(range(N_EPISODES), desc="Monte Carlo Control"):#setting up a progress bar of total episodes done(10000 each algorithm)
        episode_data = []#storing episode data for each episode
        state, _ = env.reset()
        done = False#check for whether benchmark has been met
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
        for state, action, reward in reversed(episode_data):
            G = GAMMA * G + reward
            returns_count[state, action] += 1
            Q[state, action] += (G - Q[state, action]) / returns_count[state, action]

    training_time = time.time() - start_time
    policy = np.argmax(Q, axis=1)
    return Q, policy, returns, successes / N_EPISODES, training_time

# SARSA Implementation
def sarsa():
    Q = np.zeros((n_states, n_actions)) #initializing a Q table for the number of total states and actions in the Frozen lake environment
    returns = []#initializing an empty list for all returns
    successes = 0#number of times the agent successfuly reached the target state
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

# Q Learning Implementation
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

def visualize_policy(policy, title, filename): #suggested by AI, to visualize it through an matplot lib, looked much cleaner so I decided to implement it
    policy_grid = policy.reshape(4,4)
    fig, ax = plt.subplots()
    ax.matshow(np.zeros((4, 4)), cmap="Greys")

    for i in range(4):
        for j in range(4):
            state = i*4+j
            action = policy[state]
            ax.text(j, i, ACTION_MAP[action], ha = "center", va = "center", fontsize = 12)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.savefig(filename)
    plt.close()

def animate_episodes(policy, filename, num_of_episodes = 10):
    frames = []
    for _ in range(num_of_episodes):
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

