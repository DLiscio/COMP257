"""
Assignment 6: Reinforcement Learning
Damien Liscio 301237966
Wednesday, December 4th, 2024
"""
# Import Statements
import gym
import keras
from keras.layers import Input  # type: ignore
import numpy as np
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# Set seed for reproducibility
np.random.seed(66) 
tf.random.set_seed(66)

# Create a simple DQN epsilon policy network with 4 output neurons
env = gym.make("LunarLander-v2", render_mode="rgb_array") #Initialize Lunar Lander environment
env.reset(seed=66) # Environment seed for reproducibility
input_shape = [8] # Model Input Shape (8 features)
n_outputs = 4  # Output shape (discrete actions)

model = keras.models.Sequential([  # Define DQN
    Input(shape=input_shape),  # Input layer
    keras.layers.Dense(32, activation="elu"),  # First hidden layer
    keras.layers.Dense(32, activation ="elu"),  # Second hidden layer
    keras.layers.Dense(n_outputs)  # Output layer
])

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:    
        return np.random.randint(n_outputs)  # Explore random action
    else:
        Q_values = model.predict(state[np.newaxis]) # Choose best action
        return np.argmax(Q_values[0])  # Return best Q-Value

replay_buffer = deque(maxlen=2000)   #Store experiences

def sample_experiences(batch_size):  # Sample a batch of experiences
    indices = np.random.randint(len(replay_buffer), size=batch_size)  # Get indices
    batch = [replay_buffer[index] for index in indices]  # Get experience for each index
    states, actions, rewards, next_states, dones = [  # Store as array
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones   

def play_one_step(env, state, epsilon):  # Execute one step
    action = epsilon_greedy_policy(state, epsilon)  # Use epsilon greed policy
    next_state, reward, terminated, truncated, info = env.step(action) # Take step and unpack experience
    done = terminated or truncated
    replay_buffer.append((state, action, reward, next_state, done))  # Add experience to the replay buffer
    return next_state, reward, done, truncated, info

# Training parameters
batch_size = 32  
discount_factor = 0.95
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.mean_squared_error

def training_step(batch_size):
    experiences = sample_experiences(batch_size)  # Get experience from replay buffer
    states, actions, rewards, next_states, dones = experiences # Unpack the experiences 
    next_Q_values = model.predict(next_states) # Predict the Q value of the next states
    max_next_Q_values = np.max(next_Q_values, axis=1)  # Take highest Q value
    target_Q_values = (rewards +  # Calculate target Q values
                        (1 - dones) * discount_factor * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1,1)  # Reshape target Q values into vector
    mask = tf.one_hot(actions, n_outputs)  # Encode actions as one hot matrix
    with tf.GradientTape() as tape:  # Record operations
        all_Q_values = model(states)  # Predict Q values for current state
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True) # Get Q values
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))  # Calculate loss between predicted and actual
    grads = tape.gradient(loss, model.trainable_variables) # Calculate the gradient of the loss
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) # Apply computed gradients to update parameters

# Train the agent
all_rewards = []  # Store total rewards
episode_lengths = [] # Store steps taken in each episode
for episode in range(600):  # Loop for 600 episodes
    obs, info = env.reset()  # Reset the environment each episode
    total_reward = 0 # Initialize total rewards for current episode
    steps = 0 # Initialize steps for current episode 
    for step in range(200):  # Take 200 steps per episode
        epsilon = max(1 - episode / 500, 0.01)  # Compute exploration rate
        obs, reward, done, truncated, info = play_one_step(env, obs, epsilon) # Take one step in the environment
        total_reward += reward  # Add any rewards for the step
        steps += 1  # Increase steps by 1
        if done or truncated: # Check if episode is done
            break # If done break exit loop
        if episode > 50:  # Start training model after 50 steps
            training_step(batch_size)  
    all_rewards.append(total_reward)  # Add total rewards to list
    episode_lengths.append(steps)  # Add episode length to list
            
# Analyze the agent's learning progress
plt.figure(figsize=(16,8))

# Cumulative Rewards
plt.subplot(1,2,1)
plt.plot(all_rewards, label="Cumulative Reward", color="green")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards Over Episodex")
plt.grid()
plt.legend()

# Episode Lengths
plt.subplot(1,2,2)
plt.plot(episode_lengths, label="Episode Length", color="red")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Episode Length Over Time")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Render and watch agent
def watch_agent(env, model, episodes = 5):  # Function to view 5 episodes of trained agent
    for episode in range(episodes):  # Loop through 5 episodes
        obs, info = env.reset()  # Reset the environment
        done = False  # Initialize ending sequence
        total_reward = 0  # Initlaize total rewards for the episode
        while not done:  # Conditional to continue loop until episode is over
            env.render() # Render the environment
            time.sleep(0.02) # Slight delay to view agents actions
            obs_tensor = obs[np.newaxis] # Prepare the observation for model prediction
            q_values = model.predict(obs_tensor)  # Predict Q values for current state
            action = np.argmax(q_values[0]) # Choose best action based on Q values
            obs, reward, terminated, truncated, info = env.step(action) # Take action in the environment
            done = terminated or truncated # Check if episode has ended 
            total_reward += reward # Add reward to toal reward
        print(f"Episode {episode + 1}: Total Reward = {total_reward}") # Display each episode and accumulated rewards
    env.close() # Close the environment

watch_agent(env, model, episodes=5)  # Call function to view trained agent