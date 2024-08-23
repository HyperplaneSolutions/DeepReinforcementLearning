
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
import tensorflow as tf
import time
from collections import deque, namedtuple

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps

env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")
wrapped_env = FlattenObservation(env)
state_size =wrapped_env.observation_space.shape
num_actions = wrapped_env.action_space.shape
print('State Shape:', state_size)
print('Number of actions:', num_actions)

# Create the Q-Network
q_network = Sequential([
    Input(state_size),
    Dense(64, activation = 'relu'),
    Dense(128, activation = 'relu'),
    Dense(num_actions, activation = 'linear')
    ])

target_q_network = Sequential([
    Input(state_size),
    Dense(64, activation = 'relu'),
    Dense(128, activation = 'relu'),
    Dense(num_actions, activation = 'linear')
    ])

optimizer = Adam(ALPHA)

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

def compute_loss(experiences, gamma, q_network, target_q_network):
    """ 
    Calculates the loss.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Keras model for predicting the targets
          
    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences
    
    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))
    
    # Get the q_values and reshape to match y_targets
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
        
    # Compute the loss
    loss = MSE(y_targets, q_values) 
    
    return loss

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()