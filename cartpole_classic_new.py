import tensorflow as tf
import tensorflow_quantum as tfq
from circuit_builder import *
from env_inter import *
import gym, cirq, sympy, random
import numpy as np
from functools import reduce
from collections import deque, namedtuple, defaultdict
import matplotlib.pyplot as plt
import os.path
from os import path

def Q_learning_loss(replay_memory, batch_size, loss_function, model, model_target, gamma, n_actions):
    # Sample a batch of interactions
    batch = random.choices(replay_memory, k=batch_size)
    batch = interaction(*zip(*batch))

    # Compute their target q_values and the masks on sampled actions
    future_rewards = model_target.predict([tf.constant(batch.next_state)])
    target_q_values = tf.constant(batch.reward) + (gamma * tf.reduce_max(future_rewards, axis=1)
                                                   * (1 - tf.constant(batch.done)))
    masks = tf.one_hot(batch.action, n_actions)

    # Train the model on the states and target Q-values
    with tf.GradientTape() as tape:
        q_values = model([tf.constant(batch.state)])
        # Apply the masks to the Q-values
        q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        # Calculate loss between target Q-values and model Q-values
        loss = loss_function(target_q_values, q_values_masked)
    return tape, loss
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

def generate_model_Qlearning_C(n_layers, n_actions, target):
    """Generates a Keras model for a data re-uploading PQC Q-function approximator."""
    model = tf.keras.Sequential() 
    n_actions = env.action_space.n
    input_dim = env.observation_space.shape[0]
    model.add(tf.keras.layers.Dense(64, input_dim = input_dim , activation = 'relu'))
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(n_actions, activation = 'linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'mse')
    return model
n_qubits = 4 # Dimension of the state vectors in CartPole
n_layers = 3# Number of layers in the PQC

n_actions = 2 # Number of actions in CartPole
gamma = 0.99
n_episodes = 2000


env = gym.make("CartPole-v1")
env.seed(10)
np.random.seed(10)
random.seed(10)
tf.random.set_seed(10)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
model = generate_model_Qlearning_C( n_layers, n_actions, False)
model_target = generate_model_Qlearning_C( n_layers, n_actions, True)
model_target.set_weights(model.get_weights())
#hyperparameters
gamma = 0.95
n_episodes = 2000

# Define replay memory
max_memory_length = 10000 # Maximum replay length
replay_memory = deque(maxlen=max_memory_length)

epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter
batch_size = 16
update_VQC = 10 # Train the model every update_VQC steps
update_target = 30 # Update the target model every update_target steps

loss_function = tf.keras.losses.MeanSquaredError()

interaction = namedtuple('interaction', ('state', 'action', 'reward', 'next_state', 'done'))
episode_reward_history = []
step_count = 0
for episode in range(n_episodes):
    episode_reward = 0
    state = env.reset()
    # state = [0 if i < 0 else 1 for i in state]
    done = False
    
    while not done:
        # Increase step count
        step_count += 1
        
        # Interact with env
        state, action, next_state, reward, done = interaction_env(state, model, epsilon, n_actions, env)
        
        # Store last interaction in the replay memory
        sarsd = interaction(np.copy(state), action, reward, np.copy(next_state), float(done))
        replay_memory.append(sarsd)
        # next_state = [0 if i < 0 else 1 for i in next_state]
        state = np.array(next_state)
        episode_reward += reward
        
        # Update model
        if  step_count % update_VQC == 0  and len(replay_memory) >= batch_size:
            # Compute loss
            tape, loss = Q_learning_loss(replay_memory, batch_size, loss_function, model, model_target, 
                                         gamma, n_actions)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Update target model
        if episode % update_target == 0:
            model_target.set_weights(model.get_weights())

    # Decay epsilon
    epsilon = max(epsilon * decay_epsilon, epsilon_min)
    episode_reward_history.append(episode_reward)
    if (episode+1)%10 == 0:
        print("Episode {}/{}, average last 10 rewards {}".format(episode+1, n_episodes,
                                                          np.mean(episode_reward_history[-10:])))
plt.figure(figsize=(10,5))
plt.plot(episode_reward_history)
plt.xlabel('Epsiode')
plt.ylabel('Collected rewards')
plt.show()