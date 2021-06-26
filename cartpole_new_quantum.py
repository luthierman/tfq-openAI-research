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
from make_path import * 
@tf.function
def Q_learning_update(states, rewards, next_states, actions, done, model, gamma, n_actions):
    states = tf.convert_to_tensor(states)
    rewards = tf.convert_to_tensor(rewards)
    next_states = tf.convert_to_tensor(next_states)
    actions = tf.convert_to_tensor(actions)
    done = tf.convert_to_tensor(done)

    # Compute their target q_values and the masks on sampled actions
    future_rewards = model_target([next_states])
    target_q_values = rewards + (gamma * tf.reduce_max(future_rewards, axis=1)
                                                   * (1.0 - done))
    masks = tf.one_hot(actions, n_actions)
    msbe = None
    # Train the model on the states and target Q-values
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        q_values = model([states])
        q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)
        msbe = loss
    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])
    return msbe
n_qubits = 4 # Dimension of the state vectors in CartPole
n_layers = 5# Number of layers in the PQC

n_actions = 2 # Number of actions in CartPole
env = gym.make("CartPole-v1")
env.seed(10)
np.random.seed(10)
random.seed(10)
tf.random.set_seed(10)
qubits = cirq.GridQubit.rect(1, n_qubits)
ops = [cirq.Z(q) for q in qubits]

observables = [ops[0]*ops[1], ops[2]*ops[3]] # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1
model = generate_model_Qlearning(qubits, n_layers, n_actions, observables, False)
model_target = generate_model_Qlearning(qubits, n_layers, n_actions, observables, True)

model_target.set_weights(model.get_weights())
#hyperparameters
gamma = 0.95
n_episodes = 200

# Define replay memory
max_memory_length = 10000 # Maximum replay length
replay_memory = deque(maxlen=max_memory_length)

epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter
batch_size = 32
update_VQC = 1 # Train the model every update_VQC steps
update_target = 3 # Update the target model every update_target steps

optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

# Assign the model parameters to each optimizer
w_in, w_var, w_out = 1, 0, 2

cur_path = os.getcwd()
 
cartpole_path = make_path(cur_path,"/cartpole_new")
print(cartpole_path)
master_path = make_path(cartpole_path+"/","quantum_new_final")
interaction = namedtuple('interaction', ('state', 'action', 'reward', 'next_state', 'done'))

windows = 50
learn_delay = 1000
import math
cur_loss = math.inf
best_avg_reward = -math.inf
episode_reward_history = []
step_count = 0
losses = []
cur_loss = 1
avg_reward = []

best_avg_reward = -math.inf
for episode in range(n_episodes):
    episode_reward = 0
    state = env.reset()
    done = False
    episode_losses=[]
    while not done:
        # Increase step count
        step_count += 1

        # Interact with env
        state, action, next_state, reward, done = interaction_env(state, model, epsilon, n_actions, env)
        
        # Store last interaction in the replay memory
        sarsd = interaction(np.copy(state), action, reward, np.copy(next_state), float(done))
        replay_memory.append(sarsd)
        
        state = np.array(next_state)
        episode_reward += reward
        
        # Update model
        if step_count % update_VQC == 0 and len(replay_memory) >= batch_size:
            # Sample a batch of interactions
            batch = random.choices(replay_memory, k=batch_size)
            batch = interaction(*zip(*batch))
            
            # Update Q-function
            loss = Q_learning_update(np.asarray(batch.state),
                              np.asarray(batch.reward, dtype=np.float32),
                              np.asarray(batch.next_state),
                              np.asarray(batch.action),
                              np.asarray(batch.done, dtype=np.float32),
                              model, gamma, n_actions)
            episode_losses.append(loss)
        # Update target model
        if episode % update_target == 0:
            model_target.set_weights(model.get_weights())

    # Decay epsilon
    episode_reward_history.append(episode_reward)
    avg =0
    if len(episode_reward_history)>windows:
        avg = np.mean(episode_reward_history[-windows:])
    avg_reward.append(avg)
    if avg > best_avg_reward:
        best_avg_reward = avg
    if len(episode_losses)>0:
        EPISODE_LOSSES= np.asarray(episode_losses)
        AVERAGE_EPISODE_LOSS = np.mean(EPISODE_LOSSES)
        losses.append(AVERAGE_EPISODE_LOSS)
        cur_loss = AVERAGE_EPISODE_LOSS
    else:
        losses.append(cur_loss)
    epsilon = max(epsilon * decay_epsilon, epsilon_min)
    print("\rEpisode {}/{} || Best average reward {}, Current Avg {}, Current Iteration Reward {}, eps {}".format(episode+1, n_episodes, best_avg_reward, avg, episode_reward, epsilon), end='', flush=True)
reward_file = "{h}/rewards".format(h = master_path)
average_file = "{h}/averages".format(h=master_path)
loss_file = "{h}/loss".format(h=master_path)     
np.save(reward_file , np.asarray(episode_reward_history))
np.save(average_file , np.asarray(avg_reward))
np.save(loss_file , np.asarray(losses))
plt.figure(figsize=(10,5))
plt.plot(episode_reward_history, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.xlabel('Epsiode')
plt.ylabel('Collected rewards')
plt.show()