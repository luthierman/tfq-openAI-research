import tensorflow as tf
import tensorflow_quantum as tfq
import gym, cirq, sympy, random

import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from make_path import *
from collections import namedtuple, deque
Transition = namedtuple('Transition',
						('state', 'action', 'reward', 'next_state', 'done'))

def prep_angles(classical_data, qbits):
    i = 0
    ret = []
    for i,ang in enumerate(classical_data):
        rx_g = cirq.rx(np.pi*ang)
        ret.append(rx_g(qbits[i]))
        rz_g = cirq.rz(np.pi*ang)
        ret.append(rz_g(qbits[i]))
        i += 1
    a = cirq.Circuit()
    a.append(ret)
    return a

env = gym.make("FrozenLake-v0")
def make_model():
  n_actions = env.action_space.n
  input_dim = env.observation_space.n
  model = tf.keras.Sequential() 
  model.add(tf.keras.layers.Dense(64, input_dim = input_dim , activation = 'relu'))
  model.add(tf.keras.layers.Dense(32, activation = 'relu'))
  model.add(tf.keras.layers.Dense(n_actions, activation = 'linear'))
  model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'mse')
  return model
random.seed(100)
def epsilon_greedy(epsilon, actions, model, state):
    if random.random() < epsilon:
        return random.randint(0, actions-1)
    else:
        return np.argmax(model.predict(state)[0])

def classic_prep(state):
  return np.array(tf.one_hot(state, depth=16)).reshape(1,-1)
def train(model,model_target, optimizer,memory, BATCH_SIZE, loss_function, GAMMA):
  minibatch = random.choices(memory,k=BATCH_SIZE)
  minibatch = Transition(*zip(*minibatch))
  next_states = [tf.one_hot(minibatch.next_state, depth=16)]
  future_rewards = model_target.predict(next_states)
  target_q_values = tf.constant(minibatch.reward) + (GAMMA * tf.reduce_max(future_rewards, axis=1)
                                                    * (1 - tf.cast(tf.constant(minibatch.done),tf.float32)))
  masks = tf.one_hot(minibatch.action, 4)
  with tf.GradientTape() as tape:
    q_values = model([tf.one_hot(minibatch.state, depth=16)])
            # Apply the masks to the Q-values
    q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between target Q-values and model Q-values
    loss = loss_function(target_q_values, q_values_masked)
  return tape, loss

ITERATIONS = 500 # @param {type:"integer"}
MEMORY_SIZE = 10000 # @param {type:"integer"}
GAMMA = 0.99 # @param {type: "number"}
EPSILON = 1.0 # @param {type: "number"}
EPSILON_DECAY = 0.99 # @param {type: "number"}
EPSILON_MIN = 0.01 # @param {type: "number"}
UPDATE_MODEL =1 # @param {type: "integer"}
UPDATE_TARGET =1 # @param {type: "integer"}
ALPHA = .01  # @param {type: "number"}
BATCH_SIZE =   32# @param {type:"integer"}
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate = ALPHA, epsilon=1e-8) # @param {type:""}
LOSS_FUNCTION = tf.keras.losses.MeanSquaredError() # @param {type:""}
env.seed(10)
np.random.seed(10)
random.seed(10)
tf.random.set_seed(10)
memory = deque(maxlen = MEMORY_SIZE)
import os
windows = 50
avg_r_hist = deque(maxlen=ITERATIONS)
best_avg_reward = -math.inf
rs = deque(maxlen=windows)
cur_path = os.getcwd()
frozen_path = make_path(cur_path,"/frozen_lake")
master_path = make_path(frozen_path+"/", "classical 10^1")
model = make_model()
model_target = make_model()
model_target.set_weights(model.get_weights())
state = env.reset()
max_steps = 500
rewards = []
avg_reward = deque(maxlen=ITERATIONS)
best_avg_reward = -math.inf
rs = deque(maxlen=windows)
losses = []
import math
cur_loss = math.inf
for i in range(ITERATIONS):
    s1 = env.reset()
    total_reward = 0
    done = False
    episode_losses=[]
    EPSILON = EPSILON/(i/100 + 1.)
    j = 0
    while not done:
        state1 = classic_prep(s1)
        action = epsilon_greedy(EPSILON, 4, model, state1)
        s2, reward, done, info = env.step(action)
        if j >= max_steps:
            done = True
        if reward < 0.9:
            if done:
                reward = -0.2
            else:
                reward = -0.01
        total_reward += reward
        sarnd = Transition(s1, action, reward, s2, done)
        memory.append(sarnd)
        if done and len(memory) >= BATCH_SIZE :
            tape, loss = train(model,model_target, OPTIMIZER,memory,BATCH_SIZE,LOSS_FUNCTION, GAMMA)
            episode_losses.append(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            OPTIMIZER.apply_gradients(zip(grads, model.trainable_variables))
        if i % UPDATE_TARGET== 0:
          model_target.set_weights(model.get_weights())
        if done:
            rewards.append(total_reward)
            rs.append(total_reward)
        else:
            s1 = s2
        j += 1
    if i >= windows:
        avg = np.mean(rs)
        avg_reward.append(avg)
        if avg > best_avg_reward:
            best_avg_reward = avg
    else: 
        avg_reward.append(-0.5)
    if len(episode_losses)>0:
        EPISODE_LOSSES= np.asarray(episode_losses)
        AVERAGE_EPISODE_LOSS = np.mean(EPISODE_LOSSES)
        losses.append(AVERAGE_EPISODE_LOSS)
        cur_loss = AVERAGE_EPISODE_LOSS
    else:
        losses.append(cur_loss)
    print("\rEpisode {}/{} || Best average reward {}, Current Iteration Reward {}".format(i, ITERATIONS, best_avg_reward, total_reward) , end='', flush=True)
reward_file = "{h}/rewards".format(h = master_path)
average_file = "{h}/averages".format(h=master_path)
loss_file = "{h}/loss".format(h=master_path)     
np.save(reward_file , np.asarray(rewards))
np.save(average_file , np.asarray(avg_reward))
np.save(loss_file , np.asarray(losses))
plt.ylim(-0.5,1.5)
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Generation')
plt.show()