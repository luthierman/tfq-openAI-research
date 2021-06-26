import tensorflow as tf
import tensorflow_quantum as tfq

import gym, cirq, sympy, random
import numpy as np
from functools import reduce
from collections import deque, namedtuple, defaultdict


def interaction_env(state, model, epsilon, n_actions, env):
    # Preprocess state
    state_array = np.array(state) 
    state = tf.convert_to_tensor([state_array])

    # Sample action
    coin = np.random.random()
    if coin > epsilon:
        q_vals = model([state])
        action = int(tf.argmax(q_vals[0]).numpy())
    else:
        action = np.random.choice(n_actions)

    # Apply sampled action in the environment, receive reward and next state
    next_state, reward, done, _ = env.step(action)
    
    return state_array, action, next_state, reward, done
    

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