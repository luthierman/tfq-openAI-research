import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq 
import sympy
import time
import datetime # https://stackoverflow.com/questions/10607688/how-to-create-a-file-name-with-the-current-date-time-in-python
class QDQN_alt(object):
    def __init__(self, action_space, state_space, batch, no_qubits=4) -> None:
        super().__init__()
        self.action_space = action_space
        self.state_space = state_space
        self.no_qubits = no_qubits
        self.qubits = [cirq.GridQubit(0, i) for i in range(no_qubits)]
        self.q_network = self.make_func_approx()
        self.learning_rate = 0.01
        self.opt = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.buff = 10000
        self.batch = batch      
        self.states = np.zeros((self.buff, self.state_space))
        self.actions = np.zeros((self.buff, 1))
        self.rewards = np.zeros((self.buff, 1))
        self.dones = np.zeros((self.buff, 1))
        self.next_states = np.zeros((self.buff, self.state_space))
        # Q Learning
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.01
        self.counter = 0
        self.date = datetime.date.today()
        self.model_name = "QDQN-{date}_qbits{q}_ADAM_lr{lr}_bs_{bs}_g{g}_eps{ep}_epsmin{epmin}_epsd{epd}".format(
            date=self.date,
            q=self.no_qubits,g=self.gamma, 
            lr=self.learning_rate,
            bs=self.batch,
            ep=self.epsilon,
            epmin=self.epsilon_min,
            epd=self.epsilon_decay)
        self.msbe = None
    def make_func_approx(self):
        readout_operators = [cirq.Z(self.qubits[i]) for i in range(2,4)]
        inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        diff = tfq.differentiators.ParameterShift()
        init = tf.keras.initializers.Zeros
        pqc = tfq.layers.PQC(self.make_circuit(self.qubits), readout_operators, differentiator=diff, initializer=init)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=pqc)
        return model

    def convert_data(self, classical_data, flag=True):
        ops = cirq.Circuit()
        for i, ang in enumerate(classical_data):
            ang = 0 if ang < 0 else 1
            ops.append(cirq.rx(np.pi * ang).on(self.qubits[i]))
            ops.append(cirq.rz(np.pi * ang).on(self.qubits[i]))
        if flag:
            return tfq.convert_to_tensor([ops])
        else:
            return ops

    def one_qubit_unitary(self, bit, symbols):
        return cirq.Circuit(
            cirq.X(bit)**symbols[0],
            cirq.Y(bit)**symbols[1],
            cirq.Z(bit)**symbols[2])

    def two_qubit_pool(self, source_qubit, sink_qubit, symbols):
        pool_circuit = cirq.Circuit()
        sink_basis_selector = self.one_qubit_unitary(sink_qubit, symbols[0:3])
        source_basis_selector = self.one_qubit_unitary(source_qubit, symbols[3:6])
        pool_circuit.append(sink_basis_selector)
        pool_circuit.append(source_basis_selector)
        pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
        pool_circuit.append(sink_basis_selector**-1)
        return pool_circuit

    def make_circuit(self, qubits):
        m = cirq.Circuit()
        no_vars = self.no_qubits*3*3 + 2*6  
        no_vars_str = "q0:"+str(no_vars)
        print(no_vars_str)
        symbols = sympy.symbols(no_vars_str) # 4 qubits * 3 weights per bit * 3 layers + 2 * 6 pooling = 36 + 12 = 48
        m += self.layer(symbols[:3*self.no_qubits], qubits)
        m += self.layer(symbols[3*self.no_qubits:2*3*self.no_qubits], qubits)
        m += self.layer(symbols[2*3*self.no_qubits:3*3*self.no_qubits], qubits)
        m += self.two_qubit_pool(self.qubits[0], self.qubits[2], symbols[3*3*self.no_qubits:3*3*self.no_qubits+6])
        m += self.two_qubit_pool(self.qubits[1], self.qubits[3], symbols[3*3*self.no_qubits+6:])
        print(m)
        return m
    
    def layer(self, weights, qubits):
        l = cirq.Circuit()
        for i in range(len(qubits) - 1):
            l.append(cirq.CNOT(qubits[i], qubits[i+1]))
        l.append([cirq.Moment([cirq.rx(weights[j]).on(qubits[j]) for j in range(self.no_qubits)])])
        l.append([cirq.Moment([cirq.ry(weights[j + self.no_qubits]).on(qubits[j]) for j in range(self.no_qubits)])])
        l.append([cirq.Moment([cirq.rz(weights[j + 2*self.no_qubits]).on(qubits[j]) for j in range(self.no_qubits)])])
        return l
    
    def remember(self, state, action, reward, next_state, done):
        i = self.counter % self.buff
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = int(done)
        self.counter += 1

    def get_action(self, obs):
        if random.random() < self.epsilon: 
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_network.predict(self.convert_data(obs)))
    def train(self):
        batch_indices = np.random.choice(min(self.counter, self.buff), self.batch)
        state_batch = tfq.convert_to_tensor([self.convert_data(i, False) for i in self.states[batch_indices]])
        action_batch = tf.convert_to_tensor(self.actions[batch_indices], dtype=tf.int32)
        action_batch = [[i, action_batch[i][0]] for i in range(len(action_batch))]
        reward_batch = tf.convert_to_tensor(self.rewards[batch_indices], dtype=tf.float32)
        dones_batch = tf.convert_to_tensor(self.dones[batch_indices], dtype=tf.float32)
        next_state_batch = tfq.convert_to_tensor([self.convert_data(i, False) for i in self.next_states[batch_indices]])

        with tf.GradientTape() as tape:
            next_q = self.q_network(next_state_batch)
            next_q = tf.expand_dims(tf.reduce_max(next_q, axis=1), -1)
            y = reward_batch + (1 - dones_batch) * self.gamma * next_q
            q_guess = self.q_network(state_batch, training=True)
            pred = tf.gather_nd(q_guess, action_batch)
            pred = tf.reshape(pred, [self.batch, 1])
            msbe = tf.math.reduce_mean(tf.math.square(y - pred))
            self.msbe = msbe
        grads = tape.gradient(msbe, self.q_network.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
import os.path
from os import path
def make_path(p, d):
  print("Checking if {} exists...".format(p+d))
  if path.exists(p+d) == False:
    print("making... new directory")
    os.mkdir(p+str(d))
  print("finished!")
  print(p+str(d))
  return p+str(d)

ITERATIONS = 200
batch_size = 32
windows = 50
learn_delay = 1000
qubits = [4]

batch_sizes = 32

for q in qubits:
  losses = []
  cur_loss = 1
  env = gym.make("CartPole-v1")
  env.seed(10)
  np.random.seed(10) 
  random.seed(10)
  tf.random.set_seed(10)
  agent = QDQN_alt(env.action_space.n, env.observation_space.shape[0], batch_sizes, q)
  cur_path = os.getcwd()
  print(cur_path)
  cartpole_path = make_path(cur_path,"/cartpole")
  master_path = make_path(cartpole_path+"/", agent.model_name)
  rewards = []
  losses = []
  cur_loss = 1
  avg_reward = deque(maxlen=ITERATIONS)
  best_avg_reward = -math.inf
  rs = deque(maxlen=windows)
  epi_times = []
  start_time = time.process_time()

  for i in range(ITERATIONS):
      s1 = env.reset()
      total_reward = 0
      episode_losses=[]
      done = False
      episode_start = time.process_time()
      while not done:
          action = agent.get_action(s1)
          s2, reward, done, info = env.step(action)
          total_reward += reward
          agent.remember(s1, action, reward, s2, done)
          if agent.counter >= learn_delay and done:
              agent.train()
              episode_losses.append(agent.msbe)
          if done:
              rewards.append(total_reward)
              rs.append(total_reward)
          s1 = s2
      avg = np.mean(rs)
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
      epi_end = time.process_time() -episode_start
      epi_times.append(epi_end)
      print("\rEpisode {}/{} || Best average reward {}, Current Iteration Reward {}".format(i, ITERATIONS, best_avg_reward, total_reward))
  reward_file = "{h}/rewards".format(h = master_path)
  average_file = "{h}/averages".format(h=master_path)
  times_file = "{h}/times".format(h=master_path)
  loss_file = "{h}/loss".format(h=master_path)
  np.save(reward_file , np.asarray(rewards))
  np.save(average_file , np.asarray(avg_reward))
  np.save(times_file , np.asarray(epi_times))
  np.save(loss_file , np.asarray(losses))
plt.ylim(0,200)
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Generation')
plt.show()