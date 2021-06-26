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
import os
import datetime # https://stackoverflow.com/questions/10607688/how-to-create-a-file-name-with-the-current-date-time-in-python
from circuit_builder import *
from cirq.contrib.svg import SVGCircuit
from make_path import *
class QDQN_alt(object):
    def __init__(self, action_space, state_space,  no_qubits=4, no_layers=1) -> None:
        super().__init__()
        self.action_space = action_space
        self.n_layers= no_layers
        self.state_space = state_space
        self.no_qubits = no_qubits
        self.qubits = [cirq.GridQubit(0, i) for i in range(no_qubits)]
        self.q_network = self.make_pure_model()
        
        self.learning_rate = 0.01
        self.opt = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.buff = 10000
        self.batch = 32   
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
    def make_pure_model(self):
        readout_operators = [cirq.Z(self.qubits[i]) for i in range(2,4)]
        inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        diff = tfq.differentiators.ParameterShift()
        init = tf.keras.initializers.Zeros
        pqc = tfq.layers.PQC(self.make_circuit(self.qubits, self.n_layers), readout_operators, differentiator=diff, initializer=init)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=pqc)
        return model

    def convert_data(self, classical_data, flag=True):
        ops = cirq.Circuit()
        max_scale = max(classical_data)
        min_scale = min(classical_data)
        for i, ang in enumerate(classical_data):
            ang = 2*np.pi*(ang-min_scale)/(max_scale-min_scale)
            ops.append(cirq.rx(np.pi * ang).on(self.qubits[i]))
            ops.append(cirq.rz(np.pi * ang).on(self.qubits[i]))
        if flag:
            return tfq.convert_to_tensor([ops])
        else:
            return ops

    def one_qubit_rotation(self, qubit, symbols):
        """
        Returns Cirq gates that apply a rotation of the bloch sphere about the X,
        Y and Z axis, specified by the values in `symbols`.
        """
        # print(symbols, "hi")
        return [cirq.rx(symbols[0])(qubit),
                cirq.ry(symbols[1])(qubit),
                cirq.rz(symbols[2])(qubit)]
    def two_qubit_rotation(self, bits, symbols):
        """Make a Cirq circuit that creates an arbitrary two qubit unitary."""
        circuit = cirq.Circuit()
        circuit += cirq.Circuit(self.one_qubit_rotation(bits[0], symbols[0:3]))
        circuit += cirq.Circuit(self.one_qubit_rotation(bits[1], symbols[3:6]))
        circuit += [cirq.ZZ(*bits)**symbols[6]]
        circuit += [cirq.YY(*bits)**symbols[7]]
        circuit += [cirq.XX(*bits)**symbols[8]]
        circuit += cirq.Circuit(self.one_qubit_rotation(bits[0], symbols[9:12]))
        circuit += cirq.Circuit(self.one_qubit_rotation(bits[1], symbols[12:]))
        return circuit
    def entangling_layer(self, qubits):
        """
        Returns a layer of CZ entangling gates (arranged in a circular topology).
        """
        cz_ops = [cirq.CNOT(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
        cz_ops += ([cirq.CNOT(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
        return cz_ops
    def quantum_pool_circuit(self, source_bits, sink_bits, symbols):
        """A layer that specifies a quantum pooling operation.
        A Quantum pool tries to learn to pool the relevant information from two
        qubits onto 1.
        """
        circuit = cirq.Circuit()
        for source, sink in zip(source_bits, sink_bits):
            circuit += self.two_qubit_pool(source, sink, symbols)
        return circuit
    def two_qubit_pool(self, source_qubit, sink_qubit, symbols):
        pool_circuit = cirq.Circuit()
        sink_basis_selector = cirq.Circuit(self.one_qubit_rotation(sink_qubit, symbols[0:3]))
        source_basis_selector = cirq.Circuit(self.one_qubit_rotation(source_qubit, symbols[3:6]))
        pool_circuit.append(sink_basis_selector)
        pool_circuit.append(source_basis_selector)
        pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
        pool_circuit.append(sink_basis_selector**-1)
        return pool_circuit

    def make_circuit(self, qubits, n_layers):
        m = cirq.Circuit()
        n_qubits = len(qubits)
        # 4 qubits * 3 weights per bit * 3 layers + 2 * 6 pooling = 36 + 12 = 48
        params_w_pool = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits + 2*3*n_qubits//2})')
        params = params_w_pool[:-2*3*n_qubits//2] 

        params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))
        inputs = sympy.symbols(f'x(0:{n_qubits})'+f'(0:{n_layers})')
        inputs = np.asarray(inputs).reshape((n_layers, n_qubits))
        
        # print((self.two_qubit_rotation(cirq.GridQubit.rect(1, 2), sympy.symbols('x0:15'))))
        for l in range(n_layers):
        # Variational layer
            
            m += cirq.Circuit(self.one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
            m += self.entangling_layer(qubits)
            # Encoding layer
            m += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))
        
        m += cirq.Circuit(self.one_qubit_rotation(q, params[n_layers, i]) for i,q in enumerate(qubits))
        # pooling
        
        pool_params = params_w_pool[-2*3*n_qubits//2:] 
        sources= qubits[:n_qubits//2]
        sinks = qubits[n_qubits//2:]
        m += self.quantum_pool_circuit(sources, sinks, pool_params)

        print(m)
        return m
    
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
    # @tf.function
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

ITERATIONS = 750
batch_size = 32
windows = 50
learn_delay = 500
qubits = [12]

for q in qubits:
  losses = []
  cur_loss = 1
  env = gym.make("Blackjack-v0")
  env.seed(10)
  np.random.seed(10) 
  random.seed(10)
  tf.random.set_seed(10)
  agent = QDQN_alt(env.action_space.n, 3, q, 3)
  cur_path = os.getcwd()
  blackjack_path = make_path(cur_path,"/blackjack")
  master_path = make_path(blackjack_path+"/", agent.model_name+str(1))
  rewards = []
  losses = []
  cur_loss = math.inf
  avg_reward = deque(maxlen=ITERATIONS)
  best_avg_reward = -math.inf
  for i in range(ITERATIONS):
      s1 = env.reset()
      total_reward = 0
      episode_losses=[]
      done = False
      while not done:
          action = agent.get_action(s1)
          s2, reward, done, info = env.step(action)
          total_reward += reward
          agent.remember(s1, action, reward, s2, done)
          if agent.counter > learn_delay and done:
              agent.train()
              episode_losses.append(agent.msbe)
          s1 = s2
      rewards.append(total_reward)
      avg = -math.inf
      if len(rewards)>windows:
          avg = np.mean(rewards[-windows:])
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
      print("\rEpisode {}/{} || Best average reward {}, Current Avg {}, Current Iteration Reward {}, eps {}".format(i, ITERATIONS, best_avg_reward, avg, total_reward, agent.epsilon), end='', flush=True)

  reward_file = "{h}/rewards".format(h = master_path)
  average_file = "{h}/averages".format(h=master_path)
  loss_file = "{h}/loss".format(h=master_path)
  np.save(reward_file , np.asarray(rewards))
  np.save(average_file , np.asarray(avg_reward))
  np.save(loss_file , np.asarray(losses))
plt.ylim(-1,1)
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='blue', label='Average')
plt.plot(losses, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Generation')
plt.show()