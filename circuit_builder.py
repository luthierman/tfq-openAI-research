import tensorflow as tf
import tensorflow_quantum as tfq

import cirq, sympy, random
import numpy as np
from functools import reduce
from collections import deque, namedtuple, defaultdict

def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]

def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates (arranged in a circular topology).
    """
    cz_ops = [cirq.CNOT(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CNOT(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

class Rescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Rescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,input_dim)), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply((inputs+1)/2, tf.repeat(self.w,repeats=tf.shape(inputs)[0],axis=0))

def generate_circuit(qubits, n_layers):
    """
    Takes as input qubits and a number of layers n_layers.
    Returns a data re-uploading circuit and the sympy symbols
    of variational and encoding angles.
    """
    # Number of qubits
    n_qubits = len(qubits)
    
    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
    
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))
    
    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_qubits})'+f'(0:{n_layers})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))
    
    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        # Variational layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)
        # Encoding layer
        circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

    # Last varitional layer
    circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i,q in enumerate(qubits))

    return circuit, list(params.flat), list(inputs.flat)
class ReUploadingPQC(tf.keras.layers.Layer):

    def __init__(self, qubits, n_layers, observables, activation='linear', name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers)
        
        theta_init = tf.random_uniform_initializer(minval=0., maxval=np.pi)
        self.theta = tf.Variable(initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
                                 trainable=True, name="thetas")
        
        lmbd_init = tf.ones(shape=(self.n_qubits*self.n_layers,))
        self.lmbd = tf.Variable(initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas")
        
        symbols = [str(theta) for theta in theta_symbols]+[str(s) for s in input_symbols]
        self.indices = tf.constant([sorted(symbols).index(a) for a in symbols]) # re-ordering of indices
                                                                # to match order of computation_layer.symbols
        self.activation = activation # activation to be applied on the encoding angles
        
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)        

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.keras.layers.Activation(self.activation)(tf.einsum('i,ji->ji', self.lmbd, 
                                                                              tiled_up_inputs))
        joined_vars = tf.concat([tiled_up_thetas, scaled_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1) # re-ordering of indices to match order
                                                                   # of computation_layer.symbols
        return self.computation_layer([tiled_up_circuits, joined_vars])
# expand qubits version
def generate_model_Qlearning(qubits, n_layers, n_actions, observables, target):
    """Generates a Keras model for a data re-uploading PQC Q-function approximator."""

    input_tensor = tf.keras.Input(shape=(len(qubits), ), dtype=tf.dtypes.float32, name='input')
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables, activation='tanh')([input_tensor])
    process = tf.keras.Sequential([Rescaling(len(observables))], name=target*"Target"+"Q-values")
    Q_values = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)

    return model

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

def generate_model_Qlearning_C(env, n_layers, n_actions, target):
    """Generates a Keras model for a data re-uploading PQC Q-function approximator."""
    model = tf.keras.Sequential() 
    n_actions = env.action_space.n
    input_dim = env.observation_space.shape[0]
    model.add(tf.keras.layers.Dense(64, input_dim = input_dim , activation = 'relu'))
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(n_actions, activation = 'linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'mse')
    return model
