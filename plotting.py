import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import os
import os.path
cur_path = os.getcwd()
cartpole = cur_path+"/cartpole/"
c_cartpole_models = ["CDQN-2021-05-18_58_ADAM_lr0.1_g0.95_eps1.0_epsmin0.01_epsd0.91",
                     "CDQN-2021-05-18_394_ADAM_lr0.1_g0.95_eps1.0_epsmin0.01_epsd0.91",
                     "CDQN-2021-05-18_1282_ADAM_lr0.1_g0.95_eps1.0_epsmin0.01_epsd0.91"
                      ]

q_cartpole_models = [
    # "QDQN-2021-05-18_qbits4_ADAM_lr0.01_bs_32_g0.95_eps1.0_epsmin0.01_epsd0.91",
                     "QDQN-2021-05-18_qbits4_ADAM_lr0.01_bs_32_g0.99_eps1.0_epsmin0.01_epsd0.91",
                    #  "QDQN-2021-05-18_qbits6_ADAM_lr0.01_bs_32_g0.95_eps1.0_epsmin0.01_epsd0.91",
                    "QDQN-2021-05-18_qbits8_ADAM_lr0.01_bs_32_g0.95_eps1.0_epsmin0.01_epsd0.91",
                    "QDQN-2021-05-18_qbits12_ADAM_lr0.01_bs_32_g0.95_eps1.0_epsmin0.01_epsd0.91",
                     ]

cart_c_58r = np.load(cartpole+c_cartpole_models[0]+"/"+"rewards.npy")
cart_c_58ar = np.load(cartpole+c_cartpole_models[0]+"/"+"averages.npy")
cart_c_58l = np.load(cartpole+c_cartpole_models[0]+"/"+"loss.npy")


cart_c_226r = np.load(cartpole+c_cartpole_models[1]+"/"+"rewards.npy")
cart_c_226ar = np.load(cartpole+c_cartpole_models[1]+"/"+"averages.npy")
cart_c_226l = np.load(cartpole+c_cartpole_models[1]+"/"+"loss.npy")


cart_c_1282r = np.load(cartpole+c_cartpole_models[2]+"/"+"rewards.npy")
cart_c_1282ar = np.load(cartpole+c_cartpole_models[2]+"/"+"averages.npy")
cart_c_1282l = np.load(cartpole+c_cartpole_models[2]+"/"+"loss.npy")

cart_q_4r = np.load(cartpole+q_cartpole_models[0]+"/"+"rewards.npy")
cart_q_4ar = np.load(cartpole+q_cartpole_models[0]+"/"+"averages.npy")
cart_q_4l = np.load(cartpole+q_cartpole_models[0]+"/"+"loss.npy")

cart_q_8r = np.load(cartpole+q_cartpole_models[1]+"/"+"rewards.npy")
cart_q_8ar = np.load(cartpole+q_cartpole_models[1]+"/"+"averages.npy")
cart_q_8l = np.load(cartpole+q_cartpole_models[1]+"/"+"loss.npy")


cart_q_12r = np.load(cartpole+q_cartpole_models[2]+"/"+"rewards.npy")
cart_q_12ar = np.load(cartpole+q_cartpole_models[2]+"/"+"averages.npy")
cart_q_12l = np.load(cartpole+q_cartpole_models[2]+"/"+"loss.npy")

special_ar = np.load("/Users/david/Desktop/quantum_research/cartpole_new/averages-3.npy")
special_r = np.load("/Users/david/Desktop/quantum_research/cartpole_new/rewards-3.npy")

ax=plt.subplot(111)
ax.set_xlim(50,200)
# plt.plot(cart_c_1282l, color='red', label='10^3 NN')
# plt.plot(cart_q_12l, color='blue', label='QVC 12 qubits')
ax.plot(cart_c_1282ar, color='red', label='10^3 NN')
ax.plot(cart_q_12ar, color='blue', label='QVC 12 qubits')
# plt.plot(cart_q_8r, color='blue', alpha = .2, label='Reward')
# plt.plot(cart_c_226l, color='blue', label='10^2 NN')
# plt.plot(cart_q_8l, color='blue', label='QVC 8 qubits')
ax.plot(cart_c_226ar, color='red', label='10^2 NN')
ax.plot(cart_q_8ar, color='blue', label='QVC 8 qubits')
# plt.plot(cart_q_4r, color='purple', alpha = .2, label='Reward')
# plt.plot(cart_c_58l, color='purple', label='10^1 NN')
# plt.plot(cart_q_4l, color='purple', label='QVC 4 qubits')
ax.plot(cart_c_58ar, color='red', label='10^1 NN')
ax.plot(cart_q_4ar, color='blue', label='QVC 4 qubits')
ax.plot(special_ar, color = "green", label = "ReUpload")
plt.legend()
plt.title("Avg. Reward for DQN per Episode (CartPole)")
plt.ylabel('Reward')
plt.xlabel('Iteration')

plt.show()