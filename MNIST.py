#!/usr/bin/env python
# coding: utf-8


#Change quantum_perceptron depending on the Circuit you want to use
from mnist import get_binary_mnist
from quantum_perceptron.quantum_perceptron import QuantumPerceptron
from qiskit import IBMQ
import torch
import numpy as np
from PIL import Image
import time


# Reproducibility
torch.manual_seed(1)
np.random.seed(4)

num_qubits = 27
epochs = 1

class_0 = 0
class_1 = 1
provider = False
reserv = False

shots = 100000
num_training = 50 # per class
num_testing = 50 # per class
lr = 0.001
perturbation = 0.001

trial_start = 1
trial_end = 1

incomplete_trial_flag = False
#parameters_path = 'mnist/Zn/simulator/trial1/runs/2023_01_27_14_15_29-17_.json'
parameters_path = None
epoch_correction = 0

backend = 'ibmq_qasm_simulator' #'ibmq_qasm_simulator' # 'ibmq_kolkata'
backend_options={
    "method":"statevector",
    "device":"CPU",
}

print('Preparing data...')
training_set, training_set_labels, \
testing_set, testing_set_labels = get_binary_mnist(num_training=num_training,
                                                   num_testing=num_testing,
                                                   class_0=class_0,
                                                   class_1=class_1)

tstart = time.time()
out_path="resultados/"
qnn = QuantumPerceptron(n_qubits=num_qubits,
                                input_data=training_set,
                                labels=training_set_labels,
                                test_input_data=testing_set,
                                test_labels=testing_set_labels,
                                backend=backend,
                                backend_options=backend_options,
                                epochs=epochs,
                                provider=provider,
                                shots=shots,
                                lr=lr,
                                perturbation=perturbation,
                                exp_dir=out_path,
                                #warm_start=warm_start,
                                #initial_poinqt=weights
                                )

print("time qnn:", time.time() - tstart)
#circ = qnn.main_circuit
#circ.draw(output='mpl', filename='circuito.png', fold=-1, scale=1.5)
qnn.fit()

tstart = time.time()
train_acc = qnn.classifier.score(training_set, training_set_labels)
test_acc = qnn.classifier.score(testing_set, testing_set_labels)
print(train_acc)
print(test_acc)
print("time elapsed:", time.time() - tstart)

