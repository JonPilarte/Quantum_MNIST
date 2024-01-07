from qiskit import QuantumCircuit, Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit import ParameterVector, Parameter

from qiskit.algorithms.optimizers import SPSA

from qiskit.opflow import Z, I, StateFn, PauliExpectation, AerPauliExpectation
from qiskit.opflow.gradients import Gradient
from qiskit_machine_learning.neural_networks import OpflowQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier

from datetime import datetime
import matplotlib.pyplot as plt
import json

from qiskit_machine_learning.utils.loss_functions import Loss
import numpy as np


class my_L2Loss(Loss):
    r"""
    This class computes the L2 loss (i.e. squared error) for each sample as:

    .. math::

        \text{L2Loss}(predict, target) = \sum_{i=0}^{N_{\text{elements}}} (predict_i - target_i)^2.

    """
    __loss_values__ = []
    __epoch__ = 1

    def evaluate(self, predict: np.ndarray, target: np.ndarray, verbose=True) -> np.ndarray:
        self._validate_shapes(predict, target)

        if len(predict.shape) <= 1:
            aux = -(predict - target) ** 2
            my_L2Loss.__loss_values__.append(aux)
            return aux
        else:
            aux = -np.linalg.norm(predict - target, axis=tuple(range(1, len(predict.shape)))) ** 2
            if verbose:
                print('Loss call: #', my_L2Loss.__epoch__)
                print('Average loss: ', np.mean(aux))
                print('Standard deviation: ', np.std(aux))
                print('Cumulative loss: ', np.sum(aux))
                print('Length : ', len(aux))
                print('----------')
            my_L2Loss.__loss_values__.append(aux)
            my_L2Loss.__epoch__ += 1
            return aux

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._validate_shapes(predict, target)
        return 2 * (predict - target) #Vanilla Gradient Descent (could be replaced by other gradients)


class QuantumPerceptron:
    def __init__(self,
                 n_qubits,
                 input_data,
                 labels,
                 test_input_data,
                 test_labels,
                 shots=1024,
                 epochs=20,
                 lr=None,
                 perturbation=None,
                 provider=None,
                 backend=None,
                 exp_dir='',
                 initial_layout=None,
                 iris=False,
                 warm_start=False,
                 initial_point=None,
                 testing=False):

        self.num_qubits = n_qubits

        if provider:
            print('Using provider...')
            self.q_instance = QuantumInstance(provider.get_backend(backend),
                                              shots=shots,
                                              job_callback=self.job_callback,
                                              initial_layout=initial_layout,
                                              )
            self.using_simulator = False

        else:
            print('Using local simulator...')
            self.q_instance = QuantumInstance(Aer.get_backend('aer_simulator'),
                                              shots=shots,
                                              initial_layout=initial_layout)
            self.using_simulator = True

        self.num_features = len(input_data[0])
        self.provider = provider
        self.backend = backend

        self.input_data = input_data
        self.labels = labels
        self.test_input_data = test_input_data
        self.test_labels = test_labels
        self.epochs = epochs
        self.learning_rate = lr
        self.exp_dir = exp_dir
        self.loss = my_L2Loss()
        self.perturbation = perturbation
        self.iris = iris
        self.warm_start = warm_start
        self.initial_point = initial_point
        self.testing = testing

        self.main_circuit = None
        self.classifier = None
        self.entangle_flag = True
        self.training_history = {'num_func_ev': [], 'parameters': [],
                                 'func_val': [], 'accepted': [], 'score': []}
        self.jobs_history = {'job_id': [], 'job_status': [],
                             'queue_position': [], 'job': []}
        self.aux_cont = 1
        self.build_main_circuit()

        self.cobyla_callback = {'loss':[], 'params':[], 'score':[]}

    def training_callback(self, *args):
        # This callback receives
        # the number of function evaluations, the parameters,
        # the function value, the stepsize, whether the step was accepted.
        print('Call #', self.aux_cont)

        self.training_history['num_func_ev'].append(args[0])
        self.training_history['parameters'].append(list(args[1]))
        self.training_history['func_val'].append(args[2]) # Loss
        self.training_history['accepted'].append(args[3])

        self.write_to_file(sur=self.exp_dir, name=f'{self.aux_cont}')
        self.aux_cont += 1

    def job_callback(self, *args):
        # This callback receives
        # `job_id, job_status, queue_position, job`
        self.jobs_history['job_id'].append(args[0])
        self.jobs_history['job_status'].append(args[1])
        self.jobs_history['queue_position'].append(args[2])
        self.jobs_history['job'].append(args[3])

    # TODO: esta no la necesitas
    def us_between(self, circuit, crossr, qubits, last_idx):
        for q0, q1 in crossr:
            for idx in range(q0, q1 + 1):
                circuit.u(self.data_params[last_idx % self.num_features],
                          self.data_params[(last_idx + 1) % self.num_features],
                          self.data_params[(last_idx + 2) % self.num_features],
                          idx
                          )
                last_idx += 3

        for qubit in qubits:
            circuit.u(self.data_params[last_idx % self.num_features],
                      self.data_params[(last_idx + 1) % self.num_features],
                      self.data_params[(last_idx + 2) % self.num_features],
                      qubit
                      )
            last_idx += 3

        return circuit, last_idx

    def build_main_circuit(self):

        circuit = QuantumCircuit(self.num_qubits)
        self.data_params = ParameterVector('x', self.num_features)
        self.trainable_params = ParameterVector('theta', self.num_features)
  
        angle = np.pi/2 # (Podria ser optimizado mas adelante)
        last_idx = 0

        # Expresar features (784 pixeles) usando basic gates de Kolkata (RZ, SX, X)
        # Layers Entangelment (se eliminan las aristas 12-13 y 23-24, minimo n-1 aristas (26) para que sea conexo)

        # 0-3 bloques triples, 4-15 bloques dobles, 16-27 bloques simples = 792
        

        rzx=['circuit.rzx(angle, 21, 23)',    
        'circuit.rzx(angle, 18, 21)', 
        'circuit.rzx(angle, 17, 18)', 
        'circuit.rzx(angle, 15, 18)', 
        'circuit.rzx(angle, 12, 15)', 
        'circuit.rzx(angle, 10, 12)', 
        'circuit.rzx(angle, 7, 10)', 
        'circuit.rzx(angle, 6, 7)', 
        'circuit.rzx(angle, 4, 7)', 
        'circuit.rzx(angle, 1, 4)', 
        'circuit.rzx(angle, 0, 1)', 
        'circuit.rzx(angle, 1, 2)', 
        'circuit.rzx(angle, 2, 3)', 
        'circuit.rzx(angle, 3, 5)', 
        'circuit.rzx(angle, 5, 8)', 
        'circuit.rzx(angle, 8, 9)', 
        'circuit.rzx(angle, 8, 11)', 
        'circuit.rzx(angle, 11, 14)', 
        'circuit.rzx(angle, 13, 14)', 
        'circuit.rzx(angle, 14, 16)', 
        'circuit.rzx(angle, 16, 19)', 
        'circuit.rzx(angle, 19, 20)', 
        'circuit.rzx(angle, 19, 22)', 
        'circuit.rzx(angle, 22, 25)', 
        'circuit.rzx(angle, 24, 25)', 
        'circuit.rzx(angle, 25, 26)']         

        
        
        #NEW APPROACH
        measure=(23,21,17,18,15,12,10,6,7,4,0,1,2,3,5,9,8,11,13,14,16,20,19,22,24,26) #len=26, we measure the 25th                                                                                       #qubit at the end
        pixel=0
        kind=[3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1]
        for blocs in range(0,26):
            for i in range(kind[blocs]):
                for idx in range(self.num_qubits):
                    if idx in measure[:blocs]:
                        continue    
                    circuit.h(idx)
                for idx in range(self.num_qubits):
                    if idx in measure[:blocs]:
                        continue    
                    circuit.rz(self.data_params[pixel]*self.trainable_params[pixel], idx)    
                    pixel+=1
                 
            eval(rzx[blocs])
            #circuit.measure([measure[blocs], [measure[blocs])
            
        for i in range(5):
            circuit.h(25)
            circuit.rz(self.data_params[pixel]*self.trainable_params[pixel], 25)
            pixel+=1
            
    
        #FIN BLOQUES
        
        
        circuit.draw('mpl')
        plt.show()
        
        #Qubits que se miden (en este caso el 25)
        #                       25
        observable = (I ^ 24) ^ Z ^ I 

        self.main_circuit = circuit
        qnn_expectation = StateFn(observable, is_measurement=True) @ StateFn(circuit)
        p_exp = AerPauliExpectation() if self.using_simulator else PauliExpectation()

        self.qnn = OpflowQNN(qnn_expectation,
                             input_params=self.data_params,
                             weight_params=self.trainable_params,
                             exp_val=p_exp,
                             gradient=Gradient(),
                             quantum_instance=self.q_instance)

        if self.learning_rate:
            opt = SPSA(maxiter=self.epochs,
                       learning_rate=self.learning_rate,
                       perturbation=self.perturbation,
                       callback=self.training_callback)
        else:
            opt = SPSA(maxiter=self.epochs,
                       learning_rate=None,
                       callback=self.training_callback)

        if self.warm_start:
            self.classifier = NeuralNetworkClassifier(self.qnn,
                                                      optimizer=opt,
                                                      loss=self.loss,
                                                      warm_start=True,
                                                      initial_point=self.initial_point)
        else:
            self.classifier = NeuralNetworkClassifier(self.qnn,
                                                      optimizer=opt,
                                                      loss=self.loss)

    def fit(self):
        print('Training...')
        self.classifier.fit(self.input_data, self.labels)

    def write_to_file(self, name='', sur=''):
        now = datetime.now()
        date_time = now.strftime('%Y_%m_%d_%H_%M_%S')
        file_metrics = date_time + '-' + name

        with open(sur + file_metrics + '_.json', 'w') as outfile:
            json.dump(self.training_history, outfile)
