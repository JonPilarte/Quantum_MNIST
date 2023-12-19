from qiskit import IBMQ
import os
import torch
import numpy as np
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from quantum_perceptron.quantum_perceptron_circuit3 import QuantumPerceptron # Choose the circuit
import sys
import numpy
import json


class BinaryMNIST(Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        class_0=0,
        class_1=1,
    ):
        print('Geting ready')
        self.original_mnist = MNIST(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        print('Data got')
        #self.original_mnist = self.original_mnist.data.numpy()
        self.class_0 = class_0
        self.class_1 = class_1
        self.class_maps = {
            self.class_0: 0,
            self.class_1: 1,
        }
        self.index_zeros_ones = [
            i
            for i in range(len(self.original_mnist))
            if self.original_mnist[i][1] == self.class_0
            or self.original_mnist[i][1] == self.class_1
        ]

    def __getitem__(self, index):
        real_mnist_index = self.index_zeros_ones[index]
        x, y = self.original_mnist[real_mnist_index]
        return x, self.class_maps[y]

    def __len__(self):
        return len(self.index_zeros_ones)


def get_binary_mnist(num_training, num_testing, class_0=0, class_1=1):
    data_path = os.environ.get('MNIST', '/tmp/data')
    dataset_class = BinaryMNIST
    mnist_dataset = dataset_class(
        data_path,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
        class_0=class_0,
        class_1=class_1
    )
    # Split the MNIST dataset in 80/20 fashion
    lengths = [int(len(mnist_dataset) * 0.8), int(len(mnist_dataset) * 0.2)]
    
    # Shameless fix for rounding error
    lengths[1] += 1 if sum(lengths) < len(mnist_dataset) else 0

    train_dataset, test_dataset = random_split(mnist_dataset, lengths)

    training_set = []
    training_set_labels = []

    testing_set = []
    testing_set_labels = []

    cont_class_0 = 0
    cont_class_1 = 0

    print(f"Length of train_dataset: {len(train_dataset)}")

    if num_testing == 0 and num_training == 0:
        for i in train_dataset:
            if i[1] == 0:
                training_set.append(i[0].flatten().numpy())
                training_set_labels.append(-1)
                cont_class_0 += 1

            else:
                training_set.append(i[0].flatten().numpy())
                training_set_labels.append(1)
                cont_class_1 += 1

    else:
        for i in train_dataset:
            if i[1] == 0 and cont_class_0 < num_training:
                training_set.append(i[0].flatten().numpy())
                training_set_labels.append(-1)
                cont_class_0 += 1
            else:
                if cont_class_1 < num_training:
                    training_set.append(i[0].flatten().numpy())
                    training_set_labels.append(1)
                    cont_class_1 += 1

            if cont_class_0 >= num_training and cont_class_1 >= num_training:
                break

    cont_class_0 = 0
    cont_class_1 = 0

    if num_testing == 0 and num_training == 0:
        for i in test_dataset:
            if i[1] == 0:
                testing_set.append(i[0].flatten().numpy())
                testing_set_labels.append(-1)
                cont_class_0 += 1
            else:
                testing_set.append(i[0].flatten().numpy())
                testing_set_labels.append(1)
                cont_class_1 += 1

    else:
        for i in test_dataset:
            if i[1] == 0 and cont_class_0 < num_testing:
                testing_set.append(i[0].flatten().numpy())
                testing_set_labels.append(-1)
                cont_class_0 += 1
            else:
                if cont_class_1 < num_testing:
                    testing_set.append(i[0].flatten().numpy())
                    testing_set_labels.append(1)
                    cont_class_1 += 1

            if cont_class_0 >= num_testing and cont_class_1 >= num_testing:
                break

                
    numpy.set_printoptions(threshold=sys.maxsize)
    
##    with open ('training_set.txt', 'w') as file:  #Debugging using txt files
##        file.write(str(training_set))
    
    training_set = np.array(training_set)
    training_set = np.array(0.5*(training_set+1))
    training_set_labels = np.array(training_set_labels)
    
##    with open ('training2_set.txt', 'w') as file:  
##        file.write(str(training_set))
##    
##    with open ('testing_set.txt', 'w') as file:  
##        file.write(str(testing_set))
        
    testing_set = np.array(testing_set)
    testing_set = np.array(0.5*(testing_set+1))
    testing_set_labels = np.array(testing_set_labels)

##    with open ('testing2_set.txt', 'w') as file:  
##        file.write(str(testing_set))
        
    print(f"Shapes of...")
    print(f"training set: {training_set.shape}")
    print(f"training labels: {training_set_labels.shape}")
    print(f"testing set: {testing_set.shape}")
    print(f"testing labels: {testing_set_labels.shape}")

    return training_set, training_set_labels, testing_set, testing_set_labels


if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(1)
    np.random.seed(4)

    num_qubits = 27
    epochs = 5

    class_0 = 0
    class_1 = 1
    provider = True
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

    back = 'ibmq_qasm_simulator' # 'ibmq_kolkata' #...

    if back == 'ibmq_qasm_simulator':
        backend_name = 'simulator'

    elif back == 'ibmq_kolkata':
        backend_name = 'kolkata'

    else:
        ValueError("Backend not supported")
        exit(-1)

    backend = back

    print('Preparing data...')

    training_set, training_set_labels, \
    testing_set, testing_set_labels = get_binary_mnist(num_training=num_training,
                                                       num_testing=num_testing,
                                                       class_0=class_0,
                                                       class_1=class_1)

    if provider:
        print('Loading account...')
        IBMQ.load_account()
        if reserv:
            provider = IBMQ.get_provider(hub='ibm-q-qida',
                                         group='iq-quantum',
                                         project='reservations')
            print("Account loaded succesfully!, for RESERVATION.")

        else:
            provider = IBMQ.get_provider(hub='ibm-q-qida',
                                         group='iq-quantum',
                                         project='algolab')
            print("Account loaded succesfully!")

    weights = None
    warm_start = False
    #backend = provider.get_backend(backend)

    for trial in range(trial_start, trial_end+1):
        out_path = f'mnist/{backend_name}/trial{trial}/runs/'

        if incomplete_trial_flag:
            if parameters_path is None:
                ValueError('parameters_path is None.')

            print('Using previous weights...')
            file = open(parameters_path)
            data = json.load(file)
            weights = np.array(data['parameters'][-1])
            warm_start = True
            incomplete_trial_flag = False

        qnn = QuantumPerceptron(n_qubits=num_qubits,
                                input_data=training_set,
                                labels=training_set_labels,
                                test_input_data=testing_set,
                                test_labels=testing_set_labels,
                                backend=backend,
                                epochs=epochs-epoch_correction,
                                provider=provider,
                                shots=shots,
                                lr=lr,
                                perturbation=perturbation,
                                exp_dir=out_path,
                                warm_start=warm_start,
                                initial_point=weights
                                )

        qnn.fit()

    print('Training done...')
