import experiment
from pt_train import *
from qnns.cuda_qnn import CudaPennylane, CudaEfficient
from exp_setup import *
from data_extended import uniformly_sample_orthogonal_points
import numpy as np

# experiments for LI_ORTHO training samples

def qnn():
    return CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device='cpu')

def training_data():
    X = np.array(uniformly_sample_orthogonal_points(schmidt_rank = 1, size = N, x_qubits=num_qubits, r_qubits = 0, modify=True))
    Y = torch.from_numpy(X).to(torch.complex128)
    Y = Y.reshape((Y.shape[0], int(Y.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)
    return Y.detach().clone()

experiment.train_networks(U, "experiment_not_entangled/li_ortho", training_data, num_tries, qnn, num_processes)
