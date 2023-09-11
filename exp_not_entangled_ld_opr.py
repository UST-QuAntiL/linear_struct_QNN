import experiment
from pt_train import *
from qnns.cuda_qnn import CudaPennylane, CudaEfficient
from exp_setup import *
from data_extended import sample_non_lihx_points, check_non_lihx_points
import numpy as np

# experiments for LD_NONORTHO training samples

def qnn():
    return CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device='cpu')

def training_data():
    # ensure properties are met (although this always should be the case)
    # if not - rebuild
    while True:
        X = np.array(sample_non_lihx_points(schmidt_rank = 1, size = N, x_qubits=num_qubits, r_qubits = num_qubits, modify=True))
        is_ok, reason = check_non_lihx_points(X, schmidt_rank=1, x_qubits=num_qubits, r_qubits=num_qubits)
        
        if is_ok:
            Y = torch.from_numpy(X).to(torch.complex128)
            Y = Y.reshape((Y.shape[0], int(Y.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)
            return Y

experiment.train_networks(U, "experiment_not_entangled/ld_opr", training_data, num_tries, qnn, num_processes)
