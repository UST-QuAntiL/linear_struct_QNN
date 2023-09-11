import experiment
from pt_train import *
from qnns.cuda_qnn import CudaPennylane, CudaEfficient
from exp_setup import *

# experiments for maximally entangled trianing states

def qnn():
    return CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device='cpu')

def training_data():
    ew = np.array([equally_weighted_entangled_state(N, N, N)])
    t = ew.shape[0]
    Y = torch.from_numpy(ew).to(torch.complex128)
    Y = Y.reshape((Y.shape[0], int(Y.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)
    return Y

experiment.train_networks(U, "experiment_entangled", training_data, num_tries, qnn, num_processes)