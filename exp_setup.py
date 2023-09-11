from pt_train import *

# globals for all experiments is reused in each experiment file

num_qubits = 4
num_layers = 25
N = 2**num_qubits
U = import_unitary("4_qubit_target_unitary.pt")

# number of runs for each experiment
num_tries = 100
# number of processes supported on system
num_processes = 24