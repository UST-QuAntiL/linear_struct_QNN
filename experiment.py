import os
import sys
import torch
import numpy as np
import time
import copy

from pt_train import *
from concurrent.futures import ProcessPoolExecutor

# handles parallelization of training loops in QNN training experiments
# trains and saves resulting hypothesis unitaries 

# set globally here, wont change
lr = 0.1
sched_fact = 0.5
sched_pat = 10
num_epochs = 1000


def save_network(qnn, losses, output_directory, run_identifier):
    file_id = output_directory + "/" + run_identifier

    qnn_numpy = qnn.get_matrix_V().numpy()
    np.save(file_id + "_V.npy", qnn_numpy)
    np.save(file_id + "_losses.npy", losses)


def train_network(args):
    (U, output_directory, training_data_callback, qnn, run_identifier) = args 

    target = copy.deepcopy(U)

    opt = None
    if isinstance(qnn.params, list):
        opt = torch.optim.Adam(qnn.params, lr=lr)
    else:
        opt = torch.optim.Adam([qnn.params], lr=lr)

        
    Y = training_data_callback()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=sched_fact, patience=sched_pat, min_lr=1e-10, verbose=False)

    losses = compiling_train(qnn, Y, target, num_epochs, opt, sched)
    print("Final loss: ", losses[-1])
    sys.stdout.flush()


    save_network(qnn, losses, output_directory, run_identifier)


def train_networks(U, output_directory, training_data_callback, num_tries, qnn_callback, num_processes):
    """training data callback: gives tensor cont. set of possibly entangled states"""
    
    if not os.path.exists(output_directory):
        print("creating output directory")
        sys.stdout.flush()
        os.makedirs(output_directory)

    # parallelization only via multiprocessing of tries
    torch.set_num_threads(1)

    inputs = []
    t = time.time()
    for run in range(0, num_tries):
        id = str(t) + "_" + str(run)
        inputs.append((U.clone().detach(), output_directory, training_data_callback, qnn_callback(), id))

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = executor.map(train_network, inputs)

        for res in results:
            print(res)


    