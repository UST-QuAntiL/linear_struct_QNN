
import torch
import qnns.quantum_gates as qg
import numpy as np
import sys

min_freq = 0
max_freq = 4
max_z = 2.8

# setup for predictor for chassis movement on Quantum computer
# includes training from purely classical data and training from existing quantum unitary

def comp_basis(dim, i):
    e = np.zeros((dim))
    e[i] = 1
    return e

# State preparation SP using RX and RZ to encode input as amplitude and phase
def state_prep(input, qubits, min, max, num_qubits):
    # State preparation circuit that does a (scaled) rotation for the given qubit
    X = torch.tensor([
        [0, 1],
        [1, 0]
    ], dtype=torch.complex128, device='cpu')
    amplitude = torch.tensor((input - min)/(max-min) * np.pi)
    phase = torch.tensor((input - min)/(max-min) * np.pi)
    rx = qg.RX(amplitude)
    rz = qg.RZ(phase)
    oplist = [qg.small_I for _ in range(0, num_qubits)]

    for q in qubits:
        oplist[q] = torch.matmul(rx, torch.matmul(rz,X))

    result = torch.kron(oplist[0], oplist[1])
    for i in range(2, num_qubits):
        result = torch.kron(result, oplist[i])

    return result

# state preparation of a random frequency
def random_state_prep_function(sp_qubits, min, max, num_qubits):
    def sp_fun():
        input = np.random.random() * (max-min) + min
        sv = torch.matmul(
            state_prep(input, sp_qubits, min, max, num_qubits),
            torch.from_numpy(np.array(comp_basis(2**num_qubits, 0), dtype=complex))
        )
        return sv
    return sp_fun

# execute whole quantum circuit (including SP and measurement) using supplied unitary
def run_unitary(unitary, frequency_inputs, num_qubits, sp_qubits):
    # Inputs X = shots * the zero state
    input_svs = []
    for frequency_input in frequency_inputs:
        sv = torch.matmul(
            state_prep(frequency_input, sp_qubits, min_freq, max_freq, num_qubits),
            torch.from_numpy(np.array(comp_basis(2**num_qubits, 0), dtype=complex))
        ).reshape(1, 2**num_qubits)
        input_svs.append(sv)
    
    inputs = torch.cat(input_svs).transpose(0,1)

    output_statevectors = torch.matmul(unitary, inputs)
    return output_statevectors

# execute whole circuit using supplied QNN object
def run_qnn(qnn, frequency_inputs, num_qubits, sp_qubits):
    V = qnn.get_tensor_V()

    return run_unitary(V, frequency_inputs, num_qubits, sp_qubits)

# calculate expecatation value by simulating measurement (= chassis movement prediction)
def exp_value(statevectors, qubits, num_qubits):
    # computation measurement probability for 0 in given qubit
    oplist = [qg.small_I for _ in range(0, num_qubits)]
    for q in qubits:
        oplist[q] = qg.one_top_left #project_to_zero

    projector = torch.kron(oplist[0], oplist[1])
    for i in range(2, num_qubits):
        projector = torch.kron(projector, oplist[i])


    sv_H = (statevectors.conj()).transpose(0,1)
    meas_probs = torch.diag(torch.matmul(sv_H, torch.matmul(projector, statevectors))) # diag extracts inner prod from matmul
    meas_probs = meas_probs * max_z # is scaled for maximal output Z
    return meas_probs

# l1 loss for training
def l1(exp_values, ampls):
    return torch.sum(torch.abs(exp_values - ampls))/exp_values.shape[0]

# l2 loss for training
def l2(exp_values, ampls):
    return torch.sum(torch.square(exp_values - ampls))/exp_values.shape[0]

def lossfn(qnn, freqs, ampls, num_qubits, expval_qubits, sp_qubits, lossfunc=l1):
    if isinstance(ampls, list):
        ampls = torch.tensor(ampls)
    
    svs = run_qnn(qnn, freqs, num_qubits, sp_qubits)
    exp_values = exp_value(svs, expval_qubits, num_qubits)
    return lossfunc(exp_values, ampls)

# training from classical data
def train(qnn, freqs, ampls, num_qubits, expval_qubits, num_epochs, optimizer, scheduler, sp_qubits, lossfunc=l1):
    losses = []
    amplts = torch.tensor(ampls)

    for i in range(num_epochs):
        loss = lossfn(qnn, freqs, amplts, num_qubits, expval_qubits, sp_qubits, lossfunc=lossfunc)
        losses.append(loss.item())

        if i%10 == 0:
            print("Step", i)
            print("L1:", lossfn(qnn, freqs, amplts, num_qubits, expval_qubits, sp_qubits, lossfunc=l1).item())
            print("L2:", lossfn(qnn, freqs, amplts, num_qubits, expval_qubits, sp_qubits, lossfunc=l2).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step(np.abs(loss.item())) # is still complex but never has imag -> abs is ok
    
    return losses

def export_unitary(qnn, file):
    mat = qnn.get_matrix_V()
    torch.save(mat, file)

def import_unitary(file):
    return torch.load(file)


# Quantum compiling = Training by directly accessing target operator U
def equally_weighted_entangled_state(dim_X, dim_R, schmidt_rank):
    state_summands = [np.kron(comp_basis(dim_X, i), comp_basis(dim_R, i)) for i in range(0, schmidt_rank)]
    state = np.sum(state_summands, axis=0) * (1/np.sqrt(schmidt_rank))

    return state

# execute V^\dagger U |input>
def run_compiling_qnn(qnn_H, input_states, U):
    s = torch.matmul(U, input_states)
    s = torch.matmul(qnn_H, s)

    return s

# compute loss in quantum compiling = overlap of expected and actual outputs
def compiling_loss(qnn_output, expected_output_conj):
    ip = torch.sum(torch.mul(expected_output_conj, qnn_output), dim=[1, 2]) # computes inner product for each sample independently
    loss = 1 - (torch.sum(torch.square(torch.abs(ip))) / expected_output_conj.shape[0])
    return loss

# training loop for quantum computing with direct access to target transformation
def compiling_train(qnn, input_states, U, num_epochs, optimizer, scheduler):
    losses = []
    input_states = torch.clone(input_states).detach()
    expected_outputs = torch.clone(input_states).detach().conj()

    for i in range(num_epochs):
        comp_output = run_compiling_qnn(qnn.get_tensor_V().H, input_states, U)
        loss = compiling_loss(comp_output, expected_outputs)
        losses.append(loss.item())

        if i%100 == 0:
            print("Step %d - Loss %0.20f" % (i, loss.item()))
            sys.stdout.flush()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step(np.abs(loss.item())) # is still complex but never has imag -> abs is ok
    
    return losses


# non-ortho states
def deorthogonalize(list_of_states):
    deortho_states = []
    for i, state in enumerate(list_of_states):
        if i==0:
            deortho_states.append(state)
        else:
            deortho_states.append((state + deortho_states[i-1]) * (1/np.sqrt(2)))
    
    return deortho_states

