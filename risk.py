import torch
import numpy as np



# Different function to evaluate/approximate the risk

# Exact risk - computed with direct access to U and V
def risk(U, V):
    dim = len(U)
    prod = torch.matmul(U.T.conj(), V)
    tr = abs(torch.trace(prod))**2
    risk = 1 - ((dim + tr)/(dim * (dim+1)))

    return risk

# Restricted risk: approximate risk by only allowing states that are prepared by SP
def restricted_risk(state_prep_function, U, V, num_evals):
    '''Risk restricted to states that can be produced by state prep.
    state_prep_function is a function that returns a state that is prepared according to the state preparation
    the returned states are supposed to be uniformly distributed'''
    states = [state_prep_function() for _ in range(0,num_evals)]
    sum = 0
    for state in states:
        expected_output_conj = torch.matmul(U,state).conj()
        actual_output = torch.matmul(V,state)
        inner_product = torch.sum(torch.mul(expected_output_conj, actual_output))
        fid = torch.square(torch.abs(inner_product))
        sum += fid

    return 1 - (sum/num_evals)

def get_exps(eval_exp_U, eval_exp_V, num_evals, min_freq, max_freq):
    f_inputs = [np.random.random() * (max_freq - min_freq) + min_freq for _ in range(0, num_evals)]
    exp_U = eval_exp_U(f_inputs).detach().numpy()
    exp_V = eval_exp_V(f_inputs).detach().numpy()
    return (exp_U, exp_V)

# Mean absolute error over classical function predictions W_V(f)
def mae_function_risk(eval_exp_U, eval_exp_V, num_evals, min_freq, max_freq):
    '''computes mae function risk
    eval_exp_U/V: Functions that return a list of expecatation values (amplitude) for a list of frequency inputs'''
    exp_U, exp_V = get_exps(eval_exp_U, eval_exp_V, num_evals, min_freq, max_freq)

    sum = 0
    for i in range(0, num_evals):
        sum += np.abs(exp_U[i] - exp_V[i])
    
    return sum / num_evals

# Mean squared error over classical function predictions W_V(f)
def mse_function_risk(eval_exp_U, eval_exp_V, num_evals, min_freq, max_freq):
    '''computes mae function risk
    eval_exp_U/V: Functions that return a list of expecatation values (amplitude) for a list of frequency inputs'''
    exp_U, exp_V = get_exps(eval_exp_U, eval_exp_V, num_evals, min_freq, max_freq)

    sum = 0
    for i in range(0, num_evals):
        sum += np.square(exp_U[i] - exp_V[i])
    
    return sum / num_evals

