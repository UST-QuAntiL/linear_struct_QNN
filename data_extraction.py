import glob
import numpy as np
from pt_train import *
from risk import *
import os
import sys

# Extracts and collects experiment results as average risks by evaluating the learned hypothesis unitaries
# saves results in "metrics" subdirectory for further processing (see plot*.py files)

sp_qubits = [0,1,2,3]
num_qubits = 4
rrisk_num_evals = 100
mae_risk_num_evals = 100
mse_risk_num_evals = 100
min_freq = 0
max_freq = 4
meas_qubits = [0]

def check_and_replace_lowest_highest(old_values, old_Vs, new_value, new_V):
    ret_values = old_values
    ret_Vs = old_Vs
    # low 
    if new_value <= old_values[0]:
        ret_values[0] = new_value
        ret_Vs[0] = new_V 
    
    # high
    if new_value >= old_values[1]:
        ret_values[1] = new_value
        ret_Vs[1] = new_V

    return ret_values, ret_Vs


def compute_metrics(U, directory):
    hypothesis_unitary_files = glob.glob(directory + "/*_V.npy")
    risks = []
    rrisks = []
    mae_risks = []
    mse_risks = []

    # low, high functions
    lh_risk = [1,0]
    lh_risk_vs = [None, None]
    lh_rrisk = [1,0]
    lh_rrisk_vs = [None, None]
    lh_mae = [1,0]
    lh_mae_vs = [None, None]
    lh_mse = [1,0]
    lh_mse_vs = [None, None]

    print("Computing with sp_qubits: ", sp_qubits)
    print("Computing with min_freq: ", min_freq)
    print("Computing with max_freq: ", max_freq)

    # for restricted risks
    spfun = random_state_prep_function(sp_qubits, min_freq, max_freq, num_qubits)

    for hypo_f in hypothesis_unitary_files:
        V = torch.from_numpy(np.load(hypo_f))

        # risk (as usual)
        risk_val = risk(U, V)
        risks.append(risk_val)
        lh_risk, lh_risk_vs = check_and_replace_lowest_highest(lh_risk, lh_risk_vs, risk_val, V)

        # restricted risk
        rrisk = restricted_risk(spfun, U, V, rrisk_num_evals)
        rrisks.append(rrisk)
        lh_rrisk, lh_rrisk_vs = check_and_replace_lowest_highest(lh_rrisk, lh_rrisk_vs, rrisk, V)

        # MAE risk
        mae_risk = mae_function_risk(
            lambda freqs: exp_value(run_unitary(U, freqs, num_qubits, sp_qubits), qubits=meas_qubits, num_qubits=num_qubits),
            lambda freqs: exp_value(run_unitary(V, freqs, num_qubits, sp_qubits), qubits=meas_qubits, num_qubits=num_qubits),
            num_evals = mae_risk_num_evals,
            min_freq = min_freq,
            max_freq = max_freq
        )
        mae_risks.append(mae_risk)
        lh_mae, lh_mae_vs = check_and_replace_lowest_highest(lh_mae, lh_mae_vs, mae_risk, V)

        # MSE risk
        mse_risk = mse_function_risk(
            lambda freqs: exp_value(run_unitary(U, freqs, num_qubits, sp_qubits), qubits=meas_qubits, num_qubits=num_qubits),
            lambda freqs: exp_value(run_unitary(V, freqs, num_qubits, sp_qubits), qubits=meas_qubits, num_qubits=num_qubits),
            num_evals = mse_risk_num_evals,
            min_freq = min_freq,
            max_freq = max_freq
        )
        mse_risks.append(mse_risk)
        lh_mse, lh_mse_vs = check_and_replace_lowest_highest(lh_mse, lh_mse_vs, mse_risk, V)

    avg_r = np.mean(risks)
    avg_rr = np.mean(rrisks)
    avg_mae = np.mean(mae_risks)
    # using abs since internally its complex but it can only be real at this point
    avg_mse = np.abs(np.mean(mse_risks))

    std_r = np.std(risks)
    std_rr = np.std(rrisks)
    std_mae = np.std(mae_risks)
    std_mse = np.abs(np.std(mse_risks))

    print("Average risk: ", avg_r)
    print("STD risk: ", std_r)
    print("Average restricted risk: ", avg_rr)
    print("STD restricted risk: ", std_rr)
    print("Average MAE risk: ", avg_mae)
    print("STD MAE risk: ", std_mae)
    print("Average MSE risk: ", avg_mse)
    print("STD MSE risk: ", std_mse)

    # save them
    output_directory = directory + "/metrics/"
    if not os.path.exists(output_directory):
        print("creating output directory")
        sys.stdout.flush()
        os.makedirs(output_directory)

    np.save(output_directory + "means.npy", [avg_r, avg_rr, avg_mae, avg_mse])
    np.save(output_directory + "stds.npy", [std_r, std_rr, std_mae, std_mse])
    np.save(output_directory + "lowest_risk_V.npy", lh_risk_vs[0])
    np.save(output_directory + "lowest_rrisk_V.npy", lh_rrisk_vs[0])
    np.save(output_directory + "lowest_mae_V.npy", lh_mae_vs[0])
    np.save(output_directory + "lowest_mse_V.npy", lh_mse_vs[0])
    np.save(output_directory + "highest_risk_V.npy", lh_risk_vs[1])
    np.save(output_directory + "highest_rrisk_V.npy", lh_rrisk_vs[1])
    np.save(output_directory + "highest_mae_V.npy", lh_mae_vs[1])
    np.save(output_directory + "highest_mse_V.npy", lh_mse_vs[1])


if __name__ == '__main__':
    U = import_unitary("4_qubit_target_unitary.pt")
    print("Entangled")
    compute_metrics(U, "experiment_entangled")
    print("LD_NONORTHO")
    compute_metrics(U, "experiment_not_entangled/ld_opr")
    print("LI_NONORTHO")
    compute_metrics(U, "experiment_not_entangled/li_opr")
    print("LI_ORTHO")
    compute_metrics(U, "experiment_not_entangled/li_ortho")
