# Linear Structure of Training Samples in Quantum Neural Network Applications

This repository contains the implementations, raw experiment results and plots for the QNN training experiments presented in
"Linear Structure of Training Samples in Quantum Neural Network Applications".

## Experiments

The experiments are parameterized by the settings in ``exp_setup.py``. They train QNNs to approximate the unitary given in ``4_qubit_target_unitary.pt``. The experiment save the trainined hypothesis unitary $V_S$ in their respective directories for further processing. The experiments are executed using the following files.

 - ``exp_entangled.py``: Training QNNs using maximally entangled training states (``MAX_ENT``). The results are saved in ``experiment_entangled/``.
 - ``exp_not_entangled_ld_opr.py``: Training QNNs using Schmidt rank 0 states that are linearly dependent and pairwise nonorthogonal (``LD_NONORTHO``). The results are saved in ``experiment_not_entangled/ld_opr/``
 - ``exp_not_entangled_li_opr.py``: Training QNNs using Schmidt rank 0 states that are linearly independent and pairwise nonorthogonal (``LI_NONORTHO``). The results are saved in ``experiment_not_entangled/li_opr/``
 - ``exp_not_entangled_li_ortho.py``: Training QNNs using Schmidt rank 0 states that are linearly independent and pairwise orthogonal (``LI_ORTHO``). The results are saved in ``experiment_not_entangled/li_ortho/``

## Plots

The trained hypothesis unitaries are further processed by extraing the exact risk and the mean squared error of the classical function results. The data extraction is performed using ``data_extraction.py``. It also supports additional metrics (restricted risk and mean absolute error). The plots are created using the following files.

 - ``plots.py``: Main plot for comparing risk and mean squared error.
 - ``plot_original_function.py``: Plot the classical praditions of ``4_qubit_target_unitary.pt``
 - ``plot_effect.py``: Plots the classical predictions of high and low risk unitaries for ``LD_NONORTHO`` and ``LI_ORTHO``.

## Dependencies

The code contained in this repository requires the following dependencies for reproducing the experiments:
- matplotlib (3.5.2)
- networkx (2.8.8)
- numpy (1.24.1)
- PennyLane (0.27.0)
- scipy (1.10.1)
- torch (2.0.0)

Use ``requirements.txt`` to automatically install them: ``pip install -r requirements.txt``

### Disclaimer of Warranty

Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

### Haftungsausschluss
Dies ist ein Forschungsprototyp. Die Haftung für entgangenen Gewinn, Produktionsausfall, Betriebsunterbrechung, entgangene Nutzungen, Verlust von Daten und Informationen, Finanzierungsaufwendungen sowie sonstige Vermögens- und Folgeschäden ist, außer in Fällen von grober Fahrlässigkeit, Vorsatz und Personenschäden, ausgeschlossen.
