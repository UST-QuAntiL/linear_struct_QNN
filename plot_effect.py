from matplotlib.pyplot import figure
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from exp_setup import *
import torch

# Plot chassis-movement predition of the highest risk and lowest risk hypothesis unitaries for LD_NONORTHO and LI_ORTHO

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "text.latex.preamble":  "".join([r'\usepackage{amssymb}', r'\usepackage{amsmath}']),
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)

def set_size(width, fraction=1, h_fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Modified: dont affect height if fraction is given
    fig_height_in = (width * inches_per_pt) * golden_ratio
    #fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in*h_fraction)

    return fig_dim


textfontsize = 10

fig = plt.figure(figsize=set_size(347, h_fraction=0.7))

plt.ylabel("Amplitude", fontsize=textfontsize)
plt.xlabel("Frequency $\\nu$", fontsize=textfontsize)

plt.ylim([0,2.5])
plt.xlim([0,4])



cmap = mpl.cm.get_cmap('copper_r')
cmap.set_gamma(2.0)
norm = mpl.colors.BoundaryNorm([0,1,2,3,4], cmap.N)

plt.grid()
msize=6

errorlinewidth=1



testfreqs = np.arange(0, 4, step=0.1)
sp_qubits = [0,1,2,3]
meas_qubits = [0]

# Plot original fn
U = import_unitary("4_qubit_target_unitary.pt")

# Outputs from original unitary
output_svs_U = run_unitary(U, testfreqs, num_qubits, sp_qubits)
exp_values_U = exp_value(output_svs_U, qubits=meas_qubits, num_qubits=num_qubits)
plt.plot(testfreqs, np.real(exp_values_U.detach().numpy()), label="Original function $W_U(\\nu)$")

def plot_learned_function(directory, selector="highest", measurement="mae", cidx=0, label=None):
    linestyle = "solid" if selector=="lowest" else "dashed"

    V = torch.from_numpy(np.load(directory + "/metrics/" + selector + "_" + measurement + "_V.npy"))
    output_svs_V = run_unitary(V, testfreqs, num_qubits, sp_qubits)
    exp_values_V = exp_value(output_svs_V, qubits=meas_qubits, num_qubits=num_qubits)
    plt.plot(testfreqs, 
        np.real(exp_values_V.detach().numpy()), 
        linewidth=0.7, 
        color=cmap(norm(float(cidx))), 
        linestyle=linestyle, 
        label=label)


# Outputs with highest mae
plot_learned_function("experiment_not_entangled/ld_opr", selector="highest", cidx=1)
plot_learned_function("experiment_not_entangled/li_ortho", selector="highest", cidx=3)

# Outputs with lowest mae
plot_learned_function("experiment_not_entangled/ld_opr", selector="lowest", cidx=1, label="$W_{V_S}(\\nu)$ (\\texttt{LD_NONORTHO})")
plot_learned_function("experiment_not_entangled/li_ortho", selector="lowest", cidx=3, label="$W_{V_S}(\\nu)$ (\\texttt{LI_ORTHO})")

plt.legend()

fig.savefig('plot_effect.pdf', format='pdf', bbox_inches='tight')
plt.show()