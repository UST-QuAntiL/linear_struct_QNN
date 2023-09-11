from matplotlib.pyplot import figure
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from exp_setup import *
import torch

# Plot the chassis movement predictions of the original predictor given in 4_qubit_target_unitary.pt

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

fig = plt.figure(figsize=set_size(347/2, h_fraction=0.8))
subpl = [None, fig.subplots(1, 1)]

subpl[1].set_ylabel("Amplitude ratio", fontsize=textfontsize)
subpl[1].set_xlabel("Frequency $\\nu$ (Hz)", fontsize=textfontsize)


plt.ylim([0,1.8])
plt.xlim([0,4])



cmap = mpl.cm.get_cmap('copper_r')
cmap.set_gamma(2.0)
norm = mpl.colors.BoundaryNorm([0,1,2,3,4], cmap.N)

subpl[1].grid()
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
subpl[1].plot(testfreqs, np.real(exp_values_U.detach().numpy()), label="original function")

fig.savefig('plot_original_function.pdf', format='pdf', bbox_inches='tight')
fig.show()