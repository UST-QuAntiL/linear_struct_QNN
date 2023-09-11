from matplotlib.pyplot import figure
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

# Plot of average risks after training using data obtained from data_extraction.py

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "text.latex.preamble":  "".join([r'\usepackage{amssymb}', r'\usepackage{amsmath}']),
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
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

def plot_risk_and_mse(fname):
    d = 16
    r = 1
    t = 16
    risk_bound_li_ortho = 1 - (r**2 * t + d)/(d*(d+1))
    risk_bound_ld_opr = 1 - (r**2 + d)/(d*(d+1))

    fig = plt.figure(figsize=set_size(347, h_fraction=0.7))
    subpl = fig.subplots(1, 2, sharey=True)

    x = [0,10,20,30,40,50]
    labels = ['', '\\texttt{MAX\\_ENT}', '\\texttt{LI\\_NONORTHO}', '\\texttt{LD\\_NONORTHO}', '\\texttt{LI\\_ORTHO} ', '']
    # left
    subpl[0].set_xticks(x, labels, rotation='45')
    subpl[0].grid()
    subpl[0].set_ylabel("Average $\\hat{R}_{\\text{MSE}}(U,V_S)$")

    #right
    subpl[1].set_xticks(x, labels, rotation='45')
    subpl[1].grid()
    subpl[1].set_ylabel("Average $R(U,V_S)$")

    

    cmap = mpl.cm.get_cmap('copper_r')
    cmap.set_gamma(2.0)
    norm = mpl.colors.BoundaryNorm([0,1,2,3,4], cmap.N)

    markers = {
        0: "o",
        1: "v",
        2: "*",
        3: "s",
        4: "P"
    }
    msize=4

    errorlinewidth=0.8


    # MSE
    def subplot_dir(middlepos, dirname, subp, assign_label=False):
        means = np.load(dirname + "/metrics/means.npy")
        stds = np.load(dirname + "/metrics/stds.npy")
        
        subp[0].errorbar(middlepos, means[3], yerr=stds[3], color=cmap(norm(float(3))), elinewidth=errorlinewidth, capsize=3)
        subp[0].plot(middlepos, means[3], marker=markers[0], color=cmap(norm(float(3))), markersize=msize)#, markersize=msize)

        subp[1].errorbar(middlepos, means[0], yerr=stds[0], color=cmap(norm(float(3))), elinewidth=errorlinewidth, capsize=3)
        if assign_label:
            subp[1].plot(middlepos, means[0], marker=markers[0], color=cmap(norm(float(3))), linestyle="None", markersize=msize, label="Average error")#, markersize=msize)
        else:
            subp[1].plot(middlepos, means[0], marker=markers[0], color=cmap(norm(float(3))), markersize=msize)
    
    

    subplot_dir(10, "experiment_entangled", subpl, assign_label=True)
    subplot_dir(20, "experiment_not_entangled/li_opr", subpl)
    subplot_dir(30, "experiment_not_entangled/ld_opr", subpl)
    subplot_dir(40, "experiment_not_entangled/li_ortho", subpl)
    
    # the theoretical bounds for the risk
    # MAX ENT
    subpl[1].hlines(y = 0, xmin = 7, xmax = 13, linewidth=1.5, linestyle='dotted', label="Expected risk")
    subpl[1].hlines(y = 0, xmin = 17, xmax = 23, linewidth=1.5, linestyle='dotted')
    subpl[1].hlines(y = risk_bound_ld_opr, xmin = 27, xmax = 33, linewidth=1.5, linestyle='dotted')
    subpl[1].hlines(y = risk_bound_li_ortho, xmin = 37, xmax = 43, linewidth=1.5, linestyle='dotted')

    fig.legend(loc='lower left', bbox_to_anchor=(0.12, 0.87), ncol=2)


    fig.savefig(fname, format='pdf', bbox_inches='tight')


def plot_and_output(fname, mse_only=False):
    fig = plt.figure(figsize=set_size(347, h_fraction=0.7))
    
    plt.ylabel("Average $\\hat{R}_{\\text{MSE}}(V_S)$")

    plt.ylim([-0.1,1.2])
    plt.xlim([0,50])

    markers = {
        0: "o",
        1: "v",
        2: "*",
        3: "s",
        4: "P"
    }


    cmap = mpl.cm.get_cmap('copper_r')
    cmap.set_gamma(2.0)
    norm = mpl.colors.BoundaryNorm([0,1,2,3,4], cmap.N)

    plt.grid()
    msize=4

    errorlinewidth=0.8

    def plot_directory(middlepos, dirname):
        means = np.load(dirname + "/metrics/means.npy")
        stds = np.load(dirname + "/metrics/stds.npy")

        if not mse_only:
            idx = 0
            plt.errorbar(middlepos-2 + idx, means[idx], yerr=stds[idx], color=cmap(norm(float(idx))), elinewidth=errorlinewidth)
            plt.plot(middlepos-2 + idx, means[idx], marker=markers[idx], color=cmap(norm(float(idx))), markersize=msize)#, markersize=msize)

            idx = 1
            plt.errorbar(middlepos-2 + idx, means[idx], yerr=stds[idx], color=cmap(norm(float(idx))), elinewidth=errorlinewidth)
            plt.plot(middlepos-2 + idx, means[idx], marker=markers[idx], color=cmap(norm(float(idx))), markersize=msize)#, markersize=msize)

            idx = 2
            plt.errorbar(middlepos-2 + idx, means[idx], yerr=stds[idx], color=cmap(norm(float(idx))), elinewidth=errorlinewidth)
            plt.plot(middlepos-2 + idx, means[idx], marker=markers[idx], color=cmap(norm(float(idx))), markersize=msize)#, markersize=msize)

            idx = 3
            plt.errorbar(middlepos-2 + idx, means[idx], yerr=stds[idx], color=cmap(norm(float(idx))), elinewidth=errorlinewidth)
            plt.plot(middlepos-2 + idx, means[idx], marker=markers[idx], color=cmap(norm(float(idx))), markersize=msize)#, markersize=msize)
        else:
            plt.errorbar(middlepos, means[3], yerr=stds[3], color=cmap(norm(float(3))), elinewidth=errorlinewidth, capsize=3)
            plt.plot(middlepos, means[3], marker=markers[0], color=cmap(norm(float(3))), markersize=msize)#, markersize=msize)

    plot_directory(10, "experiment_entangled")
    plot_directory(20, "experiment_not_entangled/li_opr")
    plot_directory(30, "experiment_not_entangled/ld_opr")
    plot_directory(40, "experiment_not_entangled/li_ortho")

    x = [0,10,20,30,40,50]
    labels = ['', '\\texttt{MAX\\_ENT}', '\\texttt{LI\\_OPR}', '\\texttt{LD\\_OPR}', '\\texttt{LI\\_ORTHO} ', '']
    plt.xticks(x, labels, rotation='45')

    fig.savefig(fname, format='pdf', bbox_inches='tight')

# Plots all risk measures that are defined in risk.py
plot_and_output('plot_risks_all.pdf', mse_only=False)
# Plot MSE risk only
plot_and_output('plot_mse.pdf', mse_only=True)
# Plots exact risk R(U,V_S) and MSE risk
plot_risk_and_mse('plot_combined.pdf')