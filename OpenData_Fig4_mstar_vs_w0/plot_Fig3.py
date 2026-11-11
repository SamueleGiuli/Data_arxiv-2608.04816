"""Fig. 3: m*/m vs omega0/D at U = 5D for n = 0.9, 0.7, 0.5; ghost-RISB (B=1,3,5) vs DMFT.
Usage: python plot_Fig3.py   (reads Fig3_mstar_vs_w0.h5, writes Fig3_mstar_vs_w0.pdf)
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

DENS = [0.9, 0.7, 0.5]
LAMS = [0.1, 0.5, 1.0]
L_MARK = {0.1: "o", 0.5: "d", 1.0: "v"}
YMAX = [12, 4.4, 2.7]
GREENS = cm.Greens(np.linspace(0.3, 0.9, 3))
B_STYLE = {1: (GREENS[0], "-"), 3: (GREENS[1], "--"), 5: (GREENS[2], "-.")}

plt.rcParams.update({"font.size": 11, "axes.labelsize": 16, "axes.titlesize": 14, "legend.fontsize": 11,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "xtick.direction": "in", "ytick.direction": "in", "lines.linewidth": 1.6})

h = h5py.File("Fig3_mstar_vs_w0.h5", "r")
fig, axs = plt.subplots(3, figsize=(5, 8), sharex=True)
fig.subplots_adjust(hspace=0)
for ax, n, ymax in zip(axs, DENS, YMAX):
    for L in LAMS:
        for B in (1, 3, 5):
            d = h[f"n_{n:g}/lambda_{L:g}/B{B}"]
            c, ls = B_STYLE[B]
            ax.plot(d["omega0"][:], d["mstar"][:], color=c, ls=ls, marker=L_MARK[L], mfc="white", mec=c, mew=1,
                    alpha=0.9)
        d = h[f"n_{n:g}/lambda_{L:g}/DMFT"]
        ax.plot(d["omega0"][:], d["mstar"][:], color="black", ls=":", lw=2, marker=L_MARK[L], mfc="none",
                mec="black", mew=1.2, alpha=0.9)
    ax.text(20, 0.9 * ymax, f"n={n:.1f}", bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))
    ax.set_xscale("log")
    ax.set_xlim(0.1, 50)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(r"$m^*/m$")
axs[-1].set_xlabel(r"$\omega_0/D$")
axs[-1].set_yticks([0, 1, 2])
axs[0].set_title(r"$U/D=5.0$")

leg1 = axs[2].legend(handles=[Line2D([0], [0], color="gray", marker=L_MARK[L], ls="none", label=rf"$\lambda/D={L}$")
                              for L in LAMS], loc="lower right", fancybox=False, framealpha=1, edgecolor="black",
                     markerfirst=False)
axs[2].legend(handles=[Line2D([0], [0], color="black", lw=2, ls=":", label="DMFT")] +
              [Line2D([0], [0], color=B_STYLE[B][0], lw=2, ls=B_STYLE[B][1], label=rf"$\mathcal{{B}}={B}$")
               for B in (5, 3, 1)], loc="lower left")
axs[2].add_artist(leg1)
fig.savefig("Fig3_mstar_vs_w0.pdf")
