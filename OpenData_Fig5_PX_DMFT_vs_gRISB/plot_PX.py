"""Fig. 5: phonon-displacement distribution P(x), DMFT vs ghost-RISB (B=1, B=3), for U/D=0 (top) and U/D=1 (bottom).
Usage: python plot_PX.py   (reads PX_DMFT_vs_gRISB.h5, writes PX_DMFT_vs_gRISB.pdf)
Plot conventions of the paper figure (see the h5 attribute plot_conventions): g-RISB plotted as x[:-1] -> P[1:],
DMFT shifted by DMFT_plot_xshift and (U=0) sub-sampled with DMFT_plot_stride.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

h = h5py.File("PX_DMFT_vs_gRISB.h5", "r")
panels = [h["U_0"], h["U_1"]]
u_labels = [r"$U/D = 0$", r"$U/D = 1$"]


def grisb(g):
    """Paper plotting convention for the g-RISB files: x[:-1] paired with P[1:]."""
    return g["x"][:-1], g["P_x"][1:]


lambda_markers = ["o", "D", "v", "s"]   # lambda encoded by marker shape
dmft_color = "black"
ngh_colors = cm.Greens(np.linspace(0.3, 0.9, 3))   # B = 1, 3, 5 (B = 5 not plotted)
b1_color, b3_color = ngh_colors[0], ngh_colors[1]
label_fs, tick_fs, legend_fs = 20, 15, 11
line_width = 2.0

fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True, sharey=False, gridspec_kw={"hspace": 0.0})

for panel_idx, (ax, gu) in enumerate(zip(axes, panels)):
    lams = gu.attrs["lambdas"]
    for i, lam in enumerate(lams):
        marker = lambda_markers[i]
        gl = gu[f"lambda_{lam:g}"]

        # DMFT: black dashed line with open markers
        s = gl.attrs["DMFT_plot_stride"]
        x_b, p_b = gl["DMFT/x"][::s], gl["DMFT/P_x"][::s]
        ax.plot(x_b + gu.attrs["DMFT_plot_xshift"], p_b, color=dmft_color, linestyle="--", linewidth=1.5,
                marker=marker, markersize=5.5, markerfacecolor="none", markeredgecolor=dmft_color,
                markeredgewidth=1.3, solid_capstyle="round", zorder=3)

        # B = 1: light-green solid line
        x_g0, p_g0 = grisb(gl["B1"])
        ax.plot(x_g0, p_g0, color=b1_color, linestyle="-", linewidth=line_width, marker=marker, markersize=6,
                markerfacecolor="white", markeredgecolor=b1_color, markeredgewidth=1.3, solid_capstyle="round",
                zorder=1)

        # B = 3: medium-green solid line
        x_g2, p_g2 = grisb(gl["B3"])
        ax.plot(x_g2, p_g2, color=b3_color, linestyle="-", linewidth=line_width, marker=marker, markersize=5.5,
                markerfacecolor="white", markeredgecolor=b3_color, markeredgewidth=1.2, solid_capstyle="round",
                zorder=2)

    ax.set_ylabel(r"$\mathrm{P}(\mathrm{x})$", fontsize=label_fs)
    ax.set_xlim(-3, 3)
    ax.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs, direction="out", length=5, width=1)
    ax.tick_params(axis="both", which="minor", direction="out", length=2.5, width=0.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.text(0.98, 0.95, u_labels[panel_idx], transform=ax.transAxes, ha="right", va="top", fontsize=15,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "0.92", "edgecolor": "0.50", "linewidth": 1.1},
            zorder=10)

    # Method legend: lower panel only, upper-left corner
    if panel_idx == 1:
        method_handles = [
            Line2D([0], [0], color=dmft_color, linestyle="--", linewidth=line_width, solid_capstyle="round",
                   label="DMFT"),
            Line2D([0], [0], color=b1_color, linestyle="-", linewidth=line_width, solid_capstyle="round",
                   label=r"$\mathcal{B}=1$"),
            Line2D([0], [0], color=b3_color, linestyle="-", linewidth=line_width, solid_capstyle="round",
                   label=r"$\mathcal{B}=3$"),
        ]
        method_legend = ax.legend(handles=method_handles, loc="upper left", frameon=True, fancybox=True,
                                  facecolor="white", edgecolor="0.75", framealpha=1.0, fontsize=legend_fs,
                                  handlelength=2.7, handletextpad=0.8, borderpad=0.55, labelspacing=0.4)
        ax.add_artist(method_legend)

    # Lambda legend (markers to the right of the labels)
    lambda_handles = [
        Line2D([0], [0], linestyle="none", marker=lambda_markers[i], markersize=7, markerfacecolor="0.55",
               markeredgecolor="0.55", markeredgewidth=1, label=rf"$\lambda/D = {lam:g}$")
        for i, lam in enumerate(lams)
    ]
    lambda_legend = ax.legend(handles=lambda_handles, loc="lower right", markerfirst=False, frameon=True,
                              fancybox=False, facecolor="white", edgecolor="black", framealpha=1.0,
                              fontsize=legend_fs, handlelength=1.0, handletextpad=1.2, borderpad=0.5,
                              labelspacing=0.3)
    lambda_legend.get_frame().set_linewidth(1.2)

axes[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
axes[0].spines["bottom"].set_visible(False)
axes[-1].set_xlabel(r"$\mathrm{x}$", fontsize=label_fs)
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.11, top=0.98)
fig.savefig("PX_DMFT_vs_gRISB.pdf", bbox_inches="tight")
