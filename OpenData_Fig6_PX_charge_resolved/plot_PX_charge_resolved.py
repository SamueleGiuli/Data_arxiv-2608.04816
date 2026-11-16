"""Fig. 6: total and charge-resolved phonon-displacement distributions, ED-DMFT (lines) vs ghost-RISB B=3 (squares).
Usage: python plot_PX_charge_resolved.py   (reads PX_charge_resolved.h5, writes PX_charge_resolved.pdf)
The plotting shifts DMFT_plot_xshift / gRISB_plot_xshift of each panel are applied here (see h5 attributes).
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

h = h5py.File("PX_charge_resolved.h5", "r")
panels = [h["U_0_lambda_1.1_n_1"], h["U_5_lambda_2.2_n_0.7"]]
panel_labels = [r"$U/D=0, \, \lambda/D=1.1, \, n=1$", r"$U/D=5, \, \lambda/D=2.2, \, n=0.7$"]
grisb_marker_stride = 2   # g-RISB markers on every 2nd grid point

ngh_colors = cm.Greens(np.linspace(0.3, 0.9, 3))   # n = 0, 1, 2
curve_colors = {"total": "black", 0: ngh_colors[0], 1: ngh_colors[1], 2: ngh_colors[2]}
curve_labels = {"total": "total", 0: r"$n=0$", 1: r"$n=1$", 2: r"$n=2$"}
label_fs, tick_fs, legend_fs = 20, 15, 11
dmft_line_width = 3.0
grisb_marker_size = 6.0


def markers(ax, x, p, shift, color):
    ax.plot((x + shift)[::grisb_marker_stride], p[::grisb_marker_stride], linestyle="none", marker="s",
            markersize=grisb_marker_size, markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.3,
            zorder=4)


fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True, sharey=False, gridspec_kw={"hspace": 0.0})

for ax, g, label in zip(axes, panels, panel_labels):
    ed_x = g["DMFT/X"][:] + g.attrs["DMFT_plot_xshift"]
    gx = g["B3/X"][:]
    gs = g.attrs["gRISB_plot_xshift"]

    # total: ED-DMFT black line, g-RISB black squares
    ax.plot(ed_x, g["DMFT/P_X"][:], color=curve_colors["total"], linestyle="-", linewidth=dmft_line_width,
            solid_capstyle="round", zorder=2)
    markers(ax, gx, g["B3/P_X"][:], gs, curve_colors["total"])

    # charge-resolved joint distributions
    for sector in (0, 1, 2):
        ax.plot(ed_x, g["DMFT/P_X_n"][:, sector], color=curve_colors[sector], linestyle="-",
                linewidth=dmft_line_width, solid_capstyle="round", zorder=2)
        markers(ax, gx, g["B3/P_X_n"][:, sector], gs, curve_colors[sector])

    ax.set_xlim(-3, 3)
    ax.set_ylabel(r"$\mathrm{P}(\mathrm{X})$", fontsize=label_fs)
    ax.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs, direction="out", length=5, width=1)
    ax.tick_params(axis="both", which="minor", direction="out", length=2.5, width=0.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    parameter_legend = ax.legend(handles=[Line2D([], [], color="none", label=label)], loc="upper right",
                                 frameon=True, fancybox=True, facecolor="0.92", edgecolor="0.50", framealpha=1.0,
                                 fontsize=14, handlelength=0, handletextpad=0, borderpad=0.35)
    ax.add_artist(parameter_legend)

# Method legend: lower panel only
method_handles = [
    Line2D([0], [0], color="0.45", linestyle="-", linewidth=dmft_line_width, solid_capstyle="round", label="DMFT"),
    Line2D([0], [0], color="0.45", linestyle="none", marker="s", markersize=grisb_marker_size,
           markerfacecolor="none", markeredgecolor="0.45", markeredgewidth=1.3, label=r"$\mathcal{B}=3$"),
]
method_legend = axes[1].legend(handles=method_handles, loc="center right", bbox_to_anchor=(0.995, 0.75),
                               frameon=True, fancybox=True, facecolor="white", edgecolor="0.75", framealpha=1.0,
                               fontsize=legend_fs, handlelength=2.7, handletextpad=0.8, borderpad=0.55,
                               labelspacing=0.4)
axes[1].add_artist(method_legend)

# Total / charge legend: lower panel only
component_handles = [Line2D([0], [0], color=curve_colors[k], linestyle="-", linewidth=dmft_line_width,
                            solid_capstyle="round", label=curve_labels[k]) for k in ("total", 0, 1, 2)]
component_legend = axes[1].legend(handles=component_handles, loc="center right", frameon=True, fancybox=False,
                                  facecolor="white", edgecolor="black", framealpha=1.0, fontsize=legend_fs,
                                  handlelength=2.4, handletextpad=0.8, borderpad=0.5, labelspacing=0.3)
component_legend.get_frame().set_linewidth(1.2)

axes[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
axes[0].spines["bottom"].set_visible(False)
axes[-1].set_xlabel(r"$\mathrm{X}$", fontsize=label_fs)
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.11, top=0.98)
fig.savefig("PX_charge_resolved.pdf", bbox_inches="tight")
