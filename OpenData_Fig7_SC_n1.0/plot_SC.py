"""Figs. 6 and 7: |Delta| and I_dh vs lambda (top) and phonon-displacement distribution P(X) (bottom), B=1 vs B=3.
Usage: python plot_SC.py 6   (reads Fig6_SC_n1.0.h5, writes Fig6_SC_n1.0.pdf)
       python plot_SC.py 7   (reads Fig7_SC_n0.5.h5, writes Fig7_SC_n0.5.pdf)
"""
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 11, "axes.labelsize": 16, "axes.titlesize": 14, "legend.fontsize": 11,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "xtick.direction": "in", "ytick.direction": "in"})   # same font sizes in all figures

FIG = {  # lambda at the colormap peak, colorbar max, P(X) x-range, P(X) every n-th lambda, Delta-panel x/y ticks
    6: dict(name="Fig6_SC_n1.0", Ltrg=1.1, vmax=2.0, xP=3, every=3, xticks=[0, 0.5, 1, 1.5], ylim=0.35),
    7: dict(name="Fig7_SC_n0.5", Ltrg=1.6, vmax=2.5, xP=4, every=4, xticks=[0, 0.5, 1, 1.5, 2.0, 2.5], ylim=0.25),
}


def peaked_mixed_cmap(low_color, center_color, high_color, center_value, vmin, vmax, n=256):
    """low -> center -> high colormap with an exponentially sharp peak at center_value."""
    low, center, high = (np.array(colors.to_rgb(c)) for c in (low_color, center_color, high_color))
    cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
    x = np.linspace(0, 1, n)
    xc = cnorm(center_value)
    w = np.exp(-np.abs(x - xc) * 10)[:, None]
    rgb = np.where((x < xc)[:, None], (1 - w) * low + w * center, w * center + (1 - w) * high)
    return colors.ListedColormap(rgb), cnorm


def fade_band(ax, x0, x1, color, max_alpha=0.6, n=400):
    """Vertical band between x0 and x1 whose opacity fades to zero at the edges."""
    ymin, ymax = ax.get_ylim()
    img = np.ones((2, n, 4))
    img[..., :3] = colors.to_rgb(color)
    img[..., 3] = max_alpha * np.sin(np.linspace(0, np.pi, n)) ** 2
    ax.imshow(img, extent=(x0, x1, ymin, ymax), origin="lower", aspect="auto", interpolation="bicubic", zorder=0)


fig_id = int(sys.argv[1]) if len(sys.argv) > 1 else 6
p = FIG[fig_id]
h = h5py.File(f"{p['name']}.h5", "r")
cmap, cnorm = peaked_mixed_cmap("lightgray", "darkgreen", "black", p["Ltrg"], 0.1, p["vmax"])

fig, axs = plt.subplots(2, 1, figsize=(4, 6))
for B in (1, 3):
    g = h[f"B{B}"]
    lam = g["lambda"][:]
    style = dict(marker=".", linestyle="") if B == 1 else dict(marker="", linestyle="-")
    axs[0].plot(lam, g["I_dh"][:], color="red", alpha=0.8, **style)
    axs[0].plot(lam, g["Delta"][:], color="black", alpha=0.8, **style)
    X = g["X"][:]
    for iL, L in enumerate(lam):
        if iL % p["every"] == 0 or abs(L - p["Ltrg"]) <= 0.3:
            axs[1].plot(X, g["P_X"][iL], color=cmap(cnorm(L)), **(dict(style, alpha=0.5) if B == 1 else style))

axs[0].set_xlabel(r"$\lambda$")
axs[0].set_ylabel(r"$| \Delta |$")
axs[0].set_xlim(0, 2.0)
axs[0].set_ylim(0, p["ylim"])
axs[0].set_xticks(p["xticks"])
axs[0].set_yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], ["0.0", "", "0.1", "", "0.2", "", "0.3"])
fade_band(axs[0], p["Ltrg"] - 0.2, p["Ltrg"] + 0.2, color="darkgreen")
axs[1].set_xlabel("X")
axs[1].set_ylabel("P(X)")
axs[1].set_xlim(-p["xP"], p["xP"])
axs[1].set_ylim(0, 0.6)
axs[1].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["0.0", "", "0.2", "", "0.4", "", "0.6"])

hB = lambda c, a: [Line2D([0], [0], marker=".", color=c, linestyle="", label=r"$\mathcal{B}=1$", alpha=a),
                   Line2D([0], [0], marker="", color=c, linestyle="-", label=r"$\mathcal{B}=3$", alpha=a)]
axs[1].legend(handles=hB("black", 0.5), loc="upper right")
LEG = dict(handlelength=1.4, handletextpad=0.4, borderpad=0.3, labelspacing=0.25, borderaxespad=0.3)
leg1 = axs[0].legend(handles=hB("black", 0.8), loc="upper left", bbox_to_anchor=(0.3, 1.0), title=r"$|\Delta|$", **LEG)
axs[0].add_artist(leg1)
axs[0].legend(handles=hB("red", 0.8), loc="upper right", title=r"$\mathcal{I}_{dh}$", **LEG)
sm = cm.ScalarMappable(norm=cnorm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=axs[1], pad=0.02, orientation="vertical", location="right").set_label(r"$\lambda$")
plt.tight_layout()
fig.savefig(f"{p['name']}.pdf")
