"""Self-energy and local spectral function: DMFT vs ghost-RISB (B=1, B=3) at the same real frequencies.
Usage: python plot_Sigma.py   (reads Sigma_DMFT_vs_gRISB.h5, writes Sigma_DMFT_vs_gRISB.pdf)
Rows: Re Sigma, Im Sigma, A(omega); columns: the three (U, omega0) points at lambda = 0.6.
Sigma includes the Hartree term U/2 for both methods. gRISB Sigma and both G_loc (DMFT G_loc rebuilt from the DMFT Sigma) use eta = 5e-2 (omega -> omega + i eta); dotted lines mark the eta -> 0 poles of the B=3 Sigma.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Two-column figure (printed at ~7 in, the single-column figures at ~3.4 in from ~4.3 in): the font sizes of the
# single-column figures are multiplied by FIGW / 8.8 so that all figures have the same printed font size.
FIGW = 12
FS = FIGW / 8.8
plt.rcParams.update({"font.size": 11 * FS, "axes.labelsize": 16 * FS, "axes.titlesize": 14 * FS,
                     "legend.fontsize": 11 * FS, "xtick.labelsize": 12 * FS, "ytick.labelsize": 12 * FS,
                     "xtick.direction": "in", "ytick.direction": "in"})

GREENS = cm.Greens(np.linspace(0.3, 0.9, 3))   # same greens and B line styles as Figs. 1 and 3
STYLE = {"DMFT": dict(color="black", lw=1.6, label="DMFT"),
         "B1": dict(color=GREENS[0], lw=1.6, ls="-", label=r"$\mathcal{B}=1$"),
         "B3": dict(color=GREENS[1], lw=1.6, ls="--", label=r"$\mathcal{B}=3$")}
XMAX = 2.0

h = h5py.File("Sigma_DMFT_vs_gRISB.h5", "r")
keys = sorted(h, key=lambda k: h[k].attrs["omega0"])
fig, axs = plt.subplots(3, len(keys), figsize=(FIGW, 5.6), sharex=True, sharey="row")
for j, k in enumerate(keys):
    g = h[k]
    w = g["omega"][:]
    for src in ("DMFT", "B1", "B3"):
        s = g[f"{src}/Sigma"][:]
        axs[0, j].plot(w, s.real, **STYLE[src])
        axs[1, j].plot(w, s.imag, **STYLE[src])
        axs[2, j].plot(w, g[f"{src}/A"][:], **STYLE[src])
    for p in g["B3/Sigma_poles"][:]:
        for ax in axs[:2, j]:
            ax.axvline(p, color=STYLE["B3"]["color"], lw=0.6, ls=":", alpha=0.6)
    a = g.attrs
    axs[0, j].set_title(rf"$U={a['U']:g}$,  $\omega_0={a['omega0']:g}$")
    axs[2, j].set_xlabel(r"$\omega/D$")
for ax in axs.flat:
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlim(-XMAX, XMAX)
axs[0, 0].set_ylim(-0.5, 1.0)
axs[1, 0].set_ylim(-0.6, 0.05)
axs[2, 0].set_ylim(0, 0.7)
YL = dict(fontsize=13 * FS)   # smaller than axes.labelsize: the rows of this panel are short
axs[0, 0].set_ylabel(r"Re $\Sigma(\omega)$", **YL)
axs[1, 0].set_ylabel(r"Im $\Sigma(\omega)$", **YL)
axs[2, 0].set_ylabel(r"$A(\omega)$", **YL)
fig.align_ylabels(axs[:, 0])
lam = h[keys[0]].attrs["lambda"]
h_, l_ = axs[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=3, frameon=False, title=rf"$\lambda={lam:g}$", alignment="left",
           bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("Sigma_DMFT_vs_gRISB.pdf")
