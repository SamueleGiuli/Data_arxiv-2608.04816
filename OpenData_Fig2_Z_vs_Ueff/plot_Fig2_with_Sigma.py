"""Fig. 2 stacked: Z vs U_eff/D (top, B=1 and B=3, one column per lambda) and the real-frequency self-energy /
spectral function, DMFT vs ghost-RISB, at lambda = 0.6 (bottom, one column per omega0).
Usage: python plot_Fig2_with_Sigma.py
   reads Z_vs_Ueff.h5 and ../OpenData_Sigma_DMFT_vs_gRISB/Sigma_DMFT_vs_gRISB.h5, writes Fig2_with_Sigma.pdf
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Two-column figure (printed at ~7 in, the single-column figures at ~3.4 in from ~4.3 in): the font sizes of the
# single-column figures are multiplied by FIGW / 8.8 so that all figures have the same printed font size.
FIGW = 16
FS = FIGW / 8.8
plt.rcParams.update({"font.size": 11 * FS, "axes.labelsize": 16 * FS, "axes.titlesize": 14 * FS,
                     "legend.fontsize": 11 * FS, "xtick.labelsize": 12 * FS, "ytick.labelsize": 12 * FS,
                     "xtick.direction": "in", "ytick.direction": "in"})

# ---- top: Z vs U_eff (same as plot_Z_vs_Ueff.py) ----
LAMS = [0.1, 0.2, 0.4, 0.6]
W0S = [0.1, 0.5, 1.0]
REDS = ["#f08a87", "#d23b3a", "#8a1c1e"]
GRAY = "#8a8984"
# ---- bottom: Sigma, A (same as plot_Sigma.py) ----
GREENS = cm.Greens(np.linspace(0.3, 0.9, 3))   # same greens and B line styles as Figs. 1 and 3
STYLE = {"DMFT": dict(color="black", lw=1.6, label="DMFT"),
         "B1": dict(color=GREENS[0], lw=1.6, ls="-", label=r"$\mathcal{B}=1$"),
         "B3": dict(color=GREENS[1], lw=1.6, ls="--", label=r"$\mathcal{B}=3$")}
XMAX = 2.0


def zline(ax, x, Z, color, label=None):
    Z = Z.copy()
    Z[abs(Z) < 1e-3] = 0.0                      # insulator
    met = Z > 0
    kw = dict(color=color, marker="o", ms=2.5, mec="white", mew=0.5)
    ax.plot(x[met], Z[met], "-", lw=1.4, label=label, **kw)
    ax.plot(x[~met], Z[~met], ls="none", **kw)


fig = plt.figure(figsize=(FIGW, 14), layout="constrained")
top, bot = fig.subfigures(2, 1, height_ratios=[6.5, 7.5], hspace=0.04)

# ===== top =====
h = h5py.File("Z_vs_Ueff.h5", "r")
axes = top.subplots(2, 4, sharex=True, sharey=True)
for i, B in enumerate(["B1", "B3"]):
    ref = h[f"{B}/lambda_0.0"]
    for j, lam in enumerate(LAMS):
        ax = axes[i, j]
        zline(ax, ref["U"][:], ref["Z"][:], GRAY, r"$\lambda=0$")
        for w0, c in zip(W0S, REDS):
            d = h[f"{B}/lambda_{lam:g}/omega0_{w0:g}"]
            zline(ax, d["U_eff"][:], d["Z"][:], c, rf"$\omega_0={w0:g}$")
        ax.set_xlim(0, None)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_title(rf"$\lambda = {lam:g}$")
        else:
            ax.set_xlabel(r"$U_\mathrm{eff}/D$")
    axes[i, 0].set_ylabel(rf"$Z$,  $\mathcal{{B}}={B[1]}$")
h_, l_ = axes[0, 0].get_legend_handles_labels()
top.legend(h_, l_, loc="outside upper center", ncol=4, frameon=False)

# ===== bottom =====
hs = h5py.File("../OpenData_Sigma_DMFT_vs_gRISB/Sigma_DMFT_vs_gRISB.h5", "r")
keys = sorted(hs, key=lambda k: hs[k].attrs["omega0"])
axs = bot.subplots(3, len(keys), sharex=True, sharey="row")
for j, k in enumerate(keys):
    g = hs[k]
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
bot.align_ylabels(axs[:, 0])
h_, l_ = axs[0, 0].get_legend_handles_labels()
bot.legend(h_, l_, loc="outside upper center", ncol=3, frameon=False,
           title=rf"$\lambda={hs[keys[0]].attrs['lambda']:g}$")
fig.savefig("Fig2_with_Sigma.pdf")
