"""Z vs U_eff/D for B=1 (top) and B=3 (bottom), one column per lambda.
Usage: python plot_Z_vs_Ueff.py   (reads Z_vs_Ueff.h5, writes Z_vs_Ueff.pdf)
"""
import h5py
import matplotlib.pyplot as plt

LAMS = [0.1, 0.2, 0.4, 0.6]
W0S = [0.1, 0.5, 1.0]
COLORS = ["#f08a87", "#d23b3a", "#8a1c1e"]
GRAY = "#8a8984"

# Two-column figure (printed at ~7 in, the single-column figures at ~3.4 in from ~4.3 in): the font sizes of the
# single-column figures are multiplied by FIGW / 8.8 so that all figures have the same printed font size.
FIGW = 16
FS = FIGW / 8.8
plt.rcParams.update({"font.size": 11 * FS, "axes.labelsize": 16 * FS, "axes.titlesize": 14 * FS,
                     "legend.fontsize": 11 * FS, "xtick.labelsize": 12 * FS, "ytick.labelsize": 12 * FS,
                     "xtick.direction": "in", "ytick.direction": "in"})


def line(ax, x, Z, color, label=None):
    Z = Z.copy()
    Z[abs(Z) < 1e-3] = 0.0                      # insulator
    met = Z > 0
    kw = dict(color=color, marker="o", ms=2.5, mec="white", mew=0.5)
    ax.plot(x[met], Z[met], "-", lw=1.4, label=label, **kw)
    ax.plot(x[~met], Z[~met], ls="none", **kw)


h = h5py.File("Z_vs_Ueff.h5", "r")
#fig, axes = plt.subplots(2, 4, figsize=(FIGW, 5.04), sharex=True, sharey=True)
fig, axes = plt.subplots(2, 4, figsize=(FIGW, 6.5), sharex=True, sharey=True)
for i, B in enumerate(["B1", "B3"]):
    ref = h[f"{B}/lambda_0.0"]
    for j, lam in enumerate(LAMS):
        ax = axes[i, j]
        line(ax, ref["U"][:], ref["Z"][:], GRAY, r"$\lambda=0$")
        for w0, c in zip(W0S, COLORS):
            d = h[f"{B}/lambda_{lam:g}/omega0_{w0:g}"]
            line(ax, d["U_eff"][:], d["Z"][:], c, rf"$\omega_0={w0:g}$")
        ax.set_xlim(0, None)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_title(rf"$\lambda = {lam:g}$")
        else:
            ax.set_xlabel(r"$U_\mathrm{eff}/D$")
    axes[i, 0].set_ylabel(rf"$Z$,  $B={B[1]}$")

h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=4, frameon=False)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.13, top=0.83, wspace=0.06, hspace=0.14)
fig.savefig("Z_vs_Ueff.pdf")
