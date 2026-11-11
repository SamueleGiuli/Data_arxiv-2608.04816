"""Z vs U_eff/D for B=1 (top) and B=3 (bottom), one column per lambda.
Usage: python plot_Z_vs_Ueff.py   (reads Z_vs_Ueff.h5, writes Z_vs_Ueff.pdf)
"""
import h5py
import matplotlib.pyplot as plt

LAMS = [0.1, 0.2, 0.4, 0.6]
W0S = [0.1, 0.5, 1.0]
COLORS = ["#f08a87", "#d23b3a", "#8a1c1e"]
GRAY = "#8a8984"

plt.rcParams.update({"font.size": 11, "axes.labelsize": 16, "axes.titlesize": 14, "legend.fontsize": 11,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})


def line(ax, x, Z, color, label=None):
    Z = Z.copy()
    Z[abs(Z) < 1e-3] = 0.0                      # insulator
    met = Z > 0
    kw = dict(color=color, marker="o", ms=2.5, mec="white", mew=0.5)
    ax.plot(x[met], Z[met], "-", lw=1.4, label=label, **kw)
    ax.plot(x[~met], Z[~met], ls="none", **kw)


h = h5py.File("Z_vs_Ueff.h5", "r")
#fig, axes = plt.subplots(2, 4, figsize=(16, 5.04), sharex=True, sharey=True)
fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), sharex=True, sharey=True)
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
for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)

h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("Z_vs_Ueff.pdf")
