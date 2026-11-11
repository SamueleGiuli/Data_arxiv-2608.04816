"""Self-energy and local spectral function: DMFT vs ghost-RISB (B=1, B=3) at the same real frequencies.
Usage: python plot_Sigma.py   (reads Sigma_DMFT_vs_gRISB.h5, writes Sigma_DMFT_vs_gRISB.pdf)
Rows: Re Sigma, Im Sigma, A(omega); columns: the three (U, omega0) points at lambda = 0.6.
gRISB Im Sigma is a sum of delta functions: their positions are marked by vertical lines.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

STYLE = {"DMFT": dict(color="black", lw=1.6, label="DMFT"),
         "B1": dict(color="#f08a87", lw=1.6, ls="--", label=r"$\mathcal{B}=1$"),
         "B3": dict(color="#8a1c1e", lw=1.6, label=r"$\mathcal{B}=3$")}
XMAX = 2.0

h = h5py.File("Sigma_DMFT_vs_gRISB.h5", "r")
keys = sorted(h, key=lambda k: h[k].attrs["omega0"])
fig, axs = plt.subplots(3, len(keys), figsize=(4 * len(keys), 8), sharex=True, sharey="row")
for j, k in enumerate(keys):
    g = h[k]
    w = g["omega"][:]
    for src in ("DMFT", "B1", "B3"):
        s = g[f"{src}/Sigma"][:]
        re = np.real(s).copy()
        re[np.abs(np.diff(re, prepend=re[0])) > 0.5] = np.nan     # do not draw across gRISB poles
        axs[0, j].plot(w, re, **STYLE[src])
        axs[2, j].plot(w, g[f"{src}/A"][:], **STYLE[src])
    axs[1, j].plot(w, g["DMFT/Sigma"][:].imag, **STYLE["DMFT"])
    for p, wt in zip(g["B3/Sigma_poles"][:], g["B3/Sigma_pole_weights"][:]):
        axs[1, j].axvline(p, **{**STYLE["B3"], "label": None}, alpha=0.8)
        axs[0, j].axvline(p, color=STYLE["B3"]["color"], lw=0.6, ls=":", alpha=0.6)
    a = g.attrs
    axs[0, j].set_title(rf"$U/D={a['U']:g}$,  $\omega_0/D={a['omega0']:g}$,  $\lambda/D={a['lambda']:g}$", fontsize=11)
    axs[0, j].text(0.03, 0.04, f"Z: DMFT {g['DMFT/Z_from_slope'][()]:.2f}, B=1 {g['B1/Z'][()]:.2f}, "
                   f"B=3 {g['B3/Z'][()]:.2f}", transform=axs[0, j].transAxes, fontsize=8)
    axs[2, j].set_xlabel(r"$\omega/D$")
for ax in axs.flat:
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlim(-XMAX, XMAX)
axs[0, 0].set_ylim(-1.5, 1.5)
axs[1, 0].set_ylim(-0.6, 0.05)
axs[2, 0].set_ylim(0, 1.0)
axs[0, 0].set_ylabel(r"Re $\Sigma(\omega)$")
axs[1, 0].set_ylabel(r"Im $\Sigma(\omega)$")
axs[2, 0].set_ylabel(r"$A(\omega)$")
axs[0, 0].legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("Sigma_DMFT_vs_gRISB.pdf")
