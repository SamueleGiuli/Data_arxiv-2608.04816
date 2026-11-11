"""Collect the data of Figs. 6 and 7 (s-wave superconductivity in the Hubbard-Holstein model) into an h5 file.
Usage: python make_h5.py 6   ->  ../OpenData_Fig6_SC_n1.0/Fig6_SC_n1.0.h5
       python make_h5.py 7   ->  ../OpenData_Fig7_SC_n0.5/Fig7_SC_n0.5.h5
Needs the raw data tree HH_SC_DATA (run once)."""
import os
import re
import sys
import numpy as np
import h5py

RAW = "/mnt/ceph/users/sgiuli/GRISB_PROJECTS/BETHE_GRISBOLT/HH_SC_DATA"
OUT = "/mnt/ceph/users/sgiuli/GRISB_PROJECTS/BETHE_HH_PHSYM/Data"
FIGS = {  # U, omega0, density, lambda grid (as plotted in the paper)
    6: dict(U=0.0, w0=1.0, n=1.0, lams=np.round(np.arange(1, 21) * 0.1, 4), name="Fig6_SC_n1.0"),
    7: dict(U=1.0, w0=0.5, n=0.5, lams=np.round(np.arange(1, 25) * 0.1, 4), name="Fig7_SC_n0.5"),
}
NPH = 40
CPLX = re.compile(r"\(\s*([-+0-9.Ee]+)\s*,\s*([-+0-9.Ee]+)\s*\)")


def cmat(path):
    """Matrix of Fortran complex numbers '(re,im)', one row per line; return the real part (imag checked ~0)."""
    rows = [[complex(float(a), float(b)) for a, b in CPLX.findall(l)] for l in open(path)]
    m = np.array([r for r in rows if r])
    assert np.abs(m.imag).max() < 1e-10, path
    return m.real


def dist(d, tag=""):
    x, p = np.loadtxt(f"{d}/phon_displacement_ifrag1_iph1{tag}.dist", unpack=True)
    return x, p


def read(B, f):
    out = {k: [] for k in ("lambda", "Delta", "I_dh", "P_X", "P_X_0", "P_X_2", "P_X_1a", "P_X_1b",
                           "density_matrix", "R", "Lambda", "E_kin", "E_loc", "E_tot", "sc_residual")}
    for lam in f["lams"]:
        d = f"{RAW}/Nghost{B - 1}/U{f['U']:.4f}/L{lam:.4f}/W{f['w0']:.4f}/DENS{f['n']:.4f}"
        dm = np.loadtxt(f"{d}/dimp_real.dat")
        x, P = dist(d)
        _, P1a = dist(d, "_from1")
        _, P2 = dist(d, "_from2")   # physical doubly-occupied impurity state
        _, P0 = dist(d, "_from3")   # physical empty impurity state
        _, P1b = dist(d, "_from4")
        n_check = np.trapezoid(P1a + P1b + 2 * P2, x)
        assert abs(n_check - f["n"]) < 2e-3, (d, n_check)
        ek, el, et = map(float, open(f"{d}/energy.convg").read().split()[-3:])
        out["lambda"].append(lam)
        out["Delta"].append(abs(dm[0, 1]))
        out["I_dh"].append(np.trapezoid(np.sqrt(np.clip(P0 * P2, 0, None)), x))
        for k, v in (("P_X", P), ("P_X_0", P0), ("P_X_2", P2), ("P_X_1a", P1a), ("P_X_1b", P1b)):
            out[k].append(v)
        out["density_matrix"].append(dm)
        out["R"].append(cmat(f"{d}/R.restart"))
        out["Lambda"].append(cmat(f"{d}/lambda.restart"))
        out["E_kin"].append(ek); out["E_loc"].append(el); out["E_tot"].append(et)
        out["sc_residual"].append(float(open(f"{d}/error.SC.convg").read().split()[-1]))
    out = {k: np.array(v) for k, v in out.items()}
    out["X"] = x
    return out


fig = int(sys.argv[1])
f = FIGS[fig]
path = f"{OUT}/OpenData_{f['name']}/{f['name']}.h5"
with h5py.File(path, "w") as h:
    h.attrs["description"] = (
        f"Fig. {fig}: s-wave superconducting order parameter |Delta|, double-hole overlap I_dh and phonon-displacement "
        f"distribution P(X) vs lambda for the Hubbard-Holstein model on the Bethe lattice (semicircular DOS, "
        f"half-bandwidth D=1, hopping t=0.5), ghost Gutzwiller/RISB (g-GA) at T=0. "
        f"U/D={f['U']:g}, omega0/D={f['w0']:g}, density n={f['n']:g}.")
    h.attrs["hamiltonian"] = ("H = -t sum_<ij>s c^dag_is c_js + U sum_i n_iup n_idn "
                              "+ omega0 sum_i b^dag_i b_i + g sum_i (n_i - 1)(b_i + b^dag_i)")
    h.attrs["lambda_definition"] = "lambda = 2 g^2 / omega0"
    h.attrs["U"] = f["U"]; h.attrs["omega0"] = f["w0"]; h.attrs["density"] = f["n"]
    h.attrs["N_phonon_max"] = NPH
    h.attrs["energy_unit"] = "D (half-bandwidth)"
    h.attrs["B_definition"] = ("B = 1 + N_ghost: number of auxiliary (physical + ghost) orbitals per site and spin; "
                               "B=1 is the plain Gutzwiller/RISB approximation")
    h.attrs["method"] = (
        "Superconductivity is computed through the attractive/repulsive particle-hole transformation on spin down, "
        "c_idn -> c^dag_idn. The transformed model has interaction -U, the pairing amplitude becomes the transverse "
        "magnetization and the physical density n becomes the magnetization m_z = n - 1 (kept fixed by fitting a "
        "uniform field). Spin-mixing (non-S_z conserving) g-GA solutions give the superconducting state. "
        "Each lambda point was first converged with a small static pairing field (h_x=0.01) and then reconverged with "
        "the field removed; the stored data are the field-free solutions.")
    h.attrs["layout"] = "/B<B>/{lambda, Delta, I_dh, X, P_X, P_X_0, P_X_2, P_X_1a, P_X_1b, density_matrix, R, Lambda, E_kin, E_loc, E_tot, sc_residual}"
    h.attrs["datasets"] = (
        "lambda: shape (nL,). "
        "Delta: |<c_idn c_iup>|, the s-wave order parameter = |off-diagonal element of the local density matrix| in the transformed basis, shape (nL,). "
        "I_dh: integral dX sqrt(P_0(X) P_2(X)), overlap of the phonon distributions of the empty and doubly-occupied impurity states, shape (nL,) "
        "(tiny negative round-off of P_0 P_2 is set to zero). "
        "X: phonon displacement grid, X = b + b^dag, shape (nX,). "
        "P_X: phonon displacement distribution P(X), normalized to 1, shape (nL, nX). "
        "P_X_0, P_X_2, P_X_1a, P_X_1b: P(X) resolved on the physical impurity charge state "
        "(empty, doubly occupied, the two singly occupied), P_X = P_X_0 + P_X_2 + P_X_1a + P_X_1b, shape (nL, nX). "
        "density_matrix: local density matrix of the transformed model, basis (up, dn~) with dn~ the particle-hole transformed spin down, shape (nL, 2, 2). "
        "R: g-GA renormalization matrix in the transformed basis (2 physical spin-orbitals x 2B auxiliary orbitals), shape (nL, 2, 2B). "
        "Lambda: g-GA quasiparticle Lagrange-multiplier matrix (2B x 2B), shape (nL, 2B, 2B). "
        "R and Lambda are real and given in the code's auxiliary-orbital gauge. "
        "E_kin, E_loc, E_tot: kinetic, local (interaction + phonon + e-ph) and total energy per site of the transformed model "
        "(5 decimals), shape (nL,). "
        "sc_residual: final self-consistency residual (target 1e-5), shape (nL,).")
    for B in (1, 3):
        g = h.create_group(f"B{B}")
        g.attrs["N_ghost"] = B - 1
        for k, v in read(B, f).items():
            g[k] = v
print("written", path)
