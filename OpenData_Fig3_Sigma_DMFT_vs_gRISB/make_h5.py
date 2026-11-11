"""Collect DMFT (ED, real axis) and ghost-RISB self-energies / local spectral functions at the same frequencies
into Sigma_DMFT_vs_gRISB.h5 (run once; needs Data/DMFT_data and the raw g-GA tree Data/Nghost{0,2}).
gRISB self-energy:  Sigma(w) = w - e_loc - [ R (w - Lambda)^{-1} R^dag ]^{-1}
"""
import re
import numpy as np
import h5py

DATA = "/mnt/ceph/users/sgiuli/GRISB_PROJECTS/BETHE_HH_PHSYM/Data"
OUT = f"{DATA}/OpenData_Sigma_DMFT_vs_gRISB/Sigma_DMFT_vs_gRISB.h5"
ETA = 5e-2   # Lorentzian broadening of the gRISB real-frequency data: evaluated at omega + i ETA
POINTS = [("hubbard_holstein_u03_l06_w01.h5", 0.30, 0.1, 0.6),     # (DMFT file, U, omega0, lambda)
          ("hubbard_holstein_u045_l06_w05.h5", 0.45, 0.5, 0.6),
          ("hubbard_holstein_u05_l06_w1.h5", 0.50, 1.0, 0.6)]
CPLX = re.compile(r"\(\s*([-+0-9.Ee]+)\s*,\s*([-+0-9.Ee]+)\s*\)")


def cmat(path):
    rows = [[complex(float(a), float(b)) for a, b in CPLX.findall(l)] for l in open(path)]
    m = np.array([r for r in rows if r])
    assert np.abs(m.imag).max() < 1e-10, path
    return m.real


def gf(h, key):
    """TRIQS real-frequency Gf stored as data[w, 0, 0, re/im] on a uniform MeshReFreq."""
    m = h[f"{key}/mesh"]
    w = np.linspace(m["min"][()], m["max"][()], m["size"][()])
    d = h[f"{key}/data"][:, 0, 0, :]
    return w, d[:, 0] + 1j * d[:, 1]


def bethe_g(z):
    """Local Green's function of the Bethe lattice (D=1) at complex z, retarded branch."""
    s = np.sqrt(z - 1) * np.sqrt(z + 1)
    return 2 * (z - s)


def grisb(ngh, U, w0, lam, w):
    d = f"{DATA}/Nghost{ngh}/U{U:.4f}/W{w0:.4f}/L{lam:.4f}"
    R = cmat(f"{d}/R.restart")              # 1 x B
    L = cmat(f"{d}/lambda.restart")         # B x B
    eloc = cmat(f"{d}/h_loc.dat")[0, 0]     # local one-body energy (0: particle-hole symmetric convention)
    ev, U_ = np.linalg.eigh(L)
    r2 = (R @ U_)[0] ** 2                   # g(w) = R (w - Lambda)^{-1} R^T = sum_k r2_k / (w - ev_k)
    z = w + 1j * ETA
    g = (r2[None, :] / (z[:, None] - ev[None, :])).sum(1)
    sigma = z - eloc - 1.0 / g
    # poles of Sigma = zeros of g (between consecutive eigenvalues of Lambda); weight = 1/|g'(w_p)|
    poles, weights = [], []
    for a, b in zip(ev[:-1], ev[1:]):
        if b - a < 1e-10:
            continue
        lo, hi = a + 1e-12 * (b - a), b - 1e-12 * (b - a)
        f = lambda x: (r2 / (x - ev)).sum()
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
        p = 0.5 * (lo + hi)
        poles.append(p)
        weights.append(1.0 / abs((-r2 / (p - ev) ** 2).sum()))
    z0 = 1e-5
    s_p = z0 - eloc - 1.0 / (r2 / (z0 - ev)).sum()
    Z = 1.0 / (1.0 - (2 * s_p) / (2 * z0))   # Sigma is odd at half filling
    A = -bethe_g(z - eloc - sigma).imag / np.pi
    Zfile = float(open(f"{d}/Zdos.sigma.convg").read().split()[0])
    assert abs(Z - Zfile) < 1e-4, (d, Z, Zfile)
    # add the static Hartree term U n/2 = U/2 (half filling): the g-GA runs use the particle-hole symmetric
    # form of the interaction (h_loc = 0, mu = 0), so their Sigma vanishes at omega = 0
    return dict(Sigma=sigma + U / 2, Hartree=U / 2, A=A, R=R[0], Lambda=L, e_loc=eloc, Z=Z,
                Sigma_poles=np.array(poles), Sigma_pole_weights=np.array(weights))


with h5py.File(OUT, "w") as out:
    out.attrs["description"] = (
        "Real-frequency self-energy Sigma(omega) and local spectral function A(omega) of the half-filled "
        "Hubbard-Holstein model on the Bethe lattice (semicircular DOS, half-bandwidth D=1, t=0.5) at T=0 (DMFT: "
        "beta=500): DMFT (exact diagonalization, real axis) vs ghost-RISB/g-GA with B=1 and B=3, on the same "
        "frequency grid. Same (U, omega0, lambda) points as in the Z vs U_eff figure.")
    out.attrs["hamiltonian"] = ("H = -t sum_<ij>s c^dag_is c_js + U sum_i n_iup n_idn "
                                "+ omega0 sum_i b^dag_i b_i + g sum_i (n_i - 1)(b_i + b^dag_i),  lambda = 2 g^2/omega0")
    out.attrs["eta"] = ETA
    out.attrs["energy_unit"] = "D (half-bandwidth)"
    out.attrs["convention"] = (
        "Sigma includes the static Hartree term U n/2 = U/2 for both methods (Re Sigma(0) = U/2): the DMFT Sigma is "
        "stored as computed (chemical potential mu = U/2), the g-GA Sigma (from runs with the particle-hole "
        "symmetric interaction, h_loc = 0, mu = 0) is shifted by U/2 (dataset Hartree). "
        "G_loc(omega) = integral rho(e) / (omega + mu - e - Sigma(omega)), rho the semicircular DOS, mu = U/2.")
    out.attrs["gRISB_self_energy"] = ("Sigma(omega) = omega - e_loc - [ R (omega - Lambda)^{-1} R^dag ]^{-1}, "
                                      "evaluated at omega + i eta with a broadening eta = 5e-2 (attribute eta); for eta -> 0 it is real except for "
                                      "delta-function poles Im Sigma = -pi sum_p w_p delta(omega - omega_p) (Sigma_poles, Sigma_pole_weights). "
                                      "A = -Im G_loc(omega + i eta)/pi with the same broadening. Z is the eta -> 0 value. "
                                      "B=1 gives Sigma = U/2 + omega (1 - 1/Z) (eta -> 0).")
    out.attrs["DMFT_note"] = ("The DMFT real-axis Sigma carries the ED Lorentzian broadening (finite Im Sigma(0)). "
                              "DMFT/G_loc and DMFT/A are rebuilt from DMFT/Sigma as G_Bethe(omega + i eta + mu - Sigma) with "
                              "the same eta as the gRISB data; DMFT/G_loc_stored is g_loc_w of the DMFT file, which "
                              "equals G_Bethe(omega + 0.05 i + mu - Sigma). "
                              "DMFT phonon cutoff 12, bath sites 7.")
    out.attrs["layout"] = ("/U_<U>_omega0_<w0>_lambda_<l>/{omega} ; .../DMFT/{Sigma, G_loc, A, G_loc_stored, Z_from_slope} ; "
                           ".../B<B>/{Sigma, A, R, Lambda, e_loc, Z, Sigma_poles, Sigma_pole_weights}")
    for fname, U, w0, lam in POINTS:
        with h5py.File(f"{DATA}/DMFT_data/{fname}", "r") as h:
            assert abs(h["U"][()] - U) < 1e-12 and abs(h["omega0"][()] - w0) < 1e-12 and abs(h["lambda"][()] - lam) < 1e-9
            w, sig = gf(h, "sigma_w")
            w2, gloc = gf(h, "g_loc_w")
            assert np.allclose(w, w2)
            mu = h["mu"][()]
            attrs = {k: h[k][()] for k in ("beta", "nbath", "nphonons", "g_ph", "converged")}
        g = out.create_group(f"U_{U:g}_omega0_{w0:g}_lambda_{lam:g}")
        g.attrs.update({"U": U, "omega0": w0, "lambda": lam, "g": np.sqrt(lam * w0 / 2)})
        g["omega"] = w
        gd = g.create_group("DMFT")
        gd.attrs.update({"source_file": fname, "mu": mu, **attrs})
        gd["Sigma"] = sig
        # G_loc rebuilt from the DMFT self-energy with the same broadening as the gRISB data
        gl = bethe_g(w + 1j * ETA + mu - sig)
        gd["G_loc"] = gl
        gd["A"] = -gl.imag / np.pi
        gd["G_loc_stored"] = gloc   # g_loc_w as stored in the DMFT file (= G_Bethe(omega + 0.05 i + mu - Sigma))
        assert np.abs(gloc - bethe_g(w + 0.05j + mu - sig)).max() < 1e-8
        i0 = np.argmin(abs(w))
        gd["Z_from_slope"] = 1.0 / (1.0 - np.gradient(sig.real, w)[i0])
        for ngh in (0, 2):
            r = grisb(ngh, U, w0, lam, w)
            gb = g.create_group(f"B{ngh + 1}")
            gb.attrs["N_ghost"] = ngh
            for k, v in r.items():
                gb[k] = v
print("written", OUT)
