"""Collect the data of Fig. 3 (m*/m = 1/Z vs omega0 at U=5D, n=0.9,0.7,0.5) into Fig3_mstar_vs_w0.h5.
Sources: ghost-RISB  Data_Z/Nghost{0,2,4}/DENS<n>/U5.0000/L<lambda>/W<omega0>/  (runs: BETHE_GRISBOLT/PhaseDiag/D<n>_L<l>_Ng{0,2,4_rerun})
         DMFT        BETHE_GRISBOLT/Analysis/invZ_L<lambda>_n<n>_dmft.txt"""
import re
import numpy as np
import h5py

DZ = "/mnt/ceph/users/sgiuli/GRISB_PROJECTS/BETHE_GRISBOLT/Data_Z"
DMFT = "/mnt/home/sgiuli/MyProjects/GRISB_PROJECTS/BETHE_GRISBOLT/Analysis"
OUT = f"{DZ}/OpenData_Fig3_mstar_vs_w0/Fig3_mstar_vs_w0.h5"
U = 5.0
DENS = [0.9, 0.7, 0.5]
LAMS = [0.1, 0.5, 1.0]
W0 = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
      10.0, 15.0, 20.0]
W_DMFT = {0.9: [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          0.5: [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0]}
W_DMFT[0.7] = W_DMFT[0.9]
CPLX = re.compile(r"\(\s*([-+0-9.EeDd]+)\s*,\s*([-+0-9.EeDd]+)\s*\)")


def cmat(path):
    rows = [[complex(float(a.replace("D", "E")), float(b.replace("D", "E"))) for a, b in CPLX.findall(l)]
            for l in open(path)]
    return np.array([r for r in rows if r])


def num(path):
    return float(open(path).read().split()[0])


def nph(w0):
    return 40 if w0 < 0.5 else 30 if w0 < 1.0 else 20 if w0 < 5.0 else 10


with h5py.File(OUT, "w") as h:
    h.attrs["description"] = (
        "Fig. 3: mass renormalization m*/m = 1/Z vs phonon frequency omega0 for the doped Hubbard-Holstein model "
        "on the Bethe lattice (semicircular DOS, half-bandwidth D=1, hopping t=0.5) at U = 5 D, electron densities "
        "n = 0.9, 0.7, 0.5 and lambda = 0.1, 0.5, 1.0 D. ghost-RISB with B = 1, 3, 5 (T=0) compared with DMFT.")
    h.attrs["hamiltonian"] = ("H = -t sum_<ij>s c^dag_is c_js + U sum_i n_iup n_idn - mu sum_i n_i "
                              "+ omega0 sum_i b^dag_i b_i + g sum_i (n_i - n)(b_i + b^dag_i)")
    h.attrs["lambda_definition"] = "lambda = 2 g^2 / omega0"
    h.attrs["U"] = U
    h.attrs["energy_unit"] = "D (half-bandwidth)"
    h.attrs["B_definition"] = "B = 1 + N_ghost; B=1 is the plain Gutzwiller/RISB approximation"
    h.attrs["phonon_cutoff"] = "N_phonon_max = 10 (omega0>=5), 20 (1<=omega0<5), 30 (0.5<=omega0<1), 40 (omega0<0.5)"
    h.attrs["layout"] = ("/n_<n>/lambda_<l>/B<B>/{omega0, Z, mstar, R, Lambda, E_loc, density, mu}: ghost-RISB; "
                         "/n_<n>/lambda_<l>/DMFT/{omega0, mstar}: DMFT (data of Ref. [34] of the paper)")
    h.attrs["datasets"] = (
        "omega0: phonon frequency, shape (nw,). "
        "Z: quasiparticle weight from the g-RISB self-energy, Z = [1 - dSigma/domega|_{omega=0}]^{-1}, shape (nw,); "
        "it equals R^2 only for B=1. mstar = 1/Z. "
        "R: g-RISB renormalization matrix (physical orbital x B auxiliary orbitals, one spin), shape (nw, B). "
        "Lambda: g-RISB quasiparticle Lagrange-multiplier matrix (B x B, one spin), shape (nw, B, B). "
        "R and Lambda are given in the code's auxiliary-orbital gauge; the real part is stored, the largest "
        "imaginary part (numerical noise) is in the group attribute max_abs_imag. "
        "E_loc: local energy per site = E_int + E_ph + E_eph, shape (nw,). "
        "density: electron density per site (target n, fixed by the chemical potential), mu: chemical potential.")
    for D in DENS:
        gD = h.create_group(f"n_{D:g}")
        gD.attrs["density"] = D
        for L in LAMS:
            gL = gD.create_group(f"lambda_{L:g}")
            gL.attrs["lambda"] = L
            for ngh in (0, 2, 4):
                Z, R, Lm, E, n, mu = [], [], [], [], [], []
                for w in W0:
                    p = f"{DZ}/Nghost{ngh}/DENS{D:.4f}/U{U:.4f}/L{L:.4f}/W{w:.4f}"
                    Z.append(num(f"{p}/Zdos.sigma.convg"))
                    R.append(cmat(f"{p}/R.restart")[0])
                    Lm.append(cmat(f"{p}/lambda.restart"))
                    E.append(num(f"{p}/Eint.last") + num(f"{p}/Eph.last") + num(f"{p}/Eeph.last"))
                    n.append(num(f"{p}/dens.last"))
                    mu.append(num(f"{p}/mu.convg"))
                R, Lm, Z = np.array(R), np.array(Lm), np.array(Z)
                g = gL.create_group(f"B{ngh + 1}")
                g.attrs.update({"N_ghost": ngh,
                                "max_abs_imag": max(np.abs(R.imag).max(), np.abs(Lm.imag).max())})
                g["omega0"] = np.array(W0)
                g["N_phonon_max"] = np.array([nph(w) for w in W0])
                g["Z"] = Z
                g["mstar"] = 1 / Z
                g["R"] = R.real
                g["Lambda"] = Lm.real
                g["E_loc"] = np.array(E)
                g["density"] = np.array(n)
                g["mu"] = np.array(mu)
            g = gL.create_group("DMFT")
            g.attrs["source"] = "Ref. [34] of the paper"
            g["omega0"] = np.array(W_DMFT[D])
            g["mstar"] = np.loadtxt(f"{DMFT}/invZ_L{L:.2f}_n{int(round(D * 10)):02d}_dmft.txt")
            assert len(g["mstar"]) == len(g["omega0"])
print("written", OUT)
