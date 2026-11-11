"""Collect the data of Z_vs_Ueff_byLambda into Data/OpenData_Z_vs_Ueff/Z_vs_Ueff.h5 on Ceph (run once, needs the raw Data tree)."""
import os
import re
import numpy as np
import h5py

DATA = "/mnt/ceph/users/sgiuli/GRISB_PROJECTS/BETHE_HH_PHSYM/Data"
US = [round(0.05 * j, 4) for j in range(81)]
W0S = [0.1, 0.5, 1.0]
LAMS = [0.1, 0.2, 0.4, 0.6]
NPH = {0.1: 60, 0.5: 40, 1.0: 40}


def eta(U, w0):
    x = 2.0 * w0 / U
    return x / (1.0 + x)


CPLX = re.compile(r"\(\s*([-+0-9.Ee]+)\s*,\s*([-+0-9.Ee]+)\s*\)")


def cmat(path):
    """Read a matrix of Fortran complex numbers '(re,im)'; return the real part (imag is checked to be ~0)."""
    rows = []
    for line in open(path):
        v = [complex(float(a), float(b)) for a, b in CPLX.findall(line)]
        if v:
            rows.append(v)
    m = np.array(rows)
    assert np.abs(m.imag).max() < 1e-10, path
    return m.real


def eloc(path):
    """E_loc from the last 'E_kin: ... - Eloc: ... - Etot: ...' line of the run log (full precision)."""
    val = None
    for line in open(path):
        if "Eloc:" in line:
            val = float(line.split("Eloc:")[1].split()[0])
    assert val is not None, path
    return val


def read(ngh, w0, lam):
    out = {k: [] for k in ("U", "Z", "R", "Lambda", "E_loc")}
    for u in US:
        d = f"{DATA}/Nghost{ngh}/U{u:.4f}/W{w0:.4f}/L{lam:.4f}"
        out["U"].append(u)
        out["Z"].append(float(open(f"{d}/Zdos.sigma.convg").read().split()[0]))
        out["R"].append(cmat(f"{d}/R.restart")[0])           # 1 x B  ->  B
        out["Lambda"].append(cmat(f"{d}/lambda.restart"))    # B x B
        out["E_loc"].append(eloc(f"{d}/run_U{u:.4f}.log"))
    return {k: np.array(v) for k, v in out.items()}


def store(g, d):
    for k in ("U", "Z", "R", "Lambda", "E_loc"):
        g[k] = d[k]

with h5py.File(f"{DATA}/OpenData_Z_vs_Ueff/Z_vs_Ueff.h5", "w") as h:
    h.attrs["description"] = (
        "Quasiparticle weight Z vs U for the half-filled Hubbard-Holstein model on the Bethe lattice "
        "(semicircular DOS, half-bandwidth D=1, hopping t=0.5), ghost Gutzwiller approximation, "
        "T=0, particle-hole symmetry enforced.")
    h.attrs["hamiltonian"] = ("H = -t sum_<ij>s c^dag_is c_js + U sum_i n_iup n_idn "
                              "+ omega0 sum_i b^dag_i b_i + g sum_i (n_i - 1)(b_i + b^dag_i)")
    h.attrs["lambda_definition"] = "lambda = 2 g^2 / omega0"
    h.attrs["U_eff_definition"] = "U_eff = U - eta(U) lambda,  eta = (2 omega0/U)/(1 + 2 omega0/U),  U_eff(U=0) = -lambda"
    h.attrs["energy_unit"] = "D (half-bandwidth)"
    h.attrs["B_definition"] = "B = 1 + N_ghost: number of auxiliary (physical + ghost) orbitals per site; B=1 is the plain Gutzwiller approximation"
    h.attrs["insulator"] = "Z below ~1e-3 is numerical zero (Mott insulator)"
    h.attrs["layout"] = ("/B<B>/lambda_0.0/{U,Z,R,Lambda,E_loc}: Hubbard reference (omega0-independent); "
                         "/B<B>/lambda_<l>/omega0_<w>/{U,U_eff,Z,R,Lambda,E_loc}")
    h.attrs["datasets"] = (
        "U: bare Hubbard interaction, shape (nU,). "
        "U_eff: effective static interaction, shape (nU,). "
        "Z: quasiparticle weight from the g-GA self-energy, Z = [1 - dSigma/domega|_{omega=0}]^{-1}, shape (nU,); it equals R^2 only for B=1, for B>1 it is not a simple function of R. "
        "R: g-GA renormalization matrix (physical orbital x B auxiliary orbitals, one spin), shape (nU, B). "
        "Lambda: g-GA quasiparticle Lagrange-multiplier matrix (B x B, one spin), shape (nU, B, B). "
        "E_loc: local energy per site = E_int + E_ph + E_eph (Hubbard + phonon + e-ph terms), shape (nU,); "
        "E_tot = E_kin + E_loc. "
        "R and Lambda are real (imaginary parts are zero) and given in the code's auxiliary-orbital gauge.")
    for ngh in (0, 2):
        gB = h.create_group(f"B{ngh + 1}")
        gB.attrs["N_ghost"] = ngh
        g = gB.create_group("lambda_0.0")
        g.attrs["lambda"] = 0.0
        store(g, read(ngh, 1.0, 0.0))
        for lam in LAMS:
            gl = gB.create_group(f"lambda_{lam:g}")
            gl.attrs["lambda"] = lam
            for w0 in W0S:
                d = read(ngh, w0, lam)
                U = d["U"]
                Ue = U - lam
                Ue[U > 0] = U[U > 0] - eta(U[U > 0], w0) * lam
                g = gl.create_group(f"omega0_{w0:g}")
                g.attrs.update({"omega0": w0, "lambda": lam, "g": np.sqrt(lam * w0 / 2), "N_phonon_max": NPH[w0]})
                store(g, d)
                g["U_eff"] = Ue
print(f"written {DATA}/OpenData_Z_vs_Ueff/Z_vs_Ueff.h5")
