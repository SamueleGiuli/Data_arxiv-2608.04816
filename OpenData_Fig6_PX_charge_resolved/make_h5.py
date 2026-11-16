"""Collect the total and charge-resolved phonon-displacement distributions of Fig. 6 (ED-DMFT vs ghost-RISB B=3)
into PX_charge_resolved.h5 (run once; needs the raw data under /mnt/home/emohedgas).
Usage: python make_h5.py            (writes PX_charge_resolved.h5 next to this script)
       GRISB_HH_ROOT=/path/to/mirror python make_h5.py   (same, with the raw paths prefixed by a local mirror)
Extraction follows the paper plotting script; the plotting shifts are stored as attributes and applied in
plot_PX_charge_resolved.py only.
"""
import os
from pathlib import Path

import h5py
import numpy as np

ROOT = os.environ.get("GRISB_HH_ROOT", "")
HOME = f"{ROOT}/mnt/home/emohedgas"
OUT = Path(__file__).resolve().parent / "PX_charge_resolved.h5"
DIST = "phon_displacement_ifrag1_iph1"
trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy >= 2 / < 2

PANELS = [
    dict(U=0.0, lam=1.1, n=1.0,
         ed_file=f"{HOME}/ceph/grisb_hh/hubbard_holstein_u0.0_l1.1.h5",
         grisb_dir=f"{HOME}/ceph/grisb_hh/u0/n1w02/g2/l1.1",
         ed_shift=0.031, grisb_shift=0.0),
    dict(U=5.0, lam=2.2, n=0.7,
         ed_file=f"{HOME}/edi/hubbard_holstein_u5_l2.2.h5",
         grisb_dir=f"{HOME}/ceph/grisb_hh/u5/n0.7/l2.2",
         ed_shift=-2 * 0.99, grisb_shift=-0.99),
]


# ---------------------------------------------------------------- ED-DMFT (plain HDF5 written by TRIQS HDFArchive)
def read_first_available(h, names, required=True):
    for name in names:
        if name in h:
            return np.asarray(h[name])
    if required:
        raise KeyError(f"None of {names} found; available: {list(h.keys())}")
    return None


def load_ed_data(path):
    with h5py.File(path, "r") as h:
        x = read_first_available(h, ["phonon_displacement", "displacement"]).squeeze()
        total = read_first_available(h, ["phonon_displacement_probability", "displacement_pdf"]).squeeze()
        joint = read_first_available(h, ["phonon_displacement_probability_by_occupancy",
                                         "phonon_displacement_pdf_by_occupancy",
                                         "displacement_pdf_by_occupancy"])
        sectors = read_first_available(h, ["phonon_displacement_occupancy_sectors", "phonon_occupancy_sectors",
                                           "displacement_occupancy_sectors", "occupancy_sectors"], required=False)
        meta = {k: h[k][()] for k in ("U", "lambda", "omega0", "g_ph", "beta", "mu", "nphonons", "converged")
                if k in h}
        meta["density"] = h["density"][0]
    if joint.ndim == 2 and joint.shape[0] != len(x) and joint.shape[1] == len(x):
        joint = joint.T
    assert joint.ndim == 2 and joint.shape[0] == len(x) and len(total) == len(x), path
    sectors = np.arange(joint.shape[1]) if sectors is None else np.asarray(sectors).squeeze()
    col = {int(s): c for c, s in enumerate(sectors)}
    joint = np.stack([joint[:, col[s]] for s in (0, 1, 2)], axis=1)   # columns ordered n = 0, 1, 2
    return x, total, joint, meta


# ---------------------------------------------------------------- g-RISB
def load_grisb_dist(path):
    data = np.loadtxt(path)
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def add_distributions(distributions):
    """Sum distributions on the grid of the first one (interpolating if grids differ)."""
    x_ref, comb = distributions[0]
    comb = np.array(comb, dtype=float, copy=True)
    for x, p in distributions[1:]:
        if len(x) == len(x_ref) and np.allclose(x, x_ref, rtol=1e-10, atol=1e-12):
            comb += p
        else:
            comb += np.interp(x_ref, x, p, left=0.0, right=0.0)
    return x_ref, comb


def grisb_input(path):
    out = {}
    for line in open(path):
        if "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        if k in ("NGHOST", "ULOC", "DENSITY", "BETA", "NPH", "W0_PH", "LAMBDA_PH", "G_PH", "T_HOP", "MU"):
            out[k] = float(v.strip("[] "))
    return out


with h5py.File(OUT, "w") as out:
    out.attrs["description"] = (
        "Fig. 6: total and charge-resolved phonon-displacement distribution P(X) and joint distributions P(X, n) "
        "(n = 0, 1, 2 electrons on the impurity site) of the Hubbard-Holstein model on the Bethe lattice "
        "(semicircular DOS, half-bandwidth D=1, t=0.5), omega0/D=0.2: ED-DMFT vs ghost-RISB/g-GA with B=3. "
        "Top: U/D=0, lambda/D=1.1, n=1. Bottom: U/D=5, lambda/D=2.2, n=0.7.")
    out.attrs["hamiltonian"] = ("H = -t sum_<ij>s c^dag_is c_js + U sum_i n_iup n_idn "
                                "+ omega0 sum_i b^dag_i b_i + g sum_i (n_i - 1)(b_i + b^dag_i),  lambda = 2 g^2/omega0")
    out.attrs["energy_unit"] = "D (half-bandwidth)"
    out.attrs["B_definition"] = "B = 1 + N_ghost: number of auxiliary (physical + ghost) orbitals per site and spin"
    out.attrs["displacement"] = "X = b + b^dag; P(X) normalized to 1, P(X) = sum_n P(X, n)"
    out.attrs["DMFT_note"] = ("ED-DMFT (EDIpack via edipack2triqs). Datasets phonon_displacement, "
                              "phonon_displacement_probability, phonon_displacement_probability_by_occupancy "
                              "(columns reordered to n = 0, 1, 2). Run metadata in the attributes of each DMFT group.")
    out.attrs["gRISB_charge_resolution"] = (
        "P_X_from<k> (k=1..4) are the g-RISB distributions resolved on the four local Fock states (files "
        "phon_displacement_ifrag1_iph1_from<k>.dist). Charge sectors as in the paper figure: "
        "n=0: from1;  n=1: from2 + from3;  n=2: from4 (stored as P_X_n).")
    out.attrs["plot_conventions"] = (
        "Applied in plot_PX_charge_resolved.py only: the DMFT X grid is shifted by DMFT_plot_xshift and the g-RISB "
        "X grid by gRISB_plot_xshift (0.031 / 0 for the top panel, -1.98 / -0.99 for the bottom panel); "
        "g-RISB markers on every 2nd grid point.")
    out.attrs["layout"] = ("/U_<U>_lambda_<lambda>_n_<n>/DMFT/{X, P_X, P_X_n}, "
                           ".../B3/{X, P_X, P_X_n, P_X_from1..4}; P_X_n has shape (nX, 3) for n = 0, 1, 2")
    for p in PANELS:
        g = out.create_group(f"U_{p['U']:g}_lambda_{p['lam']:g}_n_{p['n']:g}")
        g.attrs.update(U=p["U"], lambda_=p["lam"], density=p["n"],
                       DMFT_plot_xshift=p["ed_shift"], gRISB_plot_xshift=p["grisb_shift"])
        g.attrs["lambda"] = g.attrs.pop("lambda_")

        x, total, joint, meta = load_ed_data(p["ed_file"])
        assert abs(meta["U"] - p["U"]) < 1e-12 and abs(meta["lambda"] - p["lam"]) < 1e-9, p["ed_file"]
        gd = g.create_group("DMFT")
        gd.attrs.update(meta)
        gd.attrs["source_file"] = p["ed_file"].removeprefix(ROOT)
        gd["X"], gd["P_X"], gd["P_X_n"] = x, total, joint

        d = p["grisb_dir"]
        par = grisb_input(f"{d}/grisb.input")
        assert par["NGHOST"] == 2 and abs(par["ULOC"] - p["U"]) < 1e-12 and abs(par["LAMBDA_PH"] - p["lam"]) < 1e-9
        assert abs(par["DENSITY"] - p["n"]) < 1e-12, d
        xt, pt = load_grisb_dist(f"{d}/{DIST}.dist")
        frm = {k: load_grisb_dist(f"{d}/{DIST}_from{k}.dist") for k in range(1, 5)}
        for k in frm:
            assert np.allclose(frm[k][0], xt), (d, k)
        sec = [frm[1], add_distributions([frm[2], frm[3]]), frm[4]]
        assert np.allclose(sum(s[1] for s in sec), pt, atol=1e-8), d
        assert abs(trapz(pt, xt) - 1) < 1e-6, d
        gb = g.create_group("B3")
        gb.attrs.update(par)
        gb.attrs["source_dir"] = d.removeprefix(ROOT)
        gb["X"], gb["P_X"] = xt, pt
        gb["P_X_n"] = np.stack([s[1] for s in sec], axis=1)
        for k in frm:
            gb[f"P_X_from{k}"] = frm[k][1]
print("written", OUT)
