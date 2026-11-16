"""Collect the phonon-displacement distributions P(x) of Fig. 5 (DMFT vs ghost-RISB, B=1 and B=3) into
PX_DMFT_vs_gRISB.h5 (run once; needs the raw data tree BASE).
Usage: python make_h5.py            (reads BASE, writes PX_DMFT_vs_gRISB.h5 next to this script)
       GRISB_HH_ROOT=/path/to/mirror python make_h5.py   (same, with BASE prefixed by a local mirror)
The data are stored exactly as read from the raw files; the plotting offsets/sub-sampling of the paper figure are
applied in plot_PX.py only (documented in the attributes).
"""
import os
from pathlib import Path

import h5py
import numpy as np

ROOT = os.environ.get("GRISB_HH_ROOT", "")
BASE = f"{ROOT}/mnt/home/emohedgas/ceph/grisb_hh"
OUT = Path(__file__).resolve().parent / "PX_DMFT_vs_gRISB.h5"
DIST = "phon_displacement_ifrag1_iph1.dist"
trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy >= 2 / < 2

PANELS = [  # U, g-RISB sub-directory, lambdas, DMFT source (as in the paper plotting script), DMFT x-shift in the plot
    dict(U=0.0, udir="u0", lams=[0.25, 0.75, 0.85, 1.1], bench="ed_h5", bench_xshift=0.0,
         dmft_stride=[6, 6, 6, 2]),
    dict(U=1.0, udir="u5", lams=[1.3, 1.35, 1.4, 1.45], lam_tags=[13, 135, 14, 145], bench="csv",
         bench_xshift=0.02, dmft_stride=[1, 1, 1, 1]),
]


def load_ed_h5(path):
    """ED-DMFT (edipack2triqs) output: TRIQS HDFArchive = plain HDF5, read with h5py."""
    with h5py.File(path, "r") as h:
        x = np.asarray(h["phonon_displacement"])
        px = np.asarray(h["phonon_displacement_probability"])
        meta = {k: h[k][()] for k in ("U", "lambda", "omega0", "g_ph", "beta", "mu", "nphonons", "converged")}
        meta["density"] = h["density"][0]
    return x, px, meta


def load_csv(path):
    dat = np.loadtxt(path, delimiter=",")
    return dat[:, 0], dat[:, 1]


def load_grisb_dist(path):
    """Raw two-column g-RISB file (x, P(x)); no offset applied here."""
    dat = np.loadtxt(path)
    return dat[:, 0], dat[:, 1]


def grisb_input(path):
    """Selected parameters from grisb.input."""
    out = {}
    for line in open(path):
        if "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        if k in ("NGHOST", "ULOC", "DENSITY", "BETA", "NPH", "W0_PH", "LAMBDA_PH", "G_PH", "T_HOP"):
            out[k] = float(v.strip("[] "))
    return out


with h5py.File(OUT, "w") as out:
    out.attrs["description"] = (
        "Fig. 5: phonon-displacement distribution P(x) of the half-filled Hubbard-Holstein model on the Bethe lattice "
        "(semicircular DOS, half-bandwidth D=1, t=0.5), omega0/D=0.2, for U/D=0 and U/D=1 and several lambda: "
        "DMFT benchmark vs ghost-RISB/g-GA with B=1 and B=3 at T=0 (g-GA beta=500).")
    out.attrs["hamiltonian"] = ("H = -t sum_<ij>s c^dag_is c_js + U sum_i n_iup n_idn "
                                "+ omega0 sum_i b^dag_i b_i + g sum_i (n_i - 1)(b_i + b^dag_i),  lambda = 2 g^2/omega0")
    out.attrs["energy_unit"] = "D (half-bandwidth)"
    out.attrs["B_definition"] = ("B = 1 + N_ghost: number of auxiliary (physical + ghost) orbitals per site and spin; "
                                 "B=1 is the plain Gutzwiller/RISB approximation")
    out.attrs["displacement"] = "x = b + b^dag; P(x) is normalized to 1 (integral dx P(x) = 1)"
    out.attrs["DMFT_note"] = (
        "U/D=0: exact-diagonalization DMFT (EDIpack via edipack2triqs, 12 phonons, beta=100), datasets "
        "phonon_displacement / phonon_displacement_probability of hubbard_holstein_u0.0_l<lambda>.h5. "
        "U/D=1: DMFT benchmark curves provided as two-column CSV files u15_lam<tag>.dat (x, P(x)); "
        "solver details of these benchmark curves are not stored in the files.")
    out.attrs["plot_conventions"] = (
        "As in the paper figure (applied in plot_PX.py, NOT in the stored data): "
        "(i) g-RISB curves are plotted as x_i -> P(x_{i+1}) (x[:-1], P[1:]), i.e. shifted by one grid step dx=0.06 to "
        "the left; (ii) the DMFT curves are shifted by DMFT_plot_xshift (0 for U=0, +0.02 for U=1); "
        "(iii) the U=0 DMFT curves are sub-sampled with DMFT_plot_stride (every 6th point, every 2nd for lambda=1.1).")
    out.attrs["layout"] = ("/U_<U>/lambda_<lambda>/{DMFT, B1, B3}/{x, P_x}; attributes on /U_<U>: U, lambdas, "
                           "DMFT_plot_xshift; on /U_<U>/lambda_<lambda>: lambda, DMFT_plot_stride; "
                           "on B1/B3: g-RISB run parameters and source path; on DMFT: source and (ED) run metadata")
    for p in PANELS:
        gu = out.create_group(f"U_{p['U']:g}")
        gu.attrs.update(U=p["U"], lambdas=np.array(p["lams"]), DMFT_plot_xshift=p["bench_xshift"])
        for i, lam in enumerate(p["lams"]):
            gl = gu.create_group(f"lambda_{lam:g}")
            gl.attrs.update({"lambda": lam, "DMFT_plot_stride": p["dmft_stride"][i]})
            gd = gl.create_group("DMFT")
            if p["bench"] == "ed_h5":
                src = f"{BASE}/hubbard_holstein_u{p['U']:.1f}_l{lam}.h5"
                x, px, meta = load_ed_h5(src)
                assert abs(meta["U"] - p["U"]) < 1e-12 and abs(meta["lambda"] - lam) < 1e-9, src
                gd.attrs.update(meta)
                gd.attrs["solver"] = "ED-DMFT (EDIpack / edipack2triqs)"
            else:
                src = f"{BASE}/u5/n1w02/u15_lam{p['lam_tags'][i]}.dat"
                x, px = load_csv(src)
                gd.attrs["solver"] = "DMFT benchmark (two-column CSV)"
            gd.attrs["source_file"] = src.removeprefix(ROOT)
            gd["x"], gd["P_x"] = x, px
            for ngh in (0, 2):
                d = f"{BASE}/{p['udir']}/n1w02/g{ngh}/l{lam}"
                par = grisb_input(f"{d}/grisb.input")
                assert par["NGHOST"] == ngh and abs(par["ULOC"] - p["U"]) < 1e-12 and abs(par["LAMBDA_PH"] - lam) < 1e-9, d
                x, px = load_grisb_dist(f"{d}/{DIST}")
                assert abs(trapz(px, x) - 1) < 1e-6, d
                gb = gl.create_group(f"B{ngh + 1}")
                gb.attrs.update(par)
                gb.attrs["source_dir"] = d.removeprefix(ROOT)
                gb["x"], gb["P_x"] = x, px
print("written", OUT)
