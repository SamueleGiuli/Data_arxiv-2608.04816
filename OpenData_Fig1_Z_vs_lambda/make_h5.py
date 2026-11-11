"""Collect the data of Fig. 1 (Z(lambda)/Z(0), omega0=0.2D, half filling) into Fig1_Z_vs_lambda.h5.
Sources: BETHE_GRISBOLT/Analysis/U{05,10,15,20}/ZL_{ngh0,ngh2,ngh4,DMFT}_wph0.2.dat (2 columns: lambda, Z)."""
import numpy as np
import h5py

SRC = "/mnt/home/sgiuli/MyProjects/GRISB_PROJECTS/BETHE_GRISBOLT/Analysis"
OUT = "/mnt/ceph/users/sgiuli/GRISB_PROJECTS/BETHE_GRISBOLT/Data/OpenData_Fig1_Z_vs_lambda/Fig1_Z_vs_lambda.h5"
US = [0.5, 1.0, 1.5, 2.0]

with h5py.File(OUT, "w") as h:
    h.attrs["description"] = (
        "Fig. 1: quasiparticle weight Z vs phonon-mediated attraction lambda for the half-filled "
        "Hubbard-Holstein model on the Bethe lattice (semicircular DOS, half-bandwidth D=1), "
        "omega0 = 0.2 D, U = 0.5, 1.0, 1.5, 2.0 D. ghost-RISB with B = 1, 3, 5 compared with DMFT.")
    h.attrs["hamiltonian"] = ("H = -t sum_<ij>s c^dag_is c_js + U sum_i n_iup n_idn "
                              "+ omega0 sum_i b^dag_i b_i + g sum_i (n_i - 1)(b_i + b^dag_i)")
    h.attrs["lambda_definition"] = "lambda = 2 g^2 / omega0"
    h.attrs["omega0"] = 0.2
    h.attrs["energy_unit"] = "D (half-bandwidth)"
    h.attrs["B_definition"] = "B = 1 + N_ghost; B=1 is the plain Gutzwiller/RISB approximation"
    h.attrs["plotted_quantity"] = "Z(lambda)/Z(lambda=0): each curve divided by its first point"
    h.attrs["layout"] = ("/U_<U>/B<B>/{lambda, Z}: ghost-RISB, Z is the absolute quasiparticle weight; "
                         "/U_<U>/DMFT/{lambda, Z_ratio}: DMFT Z(lambda)/Z(0), digitized from Ref. [33] of the paper")
    for U in US:
        uf = f"U{int(round(U * 10)):02d}"
        gU = h.create_group(f"U_{U:g}")
        gU.attrs["U"] = U
        for ngh in (0, 2, 4):
            lam, Z = np.loadtxt(f"{SRC}/{uf}/ZL_ngh{ngh}_wph0.2.dat", unpack=True)
            g = gU.create_group(f"B{ngh + 1}")
            g.attrs["N_ghost"] = ngh
            g["lambda"] = lam
            g["Z"] = Z
        lam, Zr = np.loadtxt(f"{SRC}/{uf}/ZL_DMFT_wph0.2.dat", unpack=True)
        g = gU.create_group("DMFT")
        g.attrs["source"] = "digitized from Ref. [33] of the paper"
        g["lambda"] = lam
        g["Z_ratio"] = Zr
print("written", OUT)
